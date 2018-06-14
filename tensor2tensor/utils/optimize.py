# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import adafactor
from tensor2tensor.utils import multistep_optimizer
from tensor2tensor.utils import yellowfin

import tensorflow as tf

from tensorflow.python.framework import dtypes


def optimize(loss, learning_rate, hparams, use_tpu=False):
  """Minimize loss."""
  loss = weight_decay_and_noise(loss, hparams, learning_rate)
  loss = tf.identity(loss, name="total_loss")
  log_variable_sizes(verbose=hparams.summarize_vars)
  diet_vars = [
      v for v in tf.global_variables() if v.dtype == dtypes.float16_ref
  ]
  log_variable_sizes(
      diet_vars, "Diet Variables", verbose=hparams.summarize_vars)
  opt = ConditionalOptimizer(hparams.optimizer, learning_rate, hparams, use_tpu)
  if use_tpu:
    opt = tf.contrib.tpu.CrossShardOptimizer(opt)

  tf.summary.scalar("learning_rate", learning_rate)
  opt_summaries = ["loss"]
  if hparams.summarize_grads:
    tf.logging.info("Summarizing gradients")
    opt_summaries.extend(["gradients", "gradient_norm", "global_gradient_norm"])

  if hparams.clip_grad_norm:
    tf.logging.info("Clipping gradients, norm: %0.5f", hparams.clip_grad_norm)
  if hparams.grad_noise_scale:
    tf.logging.info("Adding noise to gradients, noise scale: %0.5f",
                    hparams.grad_noise_scale)

  train_op = tf.contrib.layers.optimize_loss(
      name="training",
      loss=loss,
      global_step=tf.train.get_or_create_global_step(),
      learning_rate=learning_rate,
      clip_gradients=hparams.clip_grad_norm or None,
      gradient_noise_scale=hparams.grad_noise_scale or None,
      optimizer=opt,
      summaries=opt_summaries,
      colocate_gradients_with_ops=True)
  return train_op


class ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams, use_tpu=False):  # pylint: disable=super-init-not-called
    if optimizer_name == "Adam" and use_tpu:
      # LazyAdamOptimizer does not work on TPU
      optimizer_name = "TrueAdam"

    tf.logging.info("Using optimizer %s", optimizer_name)

    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "MultistepAdam":
      self._opt = multistep_optimizer.MultistepAdamOptimizer(
          lr,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon,
          n=hparams.optimizer_multistep_accumulate_steps)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr,
          momentum=hparams.optimizer_momentum_momentum,
          use_nesterov=hparams.optimizer_momentum_nesterov)
    elif optimizer_name == "YellowFin":
      self._opt = yellowfin.YellowFinOptimizer(
          learning_rate=lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "TrueAdam":
      self._opt = tf.train.AdamOptimizer(
          lr,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Adafactor":
      self._opt = adafactor.adafactor_optimizer_from_hparams(hparams, lr)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list=None, **kwargs):  # pylint: disable=arguments-differ
    gradients = self._opt.compute_gradients(loss, var_list, **kwargs)
    def cast_grad(g, v):
      if v is not None and g is not None:
        g = common_layers.cast_like(g, v)
      return (g, v)
    gradients = [cast_grad(g, v) for g, v in gradients]
    return gradients

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._opt.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)


def weight_decay_and_noise(loss, hparams, learning_rate, var_list=None):
  """Apply weight decay and weight noise."""
  if var_list is None:
    var_list = tf.trainable_variables()

  decay_vars = [v for v in var_list]
  noise_vars = [v for v in var_list if "/body/" in v.name]

  weight_decay_loss = weight_decay(hparams.weight_decay, decay_vars)
  if hparams.weight_decay:
    tf.summary.scalar("losses/weight_decay", weight_decay_loss)
  weight_noise_ops = weight_noise(hparams.weight_noise, learning_rate,
                                  noise_vars)

  with tf.control_dependencies(weight_noise_ops):
    loss = tf.identity(loss)

  loss += weight_decay_loss
  return loss


def weight_noise(noise_rate, learning_rate, var_list):
  """Apply weight noise to vars in var_list."""
  if not noise_rate:
    return [tf.no_op()]

  tf.logging.info("Applying weight noise scaled by learning rate, "
                  "noise_rate: %0.5f", noise_rate)

  noise_ops = []

  for v in var_list:
    with tf.device(v._ref().device):  # pylint: disable=protected-access
      scale = noise_rate * learning_rate * 0.001
      tf.summary.scalar("weight_noise_scale", scale)
      noise = tf.truncated_normal(v.shape) * scale
      noise_op = v.assign_add(noise)
      noise_ops.append(noise_op)

  return noise_ops


def weight_decay(decay_rate, var_list, skip_biases=True):
  """Apply weight decay to vars in var_list."""
  if not decay_rate:
    return 0.

  tf.logging.info("Applying weight decay, decay_rate: %0.5f", decay_rate)

  weight_decays = []
  for v in var_list:
    # Weight decay.
    # This is a heuristic way to detect biases that works for main tf.layers.
    is_bias = len(v.shape.as_list()) == 1 and v.name.endswith("bias:0")
    if not (skip_biases and is_bias):
      with tf.device(v.device):
        v_loss = tf.nn.l2_loss(v)
      weight_decays.append(v_loss)

  return tf.add_n(weight_decays) * decay_rate


def log_variable_sizes(var_list=None, tag=None, verbose=False):
  """Log the sizes and shapes of variables, and the total size.

  Args:
    var_list: a list of variables; defaults to trainable_variables
    tag: a string; defaults to "Trainable Variables"
    verbose: bool, if True, log every weight; otherwise, log total size only.
  """
  if var_list is None:
    var_list = tf.trainable_variables()
  if tag is None:
    tag = "Trainable Variables"

  if not var_list:
    return

  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    if verbose:
      tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                      v.name[:-2].ljust(80),
                      str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)


def get_variable_initializer(hparams):
  """Get variable initializer from hparams."""
  if not hparams.initializer:
    return None

  if not tf.contrib.eager.in_eager_mode():
    tf.logging.info("Using variable initializer: %s", hparams.initializer)
  if hparams.initializer == "orthogonal":
    return tf.orthogonal_initializer(gain=hparams.initializer_gain)
  elif hparams.initializer == "uniform":
    max_val = 0.1 * hparams.initializer_gain
    return tf.random_uniform_initializer(-max_val, max_val)
  elif hparams.initializer == "normal_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="normal")
  elif hparams.initializer == "uniform_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="uniform")
  else:
    raise ValueError("Unrecognized initializer: %s" % hparams.initializer)
