# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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
from tensor2tensor.utils import adafactor as adafactor_lib
from tensor2tensor.utils import contrib
from tensor2tensor.utils import misc_utils
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import multistep_optimizer
from tensor2tensor.utils import registry
from tensor2tensor.utils import yellowfin

import tensorflow.compat.v1 as tf


from tensorflow.python.framework import dtypes  # pylint: disable=g-direct-tensorflow-import


def _mixed_precision_is_enabled(hparams):
  """Should be the same as in common_attention, avoiding import."""
  activation_dtype = hparams.activation_dtype
  weight_dtype = hparams.weight_dtype
  return activation_dtype == tf.float16 and weight_dtype == tf.float32


def optimize(loss,
             learning_rate,
             hparams,
             use_tpu=False,
             variables=None):
  """Minimize loss."""
  loss = weight_decay_and_noise(loss, hparams, learning_rate)
  loss = tf.identity(loss, name="total_loss")
  if variables is None:
    variables = tf.trainable_variables()
  # Print trainable variables.
  log_variable_sizes(variables, verbose=hparams.summarize_vars)
  # Print non-trainable variables.
  non_trainable_variables = list(
      set(tf.global_variables()) - set(variables))
  log_variable_sizes(non_trainable_variables, tag="Non-trainable variables",
                     verbose=hparams.summarize_vars)
  if hparams.summarize_vars:
    summarize_variables(variables)
    # Summarize non-trainable variables as well
    summarize_variables(non_trainable_variables, tag="Non-trainable variables")
  diet_vars = [
      v for v in tf.global_variables() if v.dtype == dtypes.float16_ref
  ]
  log_variable_sizes(
      diet_vars, "Diet Variables", verbose=hparams.summarize_vars)
  opt = ConditionalOptimizer(hparams.optimizer, learning_rate, hparams, use_tpu)
  if use_tpu:
    opt = contrib.tpu().CrossShardOptimizer(opt)
  if getattr(hparams, "gpu_automatic_mixed_precision", False):
    if use_tpu:
      raise RuntimeError("GPU auto mixed precision cannot be used with TPU")
    elif _mixed_precision_is_enabled(hparams):
      raise RuntimeError(
          "GPU auto mixed precision cannot be used with manual mixed precision")
    else:
      setattr(opt, "_use_locking", "True")
      setattr(opt, "_name", "ConditionalOptimizer")
      opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

  opt_summaries = []
  if common_layers.should_generate_summaries():
    tf.summary.scalar("learning_rate", learning_rate)
    opt_summaries.append("loss")
    if hparams.summarize_grads:
      tf.logging.info("Summarizing gradients")
      opt_summaries.extend(
          ["gradients", "gradient_norm", "global_gradient_norm"])

  if hparams.clip_grad_norm:
    tf.logging.info("Clipping gradients, norm: %0.5f", hparams.clip_grad_norm)
  if hparams.grad_noise_scale:
    tf.logging.info("Adding noise to gradients, noise scale: %0.5f",
                    hparams.grad_noise_scale)

  train_op = contrib.layers().optimize_loss(
      name="training",
      loss=loss,
      global_step=tf.train.get_or_create_global_step(),
      learning_rate=learning_rate,
      clip_gradients=hparams.clip_grad_norm or None,
      gradient_noise_scale=hparams.grad_noise_scale or None,
      optimizer=opt,
      summaries=opt_summaries,
      colocate_gradients_with_ops=True,
      variables=variables)
  return train_op


@registry.register_optimizer
def adam(learning_rate, hparams):
  """Return adam optimizer for the given params."""
  # We change the default epsilon for Adam.
  # Using LazyAdam as it's much faster for large vocabulary embeddings.
  if contrib.is_tf2:
    # in TF2 beta1 -> beta_1 :/
    return contrib.opt().LazyAdamOptimizer(
        learning_rate,
        beta_1=hparams.optimizer_adam_beta1,
        beta_2=hparams.optimizer_adam_beta2,
        epsilon=hparams.optimizer_adam_epsilon)
  else:
    return contrib.opt().LazyAdamOptimizer(
        learning_rate,
        beta1=hparams.optimizer_adam_beta1,
        beta2=hparams.optimizer_adam_beta2,
        epsilon=hparams.optimizer_adam_epsilon)


@registry.register_optimizer
def multistep_adam(learning_rate, hparams):
  return multistep_optimizer.MultistepAdamOptimizer(
      learning_rate,
      beta1=hparams.optimizer_adam_beta1,
      beta2=hparams.optimizer_adam_beta2,
      epsilon=hparams.optimizer_adam_epsilon,
      n=hparams.optimizer_multistep_accumulate_steps)


@registry.register_optimizer
def momentum(learning_rate, hparams):
  return tf.train.MomentumOptimizer(
      learning_rate,
      momentum=hparams.optimizer_momentum_momentum,
      use_nesterov=hparams.optimizer_momentum_nesterov)


@registry.register_optimizer
def yellow_fin(learning_rate, hparams):
  return yellowfin.YellowFinOptimizer(
      learning_rate=learning_rate,
      momentum=hparams.optimizer_momentum_momentum)


@registry.register_optimizer
def true_adam(learning_rate, hparams):
  return tf.train.AdamOptimizer(
      learning_rate,
      beta1=hparams.optimizer_adam_beta1,
      beta2=hparams.optimizer_adam_beta2,
      epsilon=hparams.optimizer_adam_epsilon)


@registry.register_optimizer
def adam_w(learning_rate, hparams):
  return contrib.opt().AdamWOptimizer(
      weight_decay=hparams.weight_decay,
      learning_rate=learning_rate,
      beta1=hparams.optimizer_adam_beta1,
      beta2=hparams.optimizer_adam_beta2,
      epsilon=hparams.optimizer_adam_epsilon)


@registry.register_optimizer
def adafactor(learning_rate, hparams):
  return adafactor_lib.adafactor_optimizer_from_hparams(hparams, learning_rate)




def _register_base_optimizer(name, opt):
  key = misc_utils.camelcase_to_snakecase(name)
  if key in registry.Registries.optimizers:
    return
  registry.register_optimizer(key)(
      lambda learning_rate, hparams: opt(learning_rate))


for _name, _opt in contrib.layers().OPTIMIZER_CLS_NAMES.items():
  _register_base_optimizer(_name, _opt)


class ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams, use_tpu=False):  # pylint: disable=super-init-not-called
    tf.logging.info("Using optimizer %s", optimizer_name)

    mlperf_log.transformer_print(key=mlperf_log.OPT_NAME,
                                 value=optimizer_name,
                                 hparams=hparams)
    mlperf_log.transformer_print(
        key=mlperf_log.OPT_HP_ADAM_BETA1, value=hparams.optimizer_adam_beta1,
        hparams=hparams)
    mlperf_log.transformer_print(
        key=mlperf_log.OPT_HP_ADAM_BETA2, value=hparams.optimizer_adam_beta2,
        hparams=hparams)
    mlperf_log.transformer_print(
        key=mlperf_log.OPT_HP_ADAM_EPSILON,
        value=hparams.optimizer_adam_epsilon,
        hparams=hparams)

    self._opt = registry.optimizer(optimizer_name)(lr, hparams)
    if _mixed_precision_is_enabled(hparams):
      if not hparams.mixed_precision_optimizer_loss_scaler:
        tf.logging.warning("Using mixed precision without a loss scaler will "
                           "likely cause numerical errors.")
      elif hparams.mixed_precision_optimizer_loss_scaler != "exponential":
        raise ValueError("Mixed precision training only supports the "
                         "exponential loss scaler")
      else:
        tf.logging.info(
            ("Using Exponential Update Loss Scaler with",
             "init loss scale of {}".format(
                 hparams.mixed_precision_optimizer_init_loss_scale)))
        manager = contrib.mixed_precision().ExponentialUpdateLossScaleManager(
            init_loss_scale=hparams.mixed_precision_optimizer_init_loss_scale,
            incr_every_n_steps=2000,
            decr_every_n_nan_or_inf=2,
            incr_ratio=2,
            decr_ratio=0.5)
        self._opt = contrib.mixed_precision().LossScaleOptimizer(
            self._opt, manager)

    self._zero_grads = hparams.optimizer_zero_grads

  def compute_gradients(self, loss, var_list=None, **kwargs):  # pylint: disable=arguments-differ
    if contrib.is_tf2:
      gradients = self._opt.get_gradients(loss, var_list)
      gradients = zip(gradients, var_list)
    else:
      gradients = self._opt.compute_gradients(loss, var_list, **kwargs)

    def cast_grad(g, v):
      if v is not None and g is not None:
        g = common_layers.cast_like(g, v)
      if self._zero_grads and g is None:
        g = tf.zeros_like(v)
      return (g, v)
    gradients = [cast_grad(g, v) for g, v in gradients]
    return gradients

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    if contrib.is_tf2:
      with tf.control_dependencies(
          [tf.assign_add(tf.train.get_or_create_global_step(), 1)]):
        return self._opt.apply_gradients(grads_and_vars, name=name)
    else:
      return self._opt.apply_gradients(
          grads_and_vars, global_step=global_step, name=name)


def weight_decay_and_noise(loss, hparams, learning_rate, var_list=None):
  """Apply weight decay and weight noise."""
  if var_list is None:
    var_list = tf.trainable_variables()

  decay_vars = [v for v in var_list]
  noise_vars = [v for v in var_list if "/body/" in v.name]

  weight_decay_loss = weight_decay(hparams.weight_decay, decay_vars)
  if hparams.weight_decay and common_layers.should_generate_summaries():
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
    with tf.device(v.device):  # pylint: disable=protected-access
      scale = noise_rate * learning_rate * 0.001
      if common_layers.should_generate_summaries():
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


def summarize_variables(var_list=None, tag=None):
  """Summarize the variables.

  Args:
    var_list: a list of variables; defaults to trainable_variables.
    tag: name scope of the summary; defaults to training_variables/.
  """
  if var_list is None:
    var_list = tf.trainable_variables()
  if tag is None:
    tag = "training_variables/"

  name_to_var = {v.name: v for v in var_list}
  for v_name in list(name_to_var):
    v = name_to_var[v_name]
    tf.summary.histogram(tag + v_name, v)


def get_variable_initializer(hparams):
  """Get variable initializer from hparams."""
  if not hparams.initializer:
    return None

  mlperf_log.transformer_print(key=mlperf_log.MODEL_HP_INITIALIZER_GAIN,
                               value=hparams.initializer_gain,
                               hparams=hparams)

  if not tf.executing_eagerly():
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
  elif hparams.initializer == "xavier":
    return tf.initializers.glorot_uniform()
  else:
    raise ValueError("Unrecognized initializer: %s" % hparams.initializer)
