# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

# Dependency imports

import numpy as np

from tensor2tensor.utils import yellowfin

import tensorflow as tf



def optimize(loss, learning_rate, hparams):
  """Minimize loss."""
  loss = tf.identity(loss, name="total_loss")
  opt = ConditionalOptimizer(hparams.optimizer, learning_rate, hparams)
  opt_summaries = ["learning_rate", "loss"]
  if hparams.summarize_grads:
    opt_summaries.extend(["gradients", "gradient_norm"])
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

  def __init__(self, optimizer_name, lr, hparams):
    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "YellowFin":
      tf.logging.info("Init YellowFin Optimizer.")
      self._opt = yellowfin.YellowFinOptimizer(
          learning_rate=lr, momentum=hparams.optimizer_momentum_momentum)
    elif optimizer_name == "TrueAdam":
      self._opt = tf.train.AdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list=None, **kwargs):
    return self._opt.compute_gradients(loss, var_list, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    return self._opt.apply_gradients(
        grads_and_vars, global_step=global_step, name=name)


def _sqrt_decay(step):
  """Decay like 1 / sqrt(step), multiplied by 500 to normalize."""
  return 500.0 / tf.sqrt(tf.maximum(step, 1.0))


def _exp_decay_after(step, rate, from_which_step):
  """Decay exponentially by rate (per step) starting at from_which_step."""
  return tf.cond(
      step < from_which_step,
      lambda: tf.constant(1.0),
      lambda: rate**(step - from_which_step),
      name="exponential_decay_step_cond")


def learning_rate_decay(hparams, num_worker_replicas=1, num_train_steps=1):
  """Inverse-decay learning rate until warmup_steps, then decay."""
  warmup_steps = tf.to_float(
      hparams.learning_rate_warmup_steps * num_worker_replicas)
  step = tf.to_float(tf.train.get_or_create_global_step())
  if hparams.learning_rate_decay_scheme == "noam":
    return 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
        (step + 1) * warmup_steps**-1.5, (step + 1)**-0.5)
  elif hparams.learning_rate_decay_scheme == "exp100k":
    return 0.94**(step // 100000)
  elif hparams.learning_rate_decay_scheme == "cosine":
    cycle_steps = hparams.learning_rate_cosine_cycle_steps
    return 0.5 * (1 + tf.cos(np.pi * (step % cycle_steps) / cycle_steps))
  elif hparams.learning_rate_decay_scheme == "cyclelinear10x":
    # Cycle the rate linearly by 10x every warmup_steps, up and down.
    cycle_steps = hparams.learning_rate_warmup_steps
    cycle_position = step % (2 * cycle_steps)
    cycle_position = tf.to_float(  # Normalize to the interval [-1, 1].
        cycle_position - cycle_steps) / float(cycle_steps)
    cycle_position = 1.0 - tf.abs(cycle_position)  # 0 to 1 and back to 0.
    return (cycle_position + 0.1) * 3.0  # 10x difference each cycle (0.3-3).

  inv_base = tf.exp(tf.log(0.01) / warmup_steps)
  inv_decay = inv_base**(warmup_steps - step)
  if hparams.learning_rate_decay_scheme == "sqrt":
    decay = _sqrt_decay(step - warmup_steps)
  elif hparams.learning_rate_decay_scheme == "exp10k":
    decay = _exp_decay_after(step - warmup_steps, 0.9995,
                             num_train_steps - warmup_steps - 10000)
  elif hparams.learning_rate_decay_scheme == "exp50k":
    decay = _exp_decay_after(step - warmup_steps, 0.99995,
                             num_train_steps - warmup_steps - 50000)
  elif hparams.learning_rate_decay_scheme == "exp500k":
    decay = _exp_decay_after(step - warmup_steps, 0.9999955,
                             num_train_steps - warmup_steps - 500000)
  elif hparams.learning_rate_decay_scheme == "none":
    decay = tf.constant(1.0)
  else:
    raise ValueError("Unrecognized learning rate decay scheme: %s" %
                     hparams.learning_rate_decay_scheme)
  return tf.where(step < warmup_steps, inv_decay, decay)
