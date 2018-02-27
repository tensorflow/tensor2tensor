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

# Dependency imports

import numpy as np

import tensorflow as tf


def learning_rate_factor(name, step_num, hparams):
  if name == "constant":
    return hparams.learning_rate_constant
  elif name == "linear_warmup":
    return tf.minimum(1.0, step_num / hparams.learning_rate_warmup_steps)
  elif name == "rsqrt_decay":
    return tf.rsqrt(tf.maximum(step_num, hparams.learning_rate_warmup_steps))
  elif name == "rsqrt_hidden_size":
    return hparams.hidden_size ** -0.5
  elif name == "legacy":
    return legacy_learning_rate_schedule(hparams)
  else:
    raise ValueError("unknown learning rate factor %s" % name)


def learning_rate_schedule(hparams):
  """Learning rate schedule based on hparams."""
  step_num = tf.to_float(tf.train.get_or_create_global_step())
  schedule_string = hparams.learning_rate_schedule
  names = schedule_string.split("*")
  names = [name.strip() for name in names if name.strip()]
  ret = 1.0
  for name in names:
    ret *= learning_rate_factor(name, step_num, hparams)
  return ret


def legacy_learning_rate_schedule(hparams):
  """Backwards-compatible learning-rate schedule."""
  step_num = tf.to_float(tf.train.get_or_create_global_step())
  warmup_steps = tf.to_float(hparams.learning_rate_warmup_steps)
  if hparams.learning_rate_decay_scheme == "noam":
    ret = 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
        (step_num + 1) * warmup_steps**-1.5, (step_num + 1)**-0.5)
  else:
    warmup_steps = hparams.learning_rate_warmup_steps
    warmup = _learning_rate_warmup(warmup_steps)
    decay = _learning_rate_decay(hparams, warmup_steps)
    ret = tf.where(step_num < warmup_steps, warmup, decay)
  optimizer_correction = 0.002 if "Adam" in hparams.optimizer else 1.0
  return ret * optimizer_correction * hparams.learning_rate


def _legacy_sqrt_decay(step):
  """Decay like 1 / sqrt(step), multiplied by 500 to normalize."""
  return 500.0 / tf.sqrt(tf.maximum(step, 1.0))


def _piecewise_learning_rate(step, boundaries, values):
  """Scale learning rate according to the given schedule.

  Multipliers are not cumulative.

  Args:
    step: global step
    boundaries: List of steps to transition on.
    values: Multiplier to apply at each boundary transition.

  Returns:
    Scaled value for the learning rate.
  """
  values = [1.0] + values
  return tf.train.piecewise_constant(
      step, boundaries, values, name="piecewise_lr")


def _learning_rate_decay(hparams, warmup_steps=0):
  """Learning rate decay multiplier."""
  scheme = hparams.learning_rate_decay_scheme
  warmup_steps = tf.to_float(warmup_steps)
  global_step = tf.to_float(tf.train.get_or_create_global_step())

  if not scheme or scheme == "none":
    return tf.constant(1.)

  tf.logging.info("Applying learning rate decay: %s.", scheme)

  if scheme == "exp":
    decay_steps = hparams.learning_rate_decay_steps
    p = (global_step - warmup_steps) / decay_steps
    if hparams.learning_rate_decay_staircase:
      p = tf.floor(p)
    return tf.pow(hparams.learning_rate_decay_rate, p)

  if scheme == "piecewise":
    return _piecewise_learning_rate(global_step,
                                    hparams.learning_rate_boundaries,
                                    hparams.learning_rate_multiples)

  if scheme == "cosine":
    cycle_steps = hparams.learning_rate_cosine_cycle_steps
    cycle_position = global_step % (2 * cycle_steps)
    cycle_position = cycle_steps - tf.abs(cycle_steps - cycle_position)
    return 0.5 * (1 + tf.cos(np.pi * cycle_position / cycle_steps))

  if scheme == "cyclelinear10x":
    # Cycle the rate linearly by 10x every warmup_steps, up and down.
    cycle_steps = warmup_steps
    cycle_position = global_step % (2 * cycle_steps)
    cycle_position = tf.to_float(  # Normalize to the interval [-1, 1].
        cycle_position - cycle_steps) / float(cycle_steps)
    cycle_position = 1.0 - tf.abs(cycle_position)  # 0 to 1 and back to 0.
    return (cycle_position + 0.1) * 3.0  # 10x difference each cycle (0.3-3).

  if scheme == "sqrt":
    return _legacy_sqrt_decay(global_step - warmup_steps)

  raise ValueError("Unrecognized learning rate decay scheme: %s" %
                   hparams.learning_rate_decay_scheme)


def _learning_rate_warmup(warmup_steps, warmup_schedule="exp"):
  """Learning rate warmup multiplier."""
  if not warmup_steps:
    return tf.constant(1.)

  tf.logging.info("Applying %s learning rate warmup for %d steps",
                  warmup_schedule, warmup_steps)

  warmup_steps = tf.to_float(warmup_steps)
  global_step = tf.to_float(tf.train.get_or_create_global_step())

  if warmup_schedule == "exp":
    return tf.exp(tf.log(0.01) / warmup_steps)**(warmup_steps - global_step)
  else:
    assert warmup_schedule == "linear"
    start = tf.constant(0.35)
    return ((tf.constant(1.) - start) / warmup_steps) * global_step + start
