# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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
"""trax learning rate schedules.

The learning rate schedules here all have the signature:
  lr: history -> (step -> lr)

That is, they are functions that take a trax.history.History and return a
function that takes a step and returns a learning rate.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
from tensor2tensor.trax.backend import numpy as np


@gin.configurable(blacklist=["history"])
def MultifactorSchedule(history=None,
                        factors="constant * linear_warmup",
                        constant=0.1,
                        warmup_steps=400,
                        decay_factor=0.5,
                        steps_per_decay=20000):
  """Factor-based learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.

  Args:
    history: the history of training and evaluation (History object).
    factors: a string with factors separated by "*" that defines the schedule.
    constant: float, the starting constant for the learning rate schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  factors = [n.strip() for n in factors.split("*")]

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= constant
      elif name == "linear_warmup":
        ret *= np.minimum(1.0, step / warmup_steps)
      elif name == "rsqrt_decay":
        ret /= np.sqrt(np.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= (decay_factor**(step // steps_per_decay))
      else:
        raise ValueError("Unknown factor %s." % name)
    return ret

  return learning_rate


@gin.configurable(blacklist=["history"])
def EvalAdjustingSchedule(history,
                          constant=0.1,
                          steps_to_decrease=20,
                          improvement_margin=0.001,
                          decrease_rate=1.5,
                          history_mode="eval",
                          metric="metrics/accuracy"):
  """Learning rate that decreases when eval metric stalls.

  If the chosen metric does not improve by improvement_margin for as many as
  steps_to_decrease steps, then the constant gets decreased by decrease rate.
  Finally, the MultifactorSchedule gets called with the adjusted constant.

  Args:
    history: trax.history.History, the history of training and evaluation.
    constant: float, the starting constant for the learning rate schedule.
    steps_to_decrease: int, after how many steps without improvement
      should we decrease the constant.
    improvement_margin: how much we need to improve to consider the metric
      improved.
    decrease_rate: by what fraction to decrease (i.e. lr /= decrease_rate).
    history_mode: str, which mode of the history to use.
    metric: which evaluation metric to use for adjustments.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  metrics = history.get(history_mode, metric)
  adjusted = constant
  if len(metrics) < 2:
    return MultifactorSchedule(history, constant=adjusted)

  steps_without_improvement = 0
  cur = metrics.pop()[1]  # The most-recent value of the metric.
  while len(metrics) > 1:
    # The one-before value of metrics as .pop() removes one element each time.
    prev = metrics.pop()[1]
    if cur < prev * (1 + improvement_margin):
      steps_without_improvement += 1
    else:
      cur = prev
      steps_without_improvement = 0
    if steps_without_improvement >= steps_to_decrease:
      adjusted /= decrease_rate
      cur = prev
      steps_without_improvement = 0

  return MultifactorSchedule(history, constant=adjusted)


@gin.configurable(blacklist=["history"])
def ExponentialDecaySchedule(history=None,
                        initial_learning_rate,
                        decay_steps,
                        decay_rate,
                        staircase=False):
  """Applies exponential decay to the learning rate.

  Args:
   initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
   decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
   decay_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The decay rate.
   staircase: Boolean.  If `True` decay the learning rate at discrete
        intervals

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    p = step.astype(np.float32)
    p /= decay_steps

    if staircase:
      p = np.floor(p)
    return initial_learning_rate * np.power(decay_rate, p)

  return learning_rate



@gin.configurable(blacklist=["history"])
def PolynomialSchedule(history=None,
                       initial_learning_rate,
                       decay_steps,
                       end_learning_rate=0.0001,
                       power=1.0,
                       cycle=False):
  """Polynomial-based learning rate schedule.

  This schedule applies a polynomial decay function to an optimizer step,
    given a provided `initial_learning_rate`, to reach an `end_learning_rate`
    in the given `decay_steps`.

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The minimal end learning rate.
      power: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The power of the polynomial. Defaults to linear, 1.0.
      cycle: A boolean, whether or not it should cycle beyond decay_steps.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    step_fl = step.astype(np.float32)
    decay_steps_fl = decay_steps.astype(np.float32)

    if cycle:
      multiplier = 1.0 if step == 0 else np.ceil(step / decay_steps)
      decay_steps_fl *= multiplier
    else:
      step_fl = np.min(step_fl, decay_steps)

    p = step_fl / decay_steps_fl
    return (initial_learning_rate - end_learning_rate) * np.power(
        1. - p, power) + end_learning_rate

  return learning_rate


@gin.configurable(blacklist=["history"])
def PiecewiseConstantSchedule(history=None,
                              boundaries,
                              values):
  """Piecewise constant from boundaries and interval values schedule.

  This schedule applies a polynomial decay function to an optimizer step,
    given a provided `initial_learning_rate`, to reach an `end_learning_rate`
    in the given `decay_steps`.

  Args:
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
    increasing entries, and with all elements having the same type as the
    optimizer step.
    values: A list of `Tensor`s or `float`s or `int`s that specifies the
    values for the intervals defined by `boundaries`. It should have one
    more element than `boundaries`, and all elements should have the same
    type.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    step_fl = step.astype(np.float32)

    pos = np.searchsorted(boundaries, step)
    pos = np.minimun(pos, len(boundaries) - 1)
    return values[pos]

  return learning_rate


@gin.configurable(blacklist=["history"])
def InverseTimeDecaySchedule(history=None,
                             initial_learning_rate,
                             decay_steps,
                             decay_rate,
                             staircase=False):
  """Applies inverse time decay schedule.

  This schedule applies a polynomial decay function to an optimizer step,
    given a provided `initial_learning_rate`, to reach an `end_learning_rate`
    in the given `decay_steps`.

  Args:
    initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
    decay_steps: How often to apply decay.
    decay_rate: A Python number.  The decay rate.
    staircase: Whether to apply decay in a discrete staircase, as opposed to
        continuous, fashion.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    step_fl = step.astype(np.float32)
    p = step_fl / decay_steps
    if staircase:
      p = np.floor(p)

    denom = 1. + decay_rate * p

    return initial_learning_rate / denom

  return learning_rate


@gin.configurable(blacklist=["history"])
def CosineDecaySchedule(history=None,
                        initial_learning_rate,
                        decay_steps,
                        alpha=0.0):
  """Applies cosine decay schedule.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Args:
    initial_learning_rate: A scalar `float32` or `float64` Tensor or a
        Python number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of initial_learning_rate.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    step_fl = np.minimun(step_fl, decay_steps)
    step_fl = step.astype(np.float32)

    p = step_fl / decay_steps
    cosine_decayed = 0.5 * (1. + np.cos(p * np.pi))
    decayed = (1. - alpha) * cosine_decayed + alpha
    return decayed * initial_learning_rate

  return learning_rate


@gin.configurable(blacklist=["history"])
def CosineDecayRestartsSchedule(history=None,
                                initial_learning_rate,
                                first_decay_steps,
                                t_mul=2.0,
                                m_mul=1.0,
                                alpha=0.0):
  """Applies cosine decay with restarts schedule.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  The learning rate multiplier first decays
  from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
  restart is performed. Each new warm restart runs for `t_mul` times more
  steps and with `m_mul` times smaller initial learning rate.

  Args:
    initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
        number. Number of steps to decay over.
      t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period
      m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period:
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the initial_learning_rate.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    step_fl = np.minimun(step_fl, decay_steps)
    step_fl = step.astype(np.float32)

    p = step_fl / decay_steps
    cosine_decayed = 0.5 * (1. + np.cos(p * np.pi))
    decayed = (1. - alpha) * cosine_decayed + alpha
    return decayed * initial_learning_rate

  return learning_rate