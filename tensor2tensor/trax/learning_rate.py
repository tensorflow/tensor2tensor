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


# A dictionary to memoize results of the MultifactorSchedule below.
# We memoize because returning exactly the same function objects allows
# later (in the training loop) to optimize re-compiling the function (for
# running on an accelerator) only when it changes. Note that this does not
# affect correctness, it is done purely for performance reasons.
_memoized_multifactor_schedules = {}


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

  cache_args = (factors, constant, warmup_steps)
  if cache_args in _memoized_multifactor_schedules:
    return _memoized_multifactor_schedules[cache_args]

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
        ret *= (decay_factor ** (step//steps_per_decay))
      else:
        raise ValueError("Unknown factor %s." % name)
    return ret

  _memoized_multifactor_schedules[cache_args] = learning_rate
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
