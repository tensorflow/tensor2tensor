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

"""trax learning rate schedules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import jax.numpy as np


@gin.configurable(blacklist=["history"])
def DefaultSchedule(history=None,
                    schedule="constant * linear_warmup * rsqrt_decay",
                    constant=0.001,
                    warmup_steps=100):
  """Default learning rate  schedule.

  Note: the learning rate schedule takes arguments and return a function,
  learning_rate: step -> lr, that only takes a step and return the rate.
  The reason is that learning_rate(step) is called at every training step,
  so should be efficient, while the schedule is re-computed only when
  evaluating the model, so usually only every 100 or 1000 steps.

  Interprets factors in the schedule string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)

  Args:
    history: the history of training and evaluation (History object).
    schedule: a string with factors separated by "*" that defines the schedule.
    constant: float, the starting constant for the learning rate schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history
  factors = [n.strip() for n in schedule.split("*")]

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
      else:
        raise ValueError("Unknown factor %s." % name)
    return ret

  return learning_rate


@gin.configurable(blacklist=["history"])
def EvalAdjustingSchedule(history,
                          constant=0.001,
                          steps_to_decrease=10,
                          improvement_margin=0.01,
                          decrease_rate=2.0,
                          metric="metrics/accuracy"):
  """Learning rate that decreases when eval metric stalls.

  If the chosen metric does not improve by improvement_margin for as many as
  steps_to_decrease steps, then the constant gets decreased by decrease rate.
  Finally, the default schedule gets called with the adjusted constant.

  Args:
    history: the history of training and evaluation (History object).
    constant: float, the starting constant for the learning rate schedule.
    steps_to_decrease: int, after how many steps without improvement
      should we decrease the constant.
    improvement_margin: how much we need to improve to count it.
    decrease_rate: by how much to decrease.
    metric: which evaluation metric to use for adjustments.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  metric = history.get(metric, "eval")
  adjusted = constant
  steps_without_improvement = 0
  while len(metric) > 1:
    last = metric.pop()
    if last[1] < metric[-1][1] * (1 + improvement_margin):
      steps_without_improvement += 1
    else:
      steps_without_improvement = 0
    if steps_without_improvement >= steps_to_decrease:
      adjusted /= decrease_rate
      steps_without_improvement = 0
  return DefaultSchedule(history, constant=adjusted)
