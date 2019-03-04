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

import jax.numpy as np


@gin.configurable(blacklist=["history"])
def MultifactorSchedule(history=None,
                        factors="constant * linear_warmup * rsqrt_decay",
                        constant=0.001,
                        warmup_steps=100):
  """Factor-based learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)

  Args:
    history: the history of training and evaluation (History object).
    factors: a string with factors separated by "*" that defines the schedule.
    constant: float, the starting constant for the learning rate schedule.
    warmup_steps: how many steps to warm up for in the warmup schedule.

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
      else:
        raise ValueError("Unknown factor %s." % name)
    return ret

  return learning_rate


# TODO(trax): Find a way to enable this with @jit.
# Currently disabled because it does not work with @jit. To use properly, would
# need to re-initialize this learning rate schedule function, the optimizer, and
# the update jit.
# @gin.configurable(blacklist=["history"])
# def EvalAdjustingSchedule(history,
#                           constant=0.001,
#                           steps_to_decrease=10,
#                           improvement_margin=0.01,
#                           decrease_rate=2.0,
#                           adjustment_frequency=100,
#                           history_mode="eval",
#                           metric="metrics/accuracy"):
#   """Learning rate that decreases when eval metric stalls.
#
#   If the chosen metric does not improve by improvement_margin for as many as
#   steps_to_decrease steps, then the constant gets decreased by decrease rate.
#   Finally, the MultifactorSchedule gets called with the adjusted constant.
#
#   Args:
#     history: trax.history.History, the history of training and evaluation.
#     constant: float, the starting constant for the learning rate schedule.
#     steps_to_decrease: int, after how many steps without improvement
#       should we decrease the constant.
#     improvement_margin: how much we need to improve to consider the metric
#       improved.
#     decrease_rate: by what fraction to decrease (i.e. lr /= decrease_rate).
#     adjustment_frequency: int, how often to reset the learning rate based on
#       the latest history.
#     history_mode: str, which mode of the history to use.
#     metric: which evaluation metric to use for adjustments.
#
#   Returns:
#     a function learning_rate(step): float -> float, the step-dependent lr.
#   """
#
#   def get_constant_from_history():
#     metrics = history.get(history_mode, metric)
#     adjusted = constant
#     steps_without_improvement = 0
#     while len(metrics) > 1:
#       last = metrics.pop()
#       if last[1] < metrics[-1][1] * (1 + improvement_margin):
#         steps_without_improvement += 1
#       else:
#         steps_without_improvement = 0
#       if steps_without_improvement >= steps_to_decrease:
#         adjusted /= decrease_rate
#         steps_without_improvement = 0
#     return adjusted
#
#   state = {
#       "schedule": None,
#   }
#
#   def reset_schedule():
#     state["schedule"] = MultifactorSchedule(
#         history, constant=get_constant_from_history())
#
#   reset_schedule()
#
#   def lr_step(step):
#     if step % adjustment_frequency == 0:
#       reset_schedule()
#     return state["schedule"](step)
#
#   return lr_step
