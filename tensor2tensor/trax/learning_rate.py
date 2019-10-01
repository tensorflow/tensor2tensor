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
  lr: history -> (step -> {"learning_rate": lr})

That is, they are functions that take a trax.history.History and return a
function that takes a step and returns a dict with entry "learning_rate".
"""

# TODO(pkozakowski): Revisit the decision to control nontrainable parameters
# using LR schedules, or at least rename the module.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

from absl import logging
import gin

from tensor2tensor.trax import models as trax_models
from tensor2tensor.trax import utils
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.backend import random as jax_random
from tensor2tensor.trax.rl import online_tune
from tensor2tensor.trax.rl import ppo


@gin.configurable(blacklist=["history"])
def MultifactorSchedule(history=None,
                        factors="constant * linear_warmup * rsqrt_decay",
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
    a function learning_rate(step): float -> {"learning_rate": float}, the
    step-dependent lr.
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
        ret *= (decay_factor ** (step//steps_per_decay))
      else:
        raise ValueError("Unknown factor %s." % name)
    ret = np.asarray(ret, dtype=np.float32)
    return {"learning_rate": ret}

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
    a function learning_rate(step): float -> {"learning_rate": float}, the
    step-dependent lr.
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
def PolicySchedule(
    history,
    observation_metrics=(
        ("train", "metrics/accuracy"),
        ("train", "metrics/loss"),
        ("eval", "metrics/accuracy"),
        ("eval", "metrics/loss"),
    ),
    include_controls_in_observation=False,
    control_configs=(
        # (name, start, (low, high), flip)
        ("learning_rate", 1e-3, (1e-9, 10.0), False),
    ),
    observation_range=(0.0, 10.0),
    action_multipliers=(1.0 / 1.5, 1.0 / 1.25, 1.0, 1.25, 1.5),
    policy_and_value_model=trax_models.FrameStackMLP,
    policy_and_value_two_towers=False,
    policy_and_value_vocab_size=None,
    policy_dir=gin.REQUIRED,
    temperature=1.0,
):
  """Learning rate schedule controlled by a learned policy.

  Args:
    history: the history of training and evaluation (History object).
    observation_metrics: list of pairs (mode, metric), as in the History object.
    include_controls_in_observation: bool, whether to include the controls in
      observations.
    control_configs: control configs, see trax.rl.envs.OnlineTuneEnv.
    observation_range: tuple (low, high), range to clip the metrics to.
    action_multipliers: sequence of LR multipliers that policy actions
      correspond to.
    policy_and_value_model: Trax model to use as the policy.
    policy_and_value_two_towers: bool, whether the action distribution and value
      prediction is computed by separate model towers.
    policy_and_value_vocab_size: vocabulary size of a policy and value network
      operating on serialized representation. If None, use raw continuous
      representation.
    policy_dir: directory with the policy checkpoint.
    temperature: temperature for sampling from the policy.

  Returns:
    a function nontrainable_params(step): float -> {"name": float}, the
    step-dependent schedule for nontrainable parameters.
  """

  # Turn the history into observations for the policy. If we don't have any,
  # return the initial learning rate.
  start_time = time.time()
  observations = online_tune.history_to_observations(
      history, observation_metrics, observation_range,
      control_configs if include_controls_in_observation else None
  )
  logging.vlog(
      1, "Building observations took %0.2f sec.", time.time() - start_time)
  if observations.shape[0] == 0:
    controls = {
        name: start_value
        for (name, start_value, _, _) in control_configs
    }
    return lambda _: controls

  assert policy_and_value_vocab_size is None, (
      "Serialized policies are not supported yet."
  )
  # Build the policy network and load its parameters.
  start_time = time.time()
  net = ppo.policy_and_value_net(
      n_controls=len(control_configs),
      n_actions=len(action_multipliers),
      vocab_size=policy_and_value_vocab_size,
      bottom_layers_fn=policy_and_value_model,
      two_towers=policy_and_value_two_towers,
  )
  logging.vlog(
      1, "Building the policy network took %0.2f sec.", time.time() - start_time
  )
  start_time = time.time()
  # (opt_state, state, epoch, opt_step)
  (opt_state, state, _, _) = ppo.maybe_restore_opt_state(policy_dir)
  assert opt_state is not None, "Policy checkpoint not found."
  (params, _) = opt_state
  logging.vlog(
      1, "Restoring the policy parameters took %0.2f sec.",
      time.time() - start_time
  )

  # Run the policy and sample an action.
  seed = random.randint(0, 2**31 - 1)
  rng = jax_random.get_prng(seed=seed)
  start_time = time.time()
  # ((log_probs, value_preds), state). We have no way to pass state to the next
  # step, but that should be fine.
  (log_probs, _) = (
      net(np.array([observations]), params=params, state=state, rng=rng))
  logging.vlog(
      1, "Running the policy took %0.2f sec.", time.time() - start_time
  )
  # Sample from the action distribution for the last timestep.
  assert log_probs.shape == (
      1, len(control_configs) * observations.shape[0], len(action_multipliers)
  )
  action = utils.gumbel_sample(
      log_probs[0, -len(control_configs):, :] / temperature
  )

  # Get new controls.
  controls = {
      # name: value
      control_config[0]: online_tune.update_control(  # pylint: disable=g-complex-comprehension
          control_config, control_action, history, action_multipliers
      )
      for (control_action, control_config) in zip(action, control_configs)
  }
  return lambda _: controls
