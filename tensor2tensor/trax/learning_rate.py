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

import numpy as onp

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
def ExponentialDecaySchedule(history,
                             initial_learning_rate,
                             decay_steps,
                             decay_rate,
                             staircase=False):
  """Applies exponential decay to the learning rate.

  It is computed as:
    ```python
    def decayed_learning_rate(step):
      return initial_learning_rate * decay_rate ^ (step / decay_steps)
    ```
    If the argument `staircase` is `True`, then `step / decay_steps` is
    an integer division and the decayed learning rate follows a
    staircase function.

  Args:
   initial_learning_rate: A scalar `float32`. The initial learning rate.
   decay_steps: A scalar `int32` or `int64` Must be positive.
   See the decay computation above.
   decay_rate: A scalar `float32` or `float64`. The decay rate.
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
def PolynomialSchedule(history,
                       initial_learning_rate,
                       decay_steps,
                       end_learning_rate=0.0001,
                       power=1.0,
                       cycle=False):
  """Polynomial-based learning rate schedule.

  This schedule applies a polynomial decay function to an optimizer step,
    given a provided `initial_learning_rate`, to reach an `end_learning_rate`
    in the given `decay_steps`.

   It is computed as:
    ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      return ((initial_learning_rate - end_learning_rate) *
              (1 - step / decay_steps) ^ (power)
             ) + end_learning_rate
    ```
    If `cycle` is True then a multiple of `decay_steps` is used, the first one
    that is bigger than `step`.
    ```python
    def decayed_learning_rate(step):
      decay_steps = decay_steps * ceil(step / decay_steps)
      return ((initial_learning_rate - end_learning_rate) *
              (1 - step / decay_steps) ^ (power)
             ) + end_learning_rate
    ```


  Args:
    learning_rate: A scalar `float32` or `float64`.
     The initial learning rate.
    decay_steps: A scalar `int32` or `int64`. Must be positive.
    See the decay computation above.
    end_learning_rate: A scalar `float32` or `float64`.
    The minimal end learning rate.
    power: A scalar `float32` or `float64`.
    The power of the polynomial. Defaults to linear, 1.0.
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
def PiecewiseConstantSchedule(history, boundaries, values):
  """Piecewise constant from boundaries and interval values schedule.

 Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
      for the next 10000 steps, and 0.1 for any additional steps.

  Args:
    boundaries: A list of `int`s or `float`s with strictly
    increasing entries, and with all elements having the same type as the
    optimizer step.
    values: A list of `float`s or `int`s that specifies the
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

    pos = onp.searchsorted(boundaries, step_fl)
    return values[pos]

  return learning_rate


@gin.configurable(blacklist=["history"])
def InverseTimeDecaySchedule(history,
                             initial_learning_rate,
                             decay_steps,
                             decay_rate,
                             staircase=False):
  """Applies inverse time decay schedule.

  This schedule applies a polynomial decay function to an optimizer step,
    given a provided `initial_learning_rate`, to reach an `end_learning_rate`
    in the given `decay_steps`.

  It is computed as:
    ```python
    def decayed_learning_rate(step):
      return initial_learning_rate / (1 + decay_rate * step / decay_step)
    ```
    or, if `staircase` is `True`, as:
    ```python
    def decayed_learning_rate(step):
      return initial_learning_rate / (1 + decay_rate * floor(step / decay_step))
    ```

  Args:
    initial_learning_rate: A scalar `float32` or `float64`.
    The initial learning rate.
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
def CosineDecaySchedule(history, initial_learning_rate, decay_steps, alpha=0.0):
  """Applies cosine decay schedule.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  It is computed as:
  ```python
    def decayed_learning_rate(step):
      step = min(step, decay_steps)
      cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
      decayed = (1 - alpha) * cosine_decay + alpha
      return initial_learning_rate * decayed
  ```


  Args:
    initial_learning_rate: A scalar `float32` or `float64`.
    The initial learning rate.
    decay_steps: A scalar `int32` or `int64`.
    Number of steps to decay over.
    alpha: A scalar `float32` or `float64`.
    Minimum learning rate value as a fraction of initial_learning_rate.
 """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    step_fl = step.astype(np.float32)
    step_fl = np.minimun(step_fl, decay_steps)

    p = step_fl / decay_steps
    cosine_decayed = 0.5 * (1. + np.cos(p * np.pi))
    decayed = (1. - alpha) * cosine_decayed + alpha
    return decayed * initial_learning_rate

  return learning_rate


@gin.configurable(blacklist=["history"])
def CosineDecayRestartsSchedule(history,
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
    initial_learning_rate: A scalar `float32` or `float64`.
    The initial learning rate.
    first_decay_steps: A scalar `int32` or `int64`.
    Number of steps to decay over.
    t_mul: A scalar `float32` or `float64`.
    Used to derive the number of iterations in the i-th period
    m_mul: A scalar `float32` or `float64`.
    Used to derive the initial learning rate of the i-th period:
    alpha: A scalar `float32` or `float64`.
    Minimum learning rate value as a fraction of the initial_learning_rate.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    step_fl = step.astype(np.float32)
    completed_fraction = step_fl / first_decay_steps

    if t_mul == 1.0:
      i_restart = np.floor(completed_fraction)
      completed_fraction -= i_restart
    else:
      i_restart = np.log(1. - completed_fraction * (1. - t_mul)) / np.log(t_mul)
      i_restart = np.floor(i_restart)
      sum_r = (1. - np.power(t_mul, i_restart)) / (1. - t_mul)
      completed_fraction = (completed_fraction - sum_r) / np.power(
          t_mul, i_restart)

    m_fac = np.power(m_mul, i_restart)
    cosine_decayed = 0.5 * m_fac * (1. + np.cos(completed_fraction * np.pi))
    decayed = (1. - alpha) * cosine_decayed + alpha
    return decayed * initial_learning_rate

  return learning_rate


@gin.configurable(blacklist=["history"])
def LinearCosineDecaySchedule(history,
                              initial_learning_rate,
                              decay_steps,
                              num_periods=0.5,
                              alpha=0.0,
                              beta=0.001):
  """Applies linear cosine decay schedule.

    See [Bello et al., ICML2017] Neural Optimizer Search with RL.
    https://arxiv.org/abs/1709.07417

    For the idea of warm starts here controlled by `num_periods`,
    see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983

    Note that linear cosine decay is more aggressive than cosine decay and
    larger initial learning rates can typically be used.

    It is computed as:
    ```python
        def decayed_learning_rate(step):
          step = min(step, decay_steps)
          linear_decay = (decay_steps - step) / decay_steps
          cosine_decay = 0.5 * (
              1 + cos(pi * 2 * num_periods * step / decay_steps))
          decayed = (alpha + linear_decay) * cosine_decay + beta
          return initial_learning_rate * decayed
    ```


  Args:
    initial_learning_rate: A scalar `float32` or `float64`.
    The initial learning rate.
    decay_steps: A scalar `int32` or `int64`.
    Number of steps to decay over.
    num_periods: Number of periods in the cosine part of the decay.
    See computation above.
    alpha: See computation above.
    beta: See computation above.

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    step_fl = step.astype(np.float32)
    step_fl = np.minimun(step_fl, decay_steps)

    linear_decayed = (decay_steps - step_fl) / decay_steps
    completed_fraction = step_fl / decay_steps
    fraction = 2. * num_periods * completed_fraction

    cosine_decayed = 0.5 * (1. + np.cos(fraction * np.pi))
    linear_cosine_decayed = (alpha + linear_decayed) * cosine_decayed + beta
    return linear_cosine_decayed * initial_learning_rate

  return learning_rate


@gin.configurable(blacklist=["history"])
def NoisyLinearCosineDecaySchedule(history,
                                   initial_learning_rate,
                                   decay_steps,
                                   initial_variance=1.0,
                                   variance_decay=0.55,
                                   num_periods=0.5,
                                   alpha=0.0,
                                   beta=0.001,
                                   rng=None):
  """Applies noisy linear cosine decay schedule.

    See [Bello et al., ICML2017] Neural Optimizer Search with RL.
    https://arxiv.org/abs/1709.07417

    For the idea of warm starts here controlled by `num_periods`,
    see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
    with Warm Restarts. https://arxiv.org/abs/1608.03983

    Note that linear cosine decay is more aggressive than cosine decay and
    larger initial learning rates can typically be used.

     ```python
        def decayed_learning_rate(step):
          step = min(step, decay_steps)
          linear_decay = (decay_steps - step) / decay_steps)
          cosine_decay = 0.5 * (
              1 + cos(pi * 2 * num_periods * step / decay_steps))
          decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
          return initial_learning_rate * decayed
    ```
    where eps_t is 0-centered gaussian noise with variance
    initial_variance / (1 + global_step) ** variance_decay

  Args:
    initial_learning_rate: A scalar `float32` or `float64`.
    The initial learning rate.
    decay_steps: A scalar `int32` or `int64`.
    Number of steps to decay over.
    initial_variance: initial variance for the noise. See computation above.
    variance_decay: decay for the noise's variance. See computation above.
    num_periods: Number of periods in the cosine part of the decay.
    See computation above.
    alpha: See computation above.
    beta: See computation above.
    rng: Key for random number generation

  Returns:
    a function learning_rate(step): float -> float, the step-dependent lr.
  """
  del history

  def learning_rate(step):  # pylint: disable=invalid-name
    """Step to learning rate function."""
    step_fl = step.astype(np.float32)
    step_fl = np.minimun(step_fl, decay_steps)

    variance = initial_variance / (np.power(1. + step_fl, variance_decay))
    std = np.sqrt(variance)

    linear_decayed = (decay_steps - step_fl) / decay_steps
    noisy_linear_decayed = linear_decayed + jax_random.random_normal(
        rng, shape=linear_decayed.shape) * std

    completed_fraction = step_fl / decay_steps
    fraction = 2. * num_periods * completed_fraction

    cosine_decayed = 0.5 * (1. + np.cos(fraction * np.pi))
    noisy_linear_cosine_decayed = (alpha +
                                   noisy_linear_decayed) * cosine_decayed + beta
    return noisy_linear_cosine_decayed * initial_learning_rate

  return learning_rate

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
