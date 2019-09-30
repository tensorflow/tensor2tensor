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

"""PPO in JAX.

Notation:

B, scalar   - batch size
RT, scalar  - (reward time) number of time-steps in a trajectory, or the size
              of the padded reward sequence.
AT, scalar  - (action time) number of controls in a trajectory, or the size
              of the policy network output.
OBS, tuple  - shape of a singular observation from the environment.
             Ex: For CartPole-v0 this is (4,) and Pong-v0 it's (210, 160, 3)
A, scalar   - Number of actions, assuming a discrete space.

Policy and Value function signatures:

Policy            Function :: [B, RT + 1] + OBS ->  [B, AT, A]
Value             Function :: [B, RT + 1] + OBS ->  [B, AT]
Policy and Value  Function :: [B, RT + 1] + OBS -> ([B, AT, A], [B, AT])

i.e. the policy net should take a batch of *trajectories* and at each time-step
in each batch deliver a probability distribution over actions.

NOTE: It doesn't return logits, rather the expectation is that it returns
log-probabilities instead.

NOTE: The policy and value functions need to take care to not take into account
future time-steps while deciding the actions (or value) for the current
time-step.

Policy and Value Function produces a tuple of the expected output of a policy
function and a value function.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import os
import re
import time

from absl import logging
from jax import grad
from jax import jit
from jax import lax
from jax import numpy as np
import numpy as onp

from tensor2tensor.envs import env_problem
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.trax import layers as tl
from tensor2tensor.trax import utils
from tensorflow.io import gfile


def policy_and_value_net(
    n_actions, n_controls, vocab_size, bottom_layers_fn, two_towers
):
  """A policy and value net function."""

  # Layers.

  # Now, with the current logits, one head computes action probabilities and the
  # other computes the value function.
  # NOTE: The LogSoftmax instead of the Softmax because of numerical stability.

  @tl.layer()
  def FlattenControlsIntoTime(x, **unused_kwargs):  # pylint: disable=invalid-name
    """Splits logits for actions in different controls and flattens controls."""
    return np.reshape(x, (x.shape[0], -1, n_actions))

  if vocab_size is None:
    # In continuous policies every element of the output sequence corresponds to
    # an observation.
    n_preds_per_input = n_controls
    kwargs = {}
  else:
    # In discrete policies every element of the output sequence corresponds to
    # a symbol in the discrete representation, and each control takes 1 symbol.
    n_preds_per_input = 1
    kwargs = {"vocab_size": vocab_size}

  if two_towers:
    layers = [
        tl.Dup(),
        tl.Parallel(
            [bottom_layers_fn(**kwargs),
             tl.Dense(n_preds_per_input * n_actions),
             FlattenControlsIntoTime(),  # pylint: disable=no-value-for-parameter
             tl.LogSoftmax()],
            [bottom_layers_fn(**kwargs),
             tl.Dense(n_preds_per_input),
             tl.Flatten()],
        )
    ]
  else:
    layers = [
        bottom_layers_fn(**kwargs),
        tl.Dup(),
        tl.Parallel(
            [tl.Dense(n_preds_per_input * n_actions),
             FlattenControlsIntoTime(),  # pylint: disable=no-value-for-parameter
             tl.LogSoftmax()],
            [tl.Dense(n_preds_per_input), tl.Flatten()],
        )
    ]
  return tl.Model(layers)


def optimizer_fn(optimizer, net_params):
  """Exposes a convenient interface for the optimizer.

  Args:
    optimizer: Optimizer class to use.
    net_params: A nested structure of network parameters.

  Returns:
    A tuple (opt_state, opt_update, get_params), where:
      opt_state: Pair (net_params, opt_slots) - initial optimization state.
      opt_update: Function (step, grads, opt_state) -> opt_state doing one
        optimization step.
      get_params: Function opt_state -> net_params for extracting the network
        parameters from the optimization state.
  """
  opt = optimizer()
  (init_slots, init_nontrainable_slots) = opt.tree_init(net_params)
  init_state = (net_params, init_slots)

  def opt_update(step, grads, opt_state):
    (params, slots) = opt_state
    # Pass the initial nontrainable_slots as we don't tune them during training.
    # (yet!)
    return opt.tree_update(step, grads, params, slots, init_nontrainable_slots)

  def get_params(opt_state):
    (params, _) = opt_state
    return params

  return init_state, opt_update, get_params


# Should this be collect 'n' trajectories, or
# Run the env for 'n' steps and take completed trajectories, or
# Any other option?
def collect_trajectories(env,
                         policy_fn,
                         n_trajectories=1,
                         max_timestep=None,
                         reset=True,
                         len_history_for_policy=32,
                         boundary=32,
                         state=None,
                         temperature=1.0,
                         rng=None,
                         abort_fn=None,
                         raw_trajectory=False,):
  """Collect trajectories with the given policy net and behaviour.

  Args:
    env: A gym env interface, for now this is not-batched.
    policy_fn: observations(B,RT+1) -> log-probabs(B, AT, A) callable.
    n_trajectories: int, number of trajectories.
    max_timestep: int or None, the index of the maximum time-step at which we
      return the trajectory, None for ending a trajectory only when env returns
      done.
    reset: bool, true if we want to reset the envs. The envs are also reset if
      max_max_timestep is None or < 0
    len_history_for_policy: int or None, the maximum history to keep for
      applying the policy on. If None, use the full history.
    boundary: int, pad the sequences to the multiples of this number.
    state: state for `policy_fn`.
    temperature: (float) temperature to sample action from policy_fn.
    rng: jax rng, splittable.
    abort_fn: callable, If not None, then at every env step call and abort the
      trajectory collection if it returns True, if so reset the env and return
      None.
    raw_trajectory: bool, if True a list of trajectory.Trajectory objects is
      returned, otherwise a list of numpy representations of
      `trajectory.Trajectory` is returned.

  Returns:
    A tuple (trajectory, number of trajectories that are done)
    trajectory: list of (observation, action, reward) tuples, where each element
    `i` is a tuple of numpy arrays with shapes as follows:
    observation[i] = (B, T_i + 1)
    action[i] = (B, T_i)
    reward[i] = (B, T_i)
  """

  assert isinstance(env, env_problem.EnvProblem)
  # This is an env_problem, run its collect function.
  trajs, n_done, timing_info, state = env_problem_utils.play_env_problem_with_policy(
      env,
      policy_fn,
      num_trajectories=n_trajectories,
      max_timestep=max_timestep,
      reset=reset,
      len_history_for_policy=len_history_for_policy,
      boundary=boundary,
      state=state,
      temperature=temperature,
      rng=rng,
      abort_fn=abort_fn,
      raw_trajectory=raw_trajectory,
  )
  # Skip returning raw_rewards here, since they aren't used.

  # t is the return value of Trajectory.as_numpy, so:
  # (observation, action, processed_reward, raw_reward, infos)
  return trajs, n_done, timing_info, state


# This function can probably be simplified, ask how?
# Can we do something much simpler than lax.pad, maybe np.pad?
# Others?


def get_padding_value(dtype):
  """Returns the padding value given a dtype."""
  padding_value = None
  if dtype == np.uint8:
    padding_value = np.uint8(0)
  elif dtype == np.uint16:
    padding_value = np.uint16(0)
  elif dtype == np.float32 or dtype == np.float64:
    padding_value = 0.0
  else:
    padding_value = 0
  assert padding_value is not None
  return padding_value


# TODO(afrozm): Use np.pad instead and make jittable?
def pad_trajectories(trajectories, boundary=20):
  """Pad trajectories to a bucket length that is a multiple of boundary.

  Args:
    trajectories: list[(observation, actions, rewards)], where each observation
      is shaped (t+1,) + OBS and actions & rewards are shaped (t,), with the
      length of the list being B (batch size).
    boundary: int, bucket length, the actions and rewards are padded to integer
      multiples of boundary.

  Returns:
    tuple: (padding lengths, reward_mask, padded_observations, padded_actions,
        padded_rewards) where padded_observations is shaped (B, RT+1) + OBS and
        padded_actions, padded_rewards & reward_mask are shaped (B, RT).
        Where RT is max(t) rounded up to an integer multiple of boundary.
        padded_length is how much padding we've added and
        reward_mask is 1s for actual rewards and 0s for the padding.
  """

  # Let's compute max(t) over all trajectories.
  t_max = max(r.shape[0] for (_, _, r, _) in trajectories)

  # t_max is rounded to the next multiple of `boundary`
  boundary = int(boundary)
  bucket_length = boundary * int(np.ceil(float(t_max) / boundary))

  # So all obs will be padded to t_max + 1 and actions and rewards to t_max.
  padded_observations = []
  padded_actions = []
  padded_rewards = []
  padded_infos = collections.defaultdict(list)
  padded_lengths = []
  reward_masks = []

  for (o, a, r, i) in trajectories:
    # Determine the amount to pad, this holds true for obs, actions and rewards.
    num_to_pad = bucket_length + 1 - o.shape[0]
    padded_lengths.append(num_to_pad)
    if num_to_pad == 0:
      padded_observations.append(o)
      padded_actions.append(a)
      padded_rewards.append(r)
      reward_masks.append(onp.ones_like(r, dtype=np.int32))
      if i:
        for k, v in i.items():
          padded_infos[k].append(v)
      continue

    # First pad observations.
    padding_config = tuple([(0, num_to_pad, 0)] + [(0, 0, 0)] * (o.ndim - 1))

    padding_value = get_padding_value(o.dtype)
    action_padding_value = get_padding_value(a.dtype)
    reward_padding_value = get_padding_value(r.dtype)

    padded_obs = lax.pad(o, padding_value, padding_config)
    padded_observations.append(padded_obs)

    # Now pad actions and rewards.
    padding_config = tuple([(0, num_to_pad, 0)] + [(0, 0, 0)] * (a.ndim - 1))
    padded_action = lax.pad(a, action_padding_value, padding_config)
    padded_actions.append(padded_action)

    assert r.ndim == 1
    padding_config = ((0, num_to_pad, 0),)
    padded_reward = lax.pad(r, reward_padding_value, padding_config)
    padded_rewards.append(padded_reward)

    # Also create the mask to use later.
    reward_mask = onp.ones_like(r, dtype=np.int32)
    reward_masks.append(lax.pad(reward_mask, 0, padding_config))

    if i:
      for k, v in i.items():
        # Create a padding configuration for this value.
        padding_config = [(0, num_to_pad, 0)] + [(0, 0, 0)] * (v.ndim - 1)
        padded_infos[k].append(lax.pad(v, 0.0, tuple(padding_config)))

  # Now stack these padded_infos if they exist.
  stacked_padded_infos = None
  if padded_infos:
    stacked_padded_infos = {k: np.stack(v) for k, v in padded_infos.items()}

  return padded_lengths, np.stack(reward_masks), np.stack(
      padded_observations), np.stack(padded_actions), np.stack(
          padded_rewards), stacked_padded_infos


def rewards_to_go(rewards, mask, gamma=0.99):
  r"""Computes rewards to go.

  Reward to go is defined as follows, the discounted reward that we have to
  yet collect, going forward from this point, i.e.:

  r2g_t = \sum_{l=0}^{\infty} (\gamma^{l} * reward_{t+l})

  Args:
    rewards: np.ndarray of shape (B, RT) of rewards.
    mask: np.ndarray of shape (B, RT) of mask for the rewards.
    gamma: float, discount factor.

  Returns:
    rewards to go, np.ndarray of shape (B, RT).
  """
  B, RT = rewards.shape  # pylint: disable=invalid-name,unused-variable

  masked_rewards = rewards * mask  # (B, RT)

  # The lax.scan version of this is slow, but we still show it here for
  # completeness.
  #   rewards_rev = np.flip(masked_rewards, axis=1)  # (B, T) flipped on time.
  #   rrt = np.transpose(rewards_rev)  # (T, B) transpose to scan over time.
  #
  #   def discounting_add(carry, reward):
  #     x = reward + (gamma * carry)
  #     return x, x
  #
  #   _, ys = lax.scan(discounting_add,
  #                    np.zeros_like(rrt[0], dtype=np.float32),
  #                    rrt.astype(np.float32))
  #
  #   # ys is (T, B) and T is in reverse order.
  #   return np.flip(np.transpose(ys), axis=1)

  # We use the following recurrence relation, derived from the equation above:
  #
  # r2g[t+1] = (r2g[t] - r[t]) / gamma
  #
  # This means we'll need to calculate r2g[0] first and then r2g[1] and so on ..
  #
  # **However** this leads to overflows for long sequences: r2g[t] - r[t] > 0
  # and gamma < 1.0, so the division keeps increasing.
  #
  # So we just run the recurrence in reverse, i.e.
  #
  # r2g[t] = r[t] + (gamma*r2g[t+1])
  #
  # This is much better, but might have lost updates since the (small) rewards
  # at earlier time-steps may get added to a (very?) large sum.

  # Compute r2g_{T-1} at the start and then compute backwards in time.
  r2gs = [masked_rewards[:, -1]]

  # Go from T-2 down to 0.
  for t in reversed(range(RT - 1)):
    r2gs.append(masked_rewards[:, t] + (gamma * r2gs[-1]))

  # The list should have length RT.
  assert RT == len(r2gs)

  # First we stack them in the correct way to make it (B, RT), but these are
  # still from newest (RT-1) to oldest (0), so then we flip it on time axis.
  return np.flip(np.stack(r2gs, axis=1), axis=1)


@jit
def value_loss_given_predictions(value_prediction,
                                 rewards,
                                 reward_mask,
                                 gamma=0.99,
                                 epsilon=0.2,
                                 value_prediction_old=None):
  """Computes the value loss given the prediction of the value function.

  Args:
    value_prediction: np.ndarray of shape (B, RT+1, 1)
    rewards: np.ndarray of shape (B, RT) of rewards.
    reward_mask: np.ndarray of shape (B, RT), the mask over rewards.
    gamma: float, discount factor.
    epsilon: float, clip-fraction, used if value_value_prediction_old isn't None
    value_prediction_old: np.ndarray of shape (B, RT+1, 1) of value predictions
      using the old parameters. If provided, we incorporate this in the loss as
      well. This is from the OpenAI baselines implementation.

  Returns:
    Pair (value_loss, summaries), where value_loss is the average L2 value loss,
      averaged over instances where reward_mask is 1. Summaries is a dict of
      summaries collected during value loss computation.
  """

  B, RT = rewards.shape  # pylint: disable=invalid-name
  assert (B, RT) == reward_mask.shape
  assert (B, RT + 1) == value_prediction.shape

  value_prediction = value_prediction[:, :-1] * reward_mask  # (B, RT)
  r2g = rewards_to_go(rewards, reward_mask, gamma=gamma)  # (B, RT)
  loss = (value_prediction - r2g)**2

  # From the baselines implementation.
  if value_prediction_old is not None:
    value_prediction_old = value_prediction_old[:, :-1] * reward_mask  # (B, RT)

    v_clipped = value_prediction_old + np.clip(
        value_prediction - value_prediction_old, -epsilon, epsilon)
    v_clipped_loss = (v_clipped - r2g)**2
    loss = np.maximum(v_clipped_loss, loss)

  # Take an average on only the points where mask != 0.
  value_loss = np.sum(loss) / np.sum(reward_mask)

  summaries = {
      "value_loss": value_loss,
  }

  return (value_loss, summaries)


def deltas(predicted_values, rewards, mask, gamma=0.99):
  r"""Computes TD-residuals from V(s) and rewards.

  Where a `delta`, i.e. a td-residual is defined as:

  delta_{b,t} = r_{b,t} + \gamma * v_{b,t+1} - v_{b,t}.

  Args:
    predicted_values: ndarray of shape (B, RT+1). NOTE: Expects axis 2 was
      squeezed. These represent V(s_bt) for b < B and t < RT+1
    rewards: ndarray of shape (B, RT) of rewards.
    mask: ndarray of shape (B, RT) of mask for rewards.
    gamma: float, discount factor.

  Returns:
    ndarray of shape (B, RT) of one-step TD-residuals.
  """

  # Predicted values at time t, cutting off the last to have shape (B, RT).
  predicted_values_bt = predicted_values[:, :-1]
  # Predicted values at time t+1, by cutting off the first to have shape (B, RT)
  predicted_values_btplus1 = predicted_values[:, 1:]
  # Return the deltas as defined above.
  return (rewards +
          (gamma * predicted_values_btplus1) - predicted_values_bt) * mask


def gae_advantages(td_deltas, mask, lambda_=0.95, gamma=0.99):
  r"""Computes the GAE advantages given the one step TD-residuals.

  The formula for a GAE advantage estimator is as follows:

  A_{bt} = \sum_{l=0}^{\infty}(\gamma * \lambda)^{l}(\delta_{b,t+l}).

  Internally we just call rewards_to_go, since it is the same computation.

  Args:
    td_deltas: np.ndarray of shape (B, RT) of one step TD-residuals.
    mask: np.ndarray of shape (B, T) of mask for the residuals. It maybe the
      case that the `td_deltas` are already masked correctly since they are
      produced by `deltas(...)`
    lambda_: float, lambda parameter for GAE estimators.
    gamma: float, lambda parameter for GAE estimators.

  Returns:
    GAE advantage estimates.
  """

  return rewards_to_go(td_deltas, mask, lambda_ * gamma)


def chosen_probabs(probab_actions, actions):
  """Picks out the probabilities of the actions along batch and time-steps.

  Args:
    probab_actions: ndarray of shape `[B, AT, A]`, where
      probab_actions[b, t, i] contains the log-probability of action = i at
      the t^th time-step in the b^th trajectory.
    actions: ndarray of shape `[B, AT]`, with each entry in [0, A) denoting
      which action was chosen in the b^th trajectory's t^th time-step.

  Returns:
    `[B, AT, A]` ndarray with the log-probabilities of the chosen actions.
  """
  B, AT = actions.shape  # pylint: disable=invalid-name
  assert (B, AT) == probab_actions.shape[:2]
  return probab_actions[np.arange(B)[:, None], np.arange(AT), actions]


def compute_probab_ratios(p_new, p_old, actions, action_mask):
  """Computes the probability ratios for each time-step in a trajectory.

  Args:
    p_new: ndarray of shape [B, AT, A] of the log-probabilities that the
      policy network assigns to all the actions at each time-step in each batch
      using the old parameters.
    p_old: ndarray of shape [B, AT, A], same as above, but using old policy
      network parameters.
    actions: ndarray of shape [B, AT] where each element is from [0, A).
    action_mask: ndarray of shape [B, T] masking over probabilities.

  Returns:
    probab_ratios: ndarray of shape [B, AT], where
    probab_ratios_{b,t,} = p_new_{b,t,action_{b,t}} /
                           p_old_{b,t,action_{b,t}}
  """

  B, AT = actions.shape  # pylint: disable=invalid-name
  assert (B, AT) == p_old.shape[:2]
  assert (B, AT) == p_new.shape[:2]

  logp_old = chosen_probabs(p_old, actions)
  logp_new = chosen_probabs(p_new, actions)

  assert (B, AT) == logp_old.shape
  assert (B, AT) == logp_new.shape

  # Since these are log-probabilities, we just subtract them.
  probab_ratios = np.exp(logp_new - logp_old) * action_mask
  assert (B, AT) == probab_ratios.shape
  return probab_ratios


def clipped_probab_ratios(probab_ratios, epsilon=0.2):
  return np.clip(probab_ratios, 1 - epsilon, 1 + epsilon)


def clipped_objective(probab_ratios, advantages, action_mask, epsilon=0.2):
  advantages = advantages
  return np.minimum(
      probab_ratios * advantages,
      clipped_probab_ratios(probab_ratios, epsilon=epsilon) *
      advantages) * action_mask


@jit
def ppo_loss_given_predictions(log_probab_actions_new,
                               log_probab_actions_old,
                               value_predictions_old,
                               padded_actions,
                               rewards_to_actions,
                               padded_rewards,
                               reward_mask,
                               gamma=0.99,
                               lambda_=0.95,
                               epsilon=0.2):
  """PPO objective, with an eventual minus sign, given predictions."""
  B, RT = padded_rewards.shape  # pylint: disable=invalid-name
  _, AT, A = log_probab_actions_old.shape  # pylint: disable=invalid-name

  assert (B, RT) == padded_rewards.shape
  assert (B, AT) == padded_actions.shape
  assert (B, RT) == reward_mask.shape

  assert (B, RT + 1) == value_predictions_old.shape
  assert (B, AT, A) == log_probab_actions_old.shape
  assert (B, AT, A) == log_probab_actions_new.shape

  assert (RT + 1, AT) == rewards_to_actions.shape

  # (B, RT)
  td_deltas = deltas(
      value_predictions_old,  # (B, RT+1)
      padded_rewards,
      reward_mask,
      gamma=gamma)

  # (B, RT)
  advantages = gae_advantages(
      td_deltas, reward_mask, lambda_=lambda_, gamma=gamma)

  # Normalize the advantages.
  advantage_mean = np.mean(advantages)
  advantage_std = np.std(advantages)
  advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

  # Scatter advantages over padded_actions.
  # rewards_to_actions is RT + 1 -> AT, so we pad the advantages and the reward
  # mask by 1.
  advantages = np.dot(np.pad(advantages, ((0, 0), (0, 1))), rewards_to_actions)
  action_mask = np.dot(
      np.pad(reward_mask, ((0, 0), (0, 1))), rewards_to_actions
  )

  # (B, AT)
  ratios = compute_probab_ratios(log_probab_actions_new, log_probab_actions_old,
                                 padded_actions, action_mask)
  assert (B, AT) == ratios.shape

  # (B, AT)
  objective = clipped_objective(
      ratios, advantages, action_mask, epsilon=epsilon)
  assert (B, AT) == objective.shape

  # ()
  average_objective = np.sum(objective) / np.sum(action_mask)

  # Loss is negative objective.
  ppo_loss = -average_objective

  summaries = {
      "ppo_loss": ppo_loss,
      "advantage_mean": advantage_mean,
      "advantage_std": advantage_std,
  }

  return (ppo_loss, summaries)


@jit
def combined_loss_given_predictions(log_probab_actions_new,
                                    log_probab_actions_old,
                                    value_prediction_new,
                                    value_prediction_old,
                                    padded_actions,
                                    rewards_to_actions,
                                    padded_rewards,
                                    reward_mask,
                                    gamma=0.99,
                                    lambda_=0.95,
                                    epsilon=0.2,
                                    c1=1.0,
                                    c2=0.01):
  """Computes the combined (clipped loss + value loss) given predictions."""
  # Sum values over symbols in an action's representation, because it's a simple
  # way of going from AT to RT+1 and does not decrease the expressive power.
  value_prediction_old = np.dot(
      value_prediction_old, rewards_to_actions.transpose()
  )
  value_prediction_new = np.dot(
      value_prediction_new, rewards_to_actions.transpose()
  )
  (value_loss, value_summaries) = value_loss_given_predictions(
      value_prediction_new,
      padded_rewards,
      reward_mask,
      gamma=gamma,
      value_prediction_old=value_prediction_old,
      epsilon=epsilon)
  (ppo_loss, ppo_summaries) = ppo_loss_given_predictions(
      log_probab_actions_new,
      log_probab_actions_old,
      value_prediction_old,
      padded_actions,
      rewards_to_actions,
      padded_rewards,
      reward_mask,
      gamma=gamma,
      lambda_=lambda_,
      epsilon=epsilon)
  # Pad the reward mask to be compatible with rewards_to_actions.
  padded_reward_mask = np.pad(reward_mask, ((0, 0), (0, 1)))
  action_mask = np.dot(padded_reward_mask, rewards_to_actions)
  entropy_bonus = masked_entropy(log_probab_actions_new, action_mask)
  combined_loss_ = ppo_loss + (c1 * value_loss) - (c2 * entropy_bonus)

  summaries = {
      "combined_loss": combined_loss_,
      "entropy_bonus": entropy_bonus,
  }
  for loss_summaries in (value_summaries, ppo_summaries):
    summaries.update(loss_summaries)

  return (combined_loss_, (ppo_loss, value_loss, entropy_bonus), summaries)


@functools.partial(jit, static_argnums=(3,))
def combined_loss(new_params,
                  log_probab_actions_old,
                  value_predictions_old,
                  policy_and_value_net_apply,
                  padded_observations,
                  padded_actions,
                  rewards_to_actions,
                  padded_rewards,
                  reward_mask,
                  gamma=0.99,
                  lambda_=0.95,
                  epsilon=0.2,
                  c1=1.0,
                  c2=0.01,
                  state=None,
                  rng=None):
  """Computes the combined (clipped loss + value loss) given observations."""
  (log_probab_actions_new, value_predictions_new) = (
      policy_and_value_net_apply(
          padded_observations, params=new_params, state=state, rng=rng))

  (loss, component_losses, summaries) = combined_loss_given_predictions(
      log_probab_actions_new,
      log_probab_actions_old,
      value_predictions_new,
      value_predictions_old,
      padded_actions,
      rewards_to_actions,
      padded_rewards,
      reward_mask,
      gamma=gamma,
      lambda_=lambda_,
      epsilon=epsilon,
      c1=c1,
      c2=c2,
  )
  return (loss, component_losses, summaries, state)


@functools.partial(jit, static_argnums=(2, 3, 4))
def policy_and_value_opt_step(i,
                              opt_state,
                              opt_update,
                              get_params,
                              policy_and_value_net_apply,
                              log_probab_actions_old,
                              value_predictions_old,
                              padded_observations,
                              padded_actions,
                              rewards_to_actions,
                              padded_rewards,
                              reward_mask,
                              c1=1.0,
                              c2=0.01,
                              gamma=0.99,
                              lambda_=0.95,
                              epsilon=0.1,
                              state=None,
                              rng=None):
  """Policy and Value optimizer step."""

  # Combined loss function given the new params.
  def policy_and_value_loss(params, state):
    """Returns the combined loss given just parameters."""
    (loss, _, _, state) = combined_loss(
        params,
        log_probab_actions_old,
        value_predictions_old,
        policy_and_value_net_apply,
        padded_observations,
        padded_actions,
        rewards_to_actions,
        padded_rewards,
        reward_mask,
        c1=c1,
        c2=c2,
        gamma=gamma,
        lambda_=lambda_,
        epsilon=epsilon,
        state=state,
        rng=rng)
    return loss, state

  new_params = get_params(opt_state)
  g, state = grad(policy_and_value_loss, has_aux=True)(new_params, state)
  # TODO(afrozm): Maybe clip gradients?
  return opt_update(i, g, opt_state), state


def get_time(t1, t2=None):
  if t2 is None:
    t2 = time.time()
  return round((t2 - t1) * 1000, 2)


def approximate_kl(log_prob_new, log_prob_old, mask):
  """Computes the approximate KL divergence between the old and new log-probs.

  Args:
    log_prob_new: (B, AT, A) log probs new
    log_prob_old: (B, AT, A) log probs old
    mask: (B, AT)

  Returns:
    Approximate KL.
  """
  diff = log_prob_old - log_prob_new
  # Mask out the irrelevant part.
  diff *= mask[:, :, np.newaxis]  # make mask (B, RT, 1)
  # Average on non-masked part.
  return np.sum(diff) / np.sum(mask)


def masked_entropy(log_probs, mask):
  """Computes the entropy for the given log-probs.

  Args:
    log_probs: (B, AT, A) log probs
    mask: (B, AT) mask.

  Returns:
    Entropy.
  """
  # Mask out the irrelevant part.
  lp = log_probs * mask[:, :, np.newaxis]  # make mask (B, RT, 1)
  p = np.exp(lp) * mask[:, :, np.newaxis]  # (B, RT, 1)
  # Average on non-masked part and take negative.
  return -(np.sum(lp * p) / np.sum(mask))


def get_policy_model_files(output_dir):
  return list(
      reversed(
          sorted(gfile.glob(os.path.join(output_dir, "model-??????.pkl")))))


def get_epoch_from_policy_model_file(policy_model_file):
  base_name = os.path.basename(policy_model_file)
  return int(re.match(r"model-(\d+).pkl", base_name).groups()[0])


def get_policy_model_file_from_epoch(output_dir, epoch):
  return os.path.join(output_dir, "model-%06d.pkl" % epoch)


def maybe_restore_opt_state(output_dir,
                            policy_and_value_opt_state=None,
                            policy_and_value_state=None):
  """Maybe restore the optimization state from the checkpoint dir.

  Optimization state includes parameters and optimizer slots.

  Args:
    output_dir: Directory where saved model checkpoints are stored.
    policy_and_value_opt_state: Default optimization state, returned if model
      isn't found.
    policy_and_value_state: state of the policy and value network.

  Returns:
    tuple (opt_state, state, epoch (int), opt_step (int)) where epoch is the
    epoch from which we restored the optimization state, 0 if no checkpoint was
    found, and opt_step is the total optimization step (sum of all optimization
    steps made up to the current epoch).
  """
  pkl_module = utils.get_pickle_module()
  epoch = 0
  total_opt_step = 0
  for model_file in get_policy_model_files(output_dir):
    logging.info("Trying to restore model from %s", model_file)
    try:
      with gfile.GFile(model_file, "rb") as f:
        policy_and_value_opt_state, policy_and_value_state, total_opt_step = (
            pkl_module.load(f))
      epoch = get_epoch_from_policy_model_file(model_file)
      break
    except EOFError as e:
      logging.error("Unable to load model from: %s with %s", model_file, e)
      # Try an older version.
      continue
  return (
      policy_and_value_opt_state,
      policy_and_value_state,
      epoch,
      total_opt_step,
  )


LAST_N_POLICY_MODELS_TO_KEEP = 5


def save_opt_state(output_dir,
                   policy_and_value_opt_state,
                   policy_and_value_state,
                   epoch,
                   total_opt_step):
  """Saves the policy and value network optimization state etc."""
  pkl_module = utils.get_pickle_module()
  old_model_files = get_policy_model_files(output_dir)
  params_file = os.path.join(output_dir, "model-%06d.pkl" % epoch)
  with gfile.GFile(params_file, "wb") as f:
    pkl_module.dump(
        (policy_and_value_opt_state, policy_and_value_state, total_opt_step), f)
  # Keep the last k model files lying around (note k > 1 because the latest
  # model file might be in the process of getting read async).
  for path in old_model_files[LAST_N_POLICY_MODELS_TO_KEEP:]:
    if path != params_file:
      gfile.remove(path)


def init_policy_from_world_model_checkpoint(policy_params, model_output_dir):
  """Initializes policy parameters from world model parameters."""
  pkl_module = utils.get_pickle_module()
  params_file = os.path.join(model_output_dir, "model.pkl")
  # Don't use trax.restore_state to avoid a circular import.
  with gfile.GFile(params_file, "rb") as f:
    model_params = pkl_module.load(f)[0][0]
  # TODO(pkozakowski): The following, brittle line of code is hardcoded for
  # transplanting parameters from TransformerLM to TransformerDecoder-based
  # policy network of the same configuration. Figure out a more general method.
  policy_params[0] = model_params[1:-2]
  return policy_params


def write_eval_reward_summaries(reward_stats_by_mode, summary_writer, epoch):
  """Writes evaluation reward statistics to summary and logs them.

  Args:
    reward_stats_by_mode: Nested dict of structure: {
          "raw": {
              <temperature 1>: {
                  "mean": <reward mean>,
                  "std": <reward std>, },
              <temperature 2>: ... },
          "processed": ... }
    summary_writer: jaxboard.SummaryWriter.
    epoch: Current epoch number.
  """
  for (reward_mode, reward_stats_by_temp) in reward_stats_by_mode.items():
    for (temperature, reward_stats) in reward_stats_by_temp.items():
      for (stat_name, stat) in reward_stats.items():
        summary_writer.scalar(
            "eval/{reward_mode}_reward_{stat_name}/"
            "temperature_{temperature}".format(
                reward_mode=reward_mode,
                stat_name=stat_name,
                temperature=temperature),
            stat,
            step=epoch)
      logging.info(
          "Epoch [% 6d] Policy Evaluation (%s reward) "
          "[temperature %.2f] = %10.2f (+/- %.2f)", epoch, reward_mode,
          temperature, reward_stats["mean"], reward_stats["std"])


def shuffled_index_batches(dataset_size, batch_size):
  """Generates batches of shuffled indices over a dataset."""
  def shuffled_indices():
    while True:
      perm = onp.random.permutation(dataset_size)
      for x in perm:
        yield x

  indices = shuffled_indices()
  while True:
    yield onp.array(list(itertools.islice(indices, int(batch_size))))
