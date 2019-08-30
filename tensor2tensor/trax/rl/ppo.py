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

B, scalar  - batch size
T, scalar  - number of time-steps in a trajectory, or the value of the padded
             time-step dimension.
OBS, tuple - shape of a singular observation from the environment.
             Ex: For CartPole-v0 this is (4,) and Pong-v0 it's (210, 160, 3)
A, scalar  - Number of actions, assuming a discrete space.

Policy and Value function signatures:

Policy            Function :: [B, T] + OBS ->  [B, T, A]
Value             Function :: [B, T] + OBS ->  [B, T, 1]
Policy and Value  Function :: [B, T] + OBS -> ([B, T, A], [B, T, 1])

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
import os
import time

from absl import logging
import cloudpickle as pickle
from jax import grad
from jax import jit
from jax import lax
from jax import numpy as np
from jax import random as jax_random
import numpy as onp
from tensor2tensor.envs import env_problem
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.trax import layers as tl
from tensorflow.io import gfile


def policy_and_value_net(rng_key,
                         batch_observations_shape,
                         observations_dtype,
                         n_actions,
                         bottom_layers_fn=(),
                         two_towers=True):
  """A policy and value net function."""

  # Layers.

  # Now, with the current logits, one head computes action probabilities and the
  # other computes the value function.
  # NOTE: The LogSoftmax instead of the Softmax because of numerical stability.

  if two_towers:
    layers = [
        tl.Dup(),
        tl.Parallel(
            [bottom_layers_fn(), tl.Dense(n_actions), tl.LogSoftmax()],
            [bottom_layers_fn(), tl.Dense(1)],
        )
    ]
  else:
    layers = [
        bottom_layers_fn(),
        tl.Dup(),
        tl.Parallel(
            [tl.Dense(n_actions), tl.LogSoftmax()],
            [tl.Dense(1)],
        )
    ]
  net = tl.Model(layers)
  params, state = net.initialize(batch_observations_shape, observations_dtype,
                                 rng_key)
  return params, state, net


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
                         rng=None):
  """Collect trajectories with the given policy net and behaviour.

  Args:
    env: A gym env interface, for now this is not-batched.
    policy_fn: observations(B,T+1) -> log-probabs(B,T+1, A) callable.
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
    rng: jax rng, splittable.

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
      rng=rng)
  # Skip returning raw_rewards here, since they aren't used.

  # t is the return value of Trajectory.as_numpy, so:
  # (observation, action, processed_reward, raw_reward, infos)
  return [(t[0], t[1], t[2], t[4]) for t in trajs], n_done, timing_info, state


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
        padded_rewards) where padded_observations is shaped (B, T+1) + OBS and
        padded_actions, padded_rewards & reward_mask are shaped (B, T).
        Where T is max(t) rounded up to an integer multiple of boundary.
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
    assert a.ndim == 1 and r.ndim == 1
    padding_config = ((0, num_to_pad, 0),)

    padded_action = lax.pad(a, action_padding_value, padding_config)
    padded_actions.append(padded_action)
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
    rewards: np.ndarray of shape (B, T) of rewards.
    mask: np.ndarray of shape (B, T) of mask for the rewards.
    gamma: float, discount factor.

  Returns:
    rewards to go, np.ndarray of shape (B, T).
  """
  B, T = rewards.shape  # pylint: disable=invalid-name,unused-variable

  masked_rewards = rewards * mask  # (B, T)

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
  for t in reversed(range(T - 1)):
    r2gs.append(masked_rewards[:, t] + (gamma * r2gs[-1]))

  # The list should have length T.
  assert T == len(r2gs)

  # First we stack them in the correct way to make it (B, T), but these are
  # still from newest (T-1) to oldest (0), so then we flip it on time axis.
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
    value_prediction: np.ndarray of shape (B, T+1, 1)
    rewards: np.ndarray of shape (B, T) of rewards.
    reward_mask: np.ndarray of shape (B, T), the mask over rewards.
    gamma: float, discount factor.
    epsilon: float, clip-fraction, used if value_value_prediction_old isn't None
    value_prediction_old: np.ndarray of shape (B, T+1, 1) of value predictions
      using the old parameters. If provided, we incorporate this in the loss as
      well. This is from the OpenAI baselines implementation.

  Returns:
    Pair (value_loss, summaries), where value_loss is the average L2 value loss,
      averaged over instances where reward_mask is 1. Summaries is a dict of
      summaries collected during value loss computation.
  """

  B, T = rewards.shape  # pylint: disable=invalid-name
  assert (B, T) == reward_mask.shape
  assert (B, T + 1, 1) == value_prediction.shape

  value_prediction = np.squeeze(value_prediction, axis=2)  # (B, T+1)
  value_prediction = value_prediction[:, :-1] * reward_mask  # (B, T)
  r2g = rewards_to_go(rewards, reward_mask, gamma=gamma)  # (B, T)
  loss = (value_prediction - r2g)**2

  # From the baselines implementation.
  if value_prediction_old is not None:
    value_prediction_old = np.squeeze(value_prediction_old, axis=2)  # (B, T+1)
    value_prediction_old = value_prediction_old[:, :-1] * reward_mask  # (B, T)

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
    predicted_values: ndarray of shape (B, T+1). NOTE: Expects axis 2 was
      squeezed. These represent V(s_bt) for b < B and t < T+1
    rewards: ndarray of shape (B, T) of rewards.
    mask: ndarray of shape (B, T) of mask for rewards.
    gamma: float, discount factor.

  Returns:
    ndarray of shape (B, T) of one-step TD-residuals.
  """

  # Predicted values at time t, cutting off the last to have shape (B, T).
  predicted_values_bt = predicted_values[:, :-1]
  # Predicted values at time t+1, by cutting off the first to have shape (B, T)
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
    td_deltas: np.ndarray of shape (B, T) of one step TD-residuals.
    mask: np.ndarray of shape (B, T) of mask for the residuals. It maybe the
      case that the `td_deltas` are already masked correctly since they are
      produced by `deltas(...)`
    lambda_: float, lambda parameter for GAE estimators.
    gamma: float, lambda parameter for GAE estimators.

  Returns:
    GAE advantage estimates.
  """

  return rewards_to_go(td_deltas, mask, lambda_ * gamma)


def chosen_probabs(probab_observations, actions):
  """Picks out the probabilities of the actions along batch and time-steps.

  Args:
    probab_observations: ndarray of shape `[B, T+1, A]`, where
      probab_observations[b, t, i] contains the log-probability of action = i at
      the t^th time-step in the b^th trajectory.
    actions: ndarray of shape `[B, T]`, with each entry in [0, A) denoting which
      action was chosen in the b^th trajectory's t^th time-step.

  Returns:
    `[B, T]` ndarray with the log-probabilities of the chosen actions.
  """
  B, T = actions.shape  # pylint: disable=invalid-name
  assert (B, T + 1) == probab_observations.shape[:2]
  return probab_observations[np.arange(B)[:, None], np.arange(T), actions]


def compute_probab_ratios(p_new, p_old, actions, reward_mask):
  """Computes the probability ratios for each time-step in a trajectory.

  Args:
    p_new: ndarray of shape [B, T+1, A] of the log-probabilities that the policy
      network assigns to all the actions at each time-step in each batch using
      the old parameters.
    p_old: ndarray of shape [B, T+1, A], same as above, but using old policy
      network parameters.
    actions: ndarray of shape [B, T] where each element is from [0, A).
    reward_mask: ndarray of shape [B, T] masking over probabilities.

  Returns:
    probab_ratios: ndarray of shape [B, T], where
    probab_ratios_{b,t} = p_new_{b,t,action_{b,t}} / p_old_{b,t,action_{b,t}}
  """

  B, T = actions.shape  # pylint: disable=invalid-name
  assert (B, T + 1) == p_old.shape[:2]
  assert (B, T + 1) == p_new.shape[:2]

  logp_old = chosen_probabs(p_old, actions)
  logp_new = chosen_probabs(p_new, actions)

  assert (B, T) == logp_old.shape
  assert (B, T) == logp_new.shape

  # Since these are log-probabilities, we just subtract them.
  probab_ratios = np.exp(logp_new - logp_old) * reward_mask
  assert (B, T) == probab_ratios.shape
  return probab_ratios


def clipped_probab_ratios(probab_ratios, epsilon=0.2):
  return np.clip(probab_ratios, 1 - epsilon, 1 + epsilon)


def clipped_objective(probab_ratios, advantages, reward_mask, epsilon=0.2):
  return np.minimum(
      probab_ratios * advantages,
      clipped_probab_ratios(probab_ratios, epsilon=epsilon) *
      advantages) * reward_mask


@jit
def ppo_loss_given_predictions(log_probab_actions_new,
                               log_probab_actions_old,
                               value_predictions_old,
                               padded_actions,
                               padded_rewards,
                               reward_mask,
                               gamma=0.99,
                               lambda_=0.95,
                               epsilon=0.2):
  """PPO objective, with an eventual minus sign, given predictions."""
  B, T = padded_rewards.shape  # pylint: disable=invalid-name
  assert (B, T) == padded_actions.shape
  assert (B, T) == reward_mask.shape

  _, _, A = log_probab_actions_old.shape  # pylint: disable=invalid-name
  assert (B, T + 1, 1) == value_predictions_old.shape
  assert (B, T + 1, A) == log_probab_actions_old.shape
  assert (B, T + 1, A) == log_probab_actions_new.shape

  # (B, T)
  td_deltas = deltas(
      np.squeeze(value_predictions_old, axis=2),  # (B, T+1)
      padded_rewards,
      reward_mask,
      gamma=gamma)

  # (B, T)
  advantages = gae_advantages(
      td_deltas, reward_mask, lambda_=lambda_, gamma=gamma)

  # Normalize the advantages.
  advantage_mean = np.mean(advantages)
  advantage_std = np.std(advantages)
  advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

  # (B, T)
  ratios = compute_probab_ratios(log_probab_actions_new, log_probab_actions_old,
                                 padded_actions, reward_mask)
  assert (B, T) == ratios.shape

  # (B, T)
  objective = clipped_objective(
      ratios, advantages, reward_mask, epsilon=epsilon)
  assert (B, T) == objective.shape

  # ()
  average_objective = np.sum(objective) / np.sum(reward_mask)

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
                                    padded_rewards,
                                    reward_mask,
                                    gamma=0.99,
                                    lambda_=0.95,
                                    epsilon=0.2,
                                    c1=1.0,
                                    c2=0.01):
  """Computes the combined (clipped loss + value loss) given predictions."""
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
      padded_rewards,
      reward_mask,
      gamma=gamma,
      lambda_=lambda_,
      epsilon=epsilon)
  entropy_bonus = masked_entropy(log_probab_actions_new, reward_mask)
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
  (log_probab_actions_new, value_predictions_new), state = (
      policy_and_value_net_apply(padded_observations, new_params, state,
                                 rng=rng))

  (loss, component_losses, summaries) = combined_loss_given_predictions(
      log_probab_actions_new,
      log_probab_actions_old,
      value_predictions_new,
      value_predictions_old,
      padded_actions,
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
    log_prob_new: (B, T+1, A) log probs new
    log_prob_old: (B, T+1, A) log probs old
    mask: (B, T)

  Returns:
    Approximate KL.
  """
  diff = log_prob_old - log_prob_new
  # Cut the last time-step out.
  diff = diff[:, :-1]
  # Mask out the irrelevant part.
  diff *= mask[:, :, np.newaxis]  # make mask (B, T, 1)
  # Average on non-masked part.
  return np.sum(diff) / np.sum(mask)


def masked_entropy(log_probs, mask):
  """Computes the entropy for the given log-probs.

  Args:
    log_probs: (B, T+1, A) log probs
    mask: (B, T) mask.

  Returns:
    Entropy.
  """
  # Cut the last time-step out.
  lp = log_probs[:, :-1]
  # Mask out the irrelevant part.
  lp *= mask[:, :, np.newaxis]  # make mask (B, T, 1)
  p = np.exp(lp) * mask[:, :, np.newaxis]  # (B, T, 1)
  # Average on non-masked part and take negative.
  return -(np.sum(lp * p) / np.sum(mask))


def evaluate_policy(eval_env,
                    get_predictions,
                    temperatures,
                    max_timestep=20000,
                    n_evals=1,
                    len_history_for_policy=32,
                    state=None,
                    rng=None):
  """Evaluate the policy."""

  processed_reward_sums = collections.defaultdict(list)
  raw_reward_sums = collections.defaultdict(list)
  for eval_rng in jax_random.split(rng, num=n_evals):
    for temperature in temperatures:
      trajs, _, _, state = env_problem_utils.play_env_problem_with_policy(
          eval_env,
          get_predictions,
          num_trajectories=eval_env.batch_size,
          max_timestep=max_timestep,
          reset=True,
          temperature=temperature,
          state=state,
          rng=eval_rng,
          len_history_for_policy=len_history_for_policy)
      processed_reward_sums[temperature].extend(sum(traj[2]) for traj in trajs)
      raw_reward_sums[temperature].extend(sum(traj[3]) for traj in trajs)

  # Return the mean and standard deviation for each temperature.
  def compute_stats(reward_dict):
    return {
        temperature: {"mean": onp.mean(rewards), "std": onp.std(rewards)}
        for (temperature, rewards) in reward_dict.items()
    }
  return {
      "processed": compute_stats(processed_reward_sums),
      "raw": compute_stats(raw_reward_sums),
  }, state


def maybe_restore_opt_state(output_dir, policy_and_value_opt_state,
                            policy_and_value_state):
  """Maybe restore the optimization state from the checkpoint dir.

  Optimization state includes parameters and optimizer slots.

  Args:
    output_dir: Directory where saved model checkpoints are stored.
    policy_and_value_opt_state: Default optimization state, returned if model
      isn't found.
    policy_and_value_state: state of the policy and value network.

  Returns:
    tuple (restored (bool), opt_state, state, epoch (int),
    opt_step (int)) where epoch is the epoch from which we restored the
    optimization state, 0 is restored = False, and opt_step is the total
    optimization step (sum of all optimization steps made up to the current
    epoch).
  """
  restored = False
  epoch = 0
  total_opt_step = 0
  model_files = gfile.glob(os.path.join(output_dir, "model-??????.pkl"))
  for model_file in reversed(sorted(model_files)):
    logging.info("Trying to restore model from %s", model_file)
    try:
      with gfile.GFile(model_file, "rb") as f:
        policy_and_value_opt_state, policy_and_value_state, total_opt_step = (
            pickle.load(f))
      model_file_basename = os.path.basename(model_file)  # model-??????.pkl
      restored = True
      epoch = int(filter(str.isdigit, model_file_basename))
      break
    except EOFError as e:
      logging.error("Unable to load model from: %s with %s", model_file, e)
      # Try an older version.
      continue
  return (
      restored, policy_and_value_opt_state, policy_and_value_state, epoch,
      total_opt_step,
  )


def write_eval_reward_summaries(reward_stats_by_mode, summary_writer, epoch):
  """Writes evaluation reward statistics to summary and logs them.

  Args:
    reward_stats_by_mode: Nested dict of structure:
      {
          "raw": {
              <temperature 1>: {
                  "mean": <reward mean>,
                  "std": <reward std>,
              },
              <temperature 2>: ...
          },
          "processed": ...
      }
    summary_writer: jaxboard.SummaryWriter.
    epoch: Current epoch number.
  """
  for (reward_mode, reward_stats_by_temp) in reward_stats_by_mode.items():
    for (temperature, reward_stats) in reward_stats_by_temp.items():
      for (stat_name, stat) in reward_stats.items():
        summary_writer.scalar(
            "eval/{reward_mode}_reward_{stat_name}/"
            "temperature_{temperature}".format(reward_mode=reward_mode,
                                               stat_name=stat_name,
                                               temperature=temperature),
            stat, step=epoch)
      logging.info("Epoch [% 6d] Policy Evaluation (%s reward) "
                   "[temperature %.2f] = %10.2f (+/- %.2f)",
                   epoch, reward_mode, temperature,
                   reward_stats["mean"], reward_stats["std"])
