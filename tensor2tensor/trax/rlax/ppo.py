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
import gin
import gym
from jax import grad
from jax import jit
from jax import lax
from jax import numpy as np
from jax import random as jax_random
import numpy as onp
from tensor2tensor.envs import env_problem
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.trax import jaxboard
from tensor2tensor.trax import layers as tl
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax
from tensorflow.io import gfile

DEBUG_LOGGING = False
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.1
EPOCHS = 50  # 100
N_OPTIMIZER_STEPS = 100
PRINT_EVERY_OPTIMIZER_STEP = 20
BATCH_TRAJECTORIES = 32


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
  params = net.initialize(batch_observations_shape, observations_dtype, rng_key)
  return params, net


def optimizer_fn(net_params, step_size=1e-3):
  opt = trax_opt.Adam(step_size=step_size, b1=0.9, b2=0.999, eps=1e-08)
  opt_init = lambda x: (x, opt.tree_init(x))
  opt_update = lambda i, g, s: opt.tree_update(i, g, s[0], s[1])
  get_params = lambda x: x[0]
  opt_state = opt_init(net_params)
  return opt_state, opt_update, get_params


# Should this be collect 'n' trajectories, or
# Run the env for 'n' steps and take completed trajectories, or
# Any other option?
def collect_trajectories(env,
                         policy_fn,
                         n_trajectories=1,
                         policy=env_problem_utils.GUMBEL_SAMPLING,
                         max_timestep=None,
                         epsilon=0.1,
                         reset=True,
                         len_history_for_policy=32,
                         rng=None):
  """Collect trajectories with the given policy net and behaviour.

  Args:
    env: A gym env interface, for now this is not-batched.
    policy_fn: observations(B,T+1) -> log-probabs(B,T+1, A) callable.
    n_trajectories: int, number of trajectories.
    policy: string, "greedy", "epsilon-greedy", or "categorical-sampling" i.e.
      how to use the policy_fn to return an action.
    max_timestep: int or None, the index of the maximum time-step at which we
      return the trajectory, None for ending a trajectory only when env returns
      done.
    epsilon: float, the epsilon for `epsilon-greedy` policy.
    reset: bool, true if we want to reset the envs. The envs are also reset if
      max_max_timestep is None or < 0
    len_history_for_policy: int, the maximum history to keep for applying the
      policy on.
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
  trajs, n_done, timing_info = env_problem_utils.play_env_problem_with_policy(
      env,
      policy_fn,
      num_trajectories=n_trajectories,
      max_timestep=max_timestep,
      policy_sampling=policy,
      eps=epsilon,
      reset=reset,
      len_history_for_policy=len_history_for_policy,
      rng=rng)
  # Skip returning raw_rewards here, since they aren't used.

  # t is the return value of Trajectory.as_numpy, so:
  # (observation, action, processed_reward, raw_reward, infos)
  return [(t[0], t[1], t[2], t[4]) for t in trajs], n_done, timing_info


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
    The average L2 value loss, averaged over instances where reward_mask is 1.
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
  return np.sum(loss) / np.sum(reward_mask)


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
  advantages = (advantages - np.mean(advantages)) / np.std(advantages)

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
  return -average_objective


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
  loss_value = value_loss_given_predictions(
      value_prediction_new,
      padded_rewards,
      reward_mask,
      gamma=gamma,
      value_prediction_old=value_prediction_old,
      epsilon=epsilon)
  loss_ppo = ppo_loss_given_predictions(
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
  return (loss_ppo + (c1 * loss_value) - (c2 * entropy_bonus), loss_ppo,
          loss_value, entropy_bonus)


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
                  rng=None):
  """Computes the combined (clipped loss + value loss) given observations."""
  log_probab_actions_new, value_predictions_new = policy_and_value_net_apply(
      padded_observations, new_params, rng=rng)

  # (combined_loss, ppo_loss, value_loss, entropy_bonus)
  return combined_loss_given_predictions(
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
      c2=c2)


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
                              rng=None):
  """Policy and Value optimizer step."""

  # Combined loss function given the new params.
  def policy_and_value_loss(params):
    """Returns the combined loss given just parameters."""
    (loss, _, _, _) = combined_loss(
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
        rng=rng)
    return loss

  new_params = get_params(opt_state)
  g = grad(policy_and_value_loss)(new_params)
  # TODO(afrozm): Maybe clip gradients?
  return opt_update(i, g, opt_state)


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
                    rng=None):
  """Evaluate the policy."""

  processed_reward_sums = collections.defaultdict(list)
  raw_reward_sums = collections.defaultdict(list)
  for eval_rng in jax_random.split(rng, num=n_evals):
    for temperature in temperatures:
      trajs, _, _ = env_problem_utils.play_env_problem_with_policy(
          eval_env,
          get_predictions,
          num_trajectories=eval_env.batch_size,
          max_timestep=max_timestep,
          reset=True,
          policy_sampling=env_problem_utils.GUMBEL_SAMPLING,
          temperature=temperature,
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
  }


def maybe_restore_params(output_dir, policy_and_value_net_params):
  """Maybe restore the params from the checkpoint dir.

  Args:
    output_dir: Directory where saved model checkpoints are stored.
    policy_and_value_net_params: Default params, returned if model is'nt found.

  Returns:
    triple (restore (bool), params, iter(int)) where iter is the epoch from
    which we restored the params, 0 is restore = False.
  """
  model_files = gfile.glob(os.path.join(output_dir, "model-??????.pkl"))
  for model_file in reversed(sorted(model_files)):
    logging.info("Trying to restore model from %s", model_file)
    try:
      with gfile.GFile(model_file, "rb") as f:
        loaded_policy_and_value_net_params = pickle.load(f)
        policy_and_value_net_params = loaded_policy_and_value_net_params
      model_file_basename = os.path.basename(model_file)  # model-??????.pkl
      i = int(filter(str.isdigit, model_file_basename))
      return True, policy_and_value_net_params, i
    except EOFError as e:
      logging.error("Unable to load model from: %s with %s", model_file, e)
      # Try an older version.
      continue
  return False, policy_and_value_net_params, 0


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


@gin.configurable(blacklist=["output_dir"])
def training_loop(
    env,
    eval_env,
    env_name,
    policy_and_value_net_fn,
    policy_and_value_optimizer_fn,
    output_dir,
    epochs=EPOCHS,
    n_optimizer_steps=N_OPTIMIZER_STEPS,
    print_every_optimizer_steps=PRINT_EVERY_OPTIMIZER_STEP,
    target_kl=0.01,
    boundary=20,
    max_timestep=None,
    max_timestep_eval=20000,
    random_seed=None,
    gamma=GAMMA,
    lambda_=LAMBDA,
    epsilon=EPSILON,
    c1=1.0,
    c2=0.01,
    eval_every_n=1000,
    done_frac_for_policy_save=0.5,
    enable_early_stopping=True,
    n_evals=1,
    len_history_for_policy=4,
    eval_temperatures=(1.0, 0.5),
):
  """Runs the training loop for PPO, with fixed policy and value nets.

  Args:
    env: gym.Env to use for training.
    eval_env: gym.Env to use for evaluation.
    env_name: Name of the environment.
    policy_and_value_net_fn: Function defining the policy and value network.
    policy_and_value_optimizer_fn: Function defining the optimizer.
    output_dir: Output dir.
    epochs: Number of epochs to run for.
    n_optimizer_steps: Number of optimizer steps.
    print_every_optimizer_steps: How often to log during the policy optimization
      process.
    target_kl: Policy iteration early stopping.
    boundary: We pad trajectories at integer multiples of this number.
    max_timestep: If set to an integer, maximum number of time-steps in
      a trajectory. Used in the collect procedure.
    max_timestep_eval: If set to an integer, maximum number of time-steps in an
      evaluation trajectory. Used in the collect procedure.
    random_seed: Random seed.
    gamma: Reward discount factor.
    lambda_: N-step TD-error discount factor in GAE.
    epsilon: Random action probability in epsilon-greedy sampling.
    c1: Value loss coefficient.
    c2: Entropy loss coefficient.
    eval_every_n: How frequently to eval the policy.
    done_frac_for_policy_save: Fraction of the trajectories that should be done
      to checkpoint the policy.
    enable_early_stopping: Whether to enable early stopping.
    n_evals: Number of times to evaluate.
    len_history_for_policy: How much of history to give to the policy.
    eval_temperatures: Sequence of temperatures to try for categorical sampling
      during evaluation.
  """
  gfile.makedirs(output_dir)

  # Create summary writers and history.
  train_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "train"))
  timing_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "timing"))
  eval_sw = jaxboard.SummaryWriter(os.path.join(output_dir, "eval"))

  train_sw.text("env_name", env_name)
  timing_sw.text("env_name", env_name)
  eval_sw.text("env_name", env_name)

  jax_rng_key = trax.get_random_number_generator_and_set_seed(random_seed)

  # Batch Observations Shape = [1, 1] + OBS, because we will eventually call
  # policy and value networks on shape [B, T] +_OBS
  batch_observations_shape = (1, 1) + env.observation_space.shape
  observations_dtype = env.observation_space.dtype

  assert isinstance(env.action_space, gym.spaces.Discrete)
  n_actions = env.action_space.n

  jax_rng_key, key1 = jax_random.split(jax_rng_key, num=2)

  # Initialize the policy and value network.
  policy_and_value_net_params, policy_and_value_net_apply = (
      policy_and_value_net_fn(key1, batch_observations_shape,
                              observations_dtype, n_actions))

  # Maybe restore the policy params. If there is nothing to restore, then
  # iteration = 0 and policy_and_value_net_params are returned as is.
  restore, policy_and_value_net_params, iteration = (
      maybe_restore_params(output_dir, policy_and_value_net_params))

  if restore:
    logging.info("Restored parameters from iteration [%d]", iteration)
    # We should start from the next iteration.
    iteration += 1

  policy_and_value_net_apply = jit(policy_and_value_net_apply)

  # Initialize the optimizers.
  policy_and_value_optimizer = (
      policy_and_value_optimizer_fn(policy_and_value_net_params))
  (policy_and_value_opt_state, policy_and_value_opt_update,
   policy_and_value_get_params) = policy_and_value_optimizer

  n_trajectories_done = 0
  last_saved_at = 0

  logging.info("Starting the PPO training loop.")
  for i in range(iteration, epochs):
    epoch_start_time = time.time()

    # Params we'll use to collect the trajectories.
    policy_and_value_net_params = policy_and_value_get_params(
        policy_and_value_opt_state)

    # A function to get the policy and value predictions.
    def get_predictions(observations, rng=None):
      """Returns log-probs, value predictions and key back."""
      key, key1 = jax_random.split(rng, num=2)

      log_probs, value_preds = policy_and_value_net_apply(
          observations, policy_and_value_net_params, rng=key1)

      return log_probs, value_preds, key

    # Evaluate the policy.
    policy_eval_start_time = time.time()
    if ((i + 1) % eval_every_n == 0) or (i == epochs - 1):
      jax_rng_key, key = jax_random.split(jax_rng_key, num=2)

      logging.vlog(1, "Epoch [% 6d] evaluating policy.", i)

      reward_stats = evaluate_policy(
          eval_env,
          get_predictions,
          temperatures=eval_temperatures,
          max_timestep=max_timestep_eval,
          n_evals=n_evals,
          len_history_for_policy=len_history_for_policy,
          rng=key)
      write_eval_reward_summaries(reward_stats, eval_sw, epoch=i)
    policy_eval_time = get_time(policy_eval_start_time)

    trajectory_collection_start_time = time.time()
    logging.vlog(1, "Epoch [% 6d] collecting trajectories.", i)
    jax_rng_key, key = jax_random.split(jax_rng_key)
    trajs, n_done, timing_info = collect_trajectories(
        env,
        policy_fn=get_predictions,
        n_trajectories=env.batch_size,
        max_timestep=max_timestep,
        rng=key,
        len_history_for_policy=len_history_for_policy,
        reset=(i == 0) or restore,
        epsilon=(10.0 / (i + 10.0)))  # this is a different epsilon.
    trajectory_collection_time = get_time(trajectory_collection_start_time)

    logging.vlog(1, "Collecting trajectories took %0.2f msec.",
                 trajectory_collection_time)

    avg_reward = float(sum(np.sum(traj[2]) for traj in trajs)) / len(trajs)
    max_reward = max(np.sum(traj[2]) for traj in trajs)
    min_reward = min(np.sum(traj[2]) for traj in trajs)

    train_sw.scalar("train/reward_mean_truncated", avg_reward, step=i)

    logging.vlog(1, "Rewards avg=[%0.2f], max=[%0.2f], min=[%0.2f], all=%s",
                 avg_reward, max_reward, min_reward,
                 [float(np.sum(traj[2])) for traj in trajs])

    logging.vlog(1,
                 "Trajectory Length average=[%0.2f], max=[%0.2f], min=[%0.2f]",
                 float(sum(len(traj[0]) for traj in trajs)) / len(trajs),
                 max(len(traj[0]) for traj in trajs),
                 min(len(traj[0]) for traj in trajs))
    logging.vlog(2, "Trajectory Lengths: %s", [len(traj[0]) for traj in trajs])

    padding_start_time = time.time()
    (_, reward_mask, padded_observations, padded_actions,
     padded_rewards, padded_infos) = pad_trajectories(
         trajs, boundary=boundary)
    padding_time = get_time(padding_start_time)

    logging.vlog(1, "Padding trajectories took %0.2f msec.",
                 get_time(padding_start_time))
    logging.vlog(1, "Padded Observations' shape [%s]",
                 str(padded_observations.shape))
    logging.vlog(1, "Padded Actions' shape [%s]", str(padded_actions.shape))
    logging.vlog(1, "Padded Rewards' shape [%s]", str(padded_rewards.shape))

    # Some assertions.
    B, T = padded_actions.shape  # pylint: disable=invalid-name
    assert (B, T) == padded_rewards.shape
    assert (B, T) == reward_mask.shape
    assert (B, T + 1) == padded_observations.shape[:2]
    assert (B, T + 1) + env.observation_space.shape == padded_observations.shape

    log_prob_recompute_start_time = time.time()
    assert ("log_prob_actions" in padded_infos and
            "value_predictions" in padded_infos)
    # These are the actual log-probabs and value predictions seen while picking
    # the actions.
    actual_log_probabs_traj = padded_infos["log_prob_actions"]
    actual_value_predictions_traj = padded_infos["value_predictions"]

    assert (B, T) == actual_log_probabs_traj.shape[:2]
    A = actual_log_probabs_traj.shape[2]  # pylint: disable=invalid-name
    assert (B, T, 1) == actual_value_predictions_traj.shape

    # TODO(afrozm): log-probabs doesn't need to be (B, T+1, A) it can do with
    # (B, T, A), so make that change throughout.

    # NOTE: We don't have the log-probabs and value-predictions for the last
    # observation, so we re-calculate for everything, but use the original ones
    # for all but the last time-step.
    jax_rng_key, key = jax_random.split(jax_rng_key)
    log_probabs_traj, value_predictions_traj, _ = get_predictions(
        padded_observations, rng=key)

    assert (B, T + 1, A) == log_probabs_traj.shape
    assert (B, T + 1, 1) == value_predictions_traj.shape

    # Concatenate the last time-step's log-probabs and value predictions to the
    # actual log-probabs and value predictions and use those going forward.
    log_probabs_traj = np.concatenate(
        (actual_log_probabs_traj, log_probabs_traj[:, -1:, :]), axis=1)
    value_predictions_traj = np.concatenate(
        (actual_value_predictions_traj, value_predictions_traj[:, -1:, :]),
        axis=1)

    log_prob_recompute_time = get_time(log_prob_recompute_start_time)

    # Linear annealing from 0.1 to 0.0
    # epsilon_schedule = epsilon if epochs == 1 else epsilon * (1.0 -
    #                                                           (i /
    #                                                            (epochs - 1)))

    # Constant epsilon.
    epsilon_schedule = epsilon

    # Compute value and ppo losses.
    jax_rng_key, key1 = jax_random.split(jax_rng_key, num=2)
    logging.vlog(2, "Starting to compute P&V loss.")
    loss_compute_start_time = time.time()
    cur_combined_loss, cur_ppo_loss, cur_value_loss, entropy_bonus = (
        combined_loss(
            policy_and_value_net_params,
            log_probabs_traj,
            value_predictions_traj,
            policy_and_value_net_apply,
            padded_observations,
            padded_actions,
            padded_rewards,
            reward_mask,
            gamma=gamma,
            lambda_=lambda_,
            epsilon=epsilon_schedule,
            c1=c1,
            c2=c2,
            rng=key1))
    loss_compute_time = get_time(loss_compute_start_time)
    logging.vlog(
        1,
        "Calculating P&V loss [%10.2f(%10.2f, %10.2f, %10.2f)] took %0.2f msec.",
        cur_combined_loss, cur_value_loss, cur_ppo_loss, entropy_bonus,
        get_time(loss_compute_start_time))

    jax_rng_key, key1 = jax_random.split(jax_rng_key, num=2)
    logging.vlog(1, "Policy and Value Optimization")
    optimization_start_time = time.time()
    keys = jax_random.split(key1, num=n_optimizer_steps)
    for j in range(n_optimizer_steps):
      k1, k2, k3 = jax_random.split(keys[j], num=3)
      t = time.time()
      # Update the optimizer state.
      policy_and_value_opt_state = policy_and_value_opt_step(
          j,
          policy_and_value_opt_state,
          policy_and_value_opt_update,
          policy_and_value_get_params,
          policy_and_value_net_apply,
          log_probabs_traj,
          value_predictions_traj,
          padded_observations,
          padded_actions,
          padded_rewards,
          reward_mask,
          c1=c1,
          c2=c2,
          gamma=gamma,
          lambda_=lambda_,
          epsilon=epsilon_schedule,
          rng=k1)

      # Compute the approx KL for early stopping.
      new_policy_and_value_net_params = policy_and_value_get_params(
          policy_and_value_opt_state)

      log_probab_actions_new, _ = policy_and_value_net_apply(
          padded_observations, new_policy_and_value_net_params, rng=k2)

      approx_kl = approximate_kl(log_probab_actions_new, log_probabs_traj,
                                 reward_mask)

      early_stopping = enable_early_stopping and approx_kl > 1.5 * target_kl
      if early_stopping:
        logging.vlog(
            1, "Early stopping policy and value optimization at iter: %d, "
            "with approx_kl: %0.2f", j, approx_kl)
        # We don't return right-away, we want the below to execute on the last
        # iteration.

      t2 = time.time()
      if (((j + 1) % print_every_optimizer_steps == 0) or
          (j == n_optimizer_steps - 1) or early_stopping):
        # Compute and log the loss.
        (loss_combined, loss_ppo, loss_value, entropy_bonus) = (
            combined_loss(
                new_policy_and_value_net_params,
                log_probabs_traj,
                value_predictions_traj,
                policy_and_value_net_apply,
                padded_observations,
                padded_actions,
                padded_rewards,
                reward_mask,
                gamma=gamma,
                lambda_=lambda_,
                epsilon=epsilon_schedule,
                c1=c1,
                c2=c2,
                rng=k3))
        logging.vlog(1, "One Policy and Value grad desc took: %0.2f msec",
                     get_time(t, t2))
        logging.vlog(
            1, "Combined Loss(value, ppo, entropy_bonus) [%10.2f] ->"
            " [%10.2f(%10.2f,%10.2f,%10.2f)]", cur_combined_loss, loss_combined,
            loss_value, loss_ppo, entropy_bonus)

      if early_stopping:
        break

    optimization_time = get_time(optimization_start_time)

    logging.vlog(
        1, "Total Combined Loss reduction [%0.2f]%%",
        (100 * (cur_combined_loss - loss_combined) / np.abs(cur_combined_loss)))

    # Save parameters every time we see the end of at least a fraction of batch
    # number of trajectories that are done (not completed -- completed includes
    # truncated and done).
    # Also don't save too frequently, enforce a minimum gap.
    # Or if this is the last iteration.
    policy_save_start_time = time.time()
    n_trajectories_done += n_done
    # TODO(afrozm): Refactor to trax.save_state.
    if (((n_trajectories_done >= done_frac_for_policy_save * env.batch_size) and
         (i - last_saved_at > eval_every_n) and
         (((i + 1) % eval_every_n == 0))) or (i == epochs - 1)):
      logging.vlog(1, "Epoch [% 6d] saving model.", i)
      old_model_files = gfile.glob(os.path.join(output_dir, "model-??????.pkl"))
      params_file = os.path.join(output_dir, "model-%06d.pkl" % i)
      with gfile.GFile(params_file, "wb") as f:
        pickle.dump(policy_and_value_net_params, f)
      # Remove the old model files.
      for path in old_model_files:
        gfile.remove(path)
      # Reset this number.
      n_trajectories_done = 0
      last_saved_at = i
    policy_save_time = get_time(policy_save_start_time)

    epoch_time = get_time(epoch_start_time)

    logging.info(
        "Epoch [% 6d], Reward[min, max, avg] [%5.2f,%5.2f,%5.2f], Combined"
        " Loss(value, ppo, entropy) [%2.5f(%2.5f,%2.5f,%2.5f)]", i, min_reward,
        max_reward, avg_reward, loss_combined, loss_value, loss_ppo,
        entropy_bonus)

    timing_dict = {
        "epoch": epoch_time,
        "policy_eval": policy_eval_time,
        "trajectory_collection": trajectory_collection_time,
        "padding": padding_time,
        "log_prob_recompute": log_prob_recompute_time,
        "loss_compute": loss_compute_time,
        "optimization": optimization_time,
        "policy_save": policy_save_time,
    }

    timing_dict.update(timing_info)

    for k, v in timing_dict.items():
      timing_sw.scalar("timing/%s" % k, v, step=i)

    max_key_len = max(len(k) for k in timing_dict)
    timing_info_list = [
        "%s : % 10.2f" % (k.rjust(max_key_len + 1), v)
        for k, v in sorted(timing_dict.items())
    ]
    logging.info("Epoch [% 6d], Timings: \n%s", i, "\n".join(timing_info_list))

    # Reset restore.
    restore = False

    # Flush summary writers once in a while.
    if (i + 1) % 1000 == 0 or i == epochs - 1:
      train_sw.flush()
      timing_sw.flush()
      eval_sw.flush()
