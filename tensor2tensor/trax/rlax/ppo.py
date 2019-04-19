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

Policy Function :: [B, T] + OBS -> [B, T, A]
Value  Function :: [B, T] + OBS -> [B, T, 1]

i.e. the policy net should take a batch of *trajectories* and at each time-step
in each batch deliver a probability distribution over actions.

NOTE: It doesn't return logits, rather the expectation is that it return a
normalized distribution instead.

NOTE: The policy and value functions need to take care to not take into account
future time-steps while deciding the actions (or value) for the current
time-step.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import time

from absl import logging
import gym
from jax import grad
from jax import jit
from jax import lax
from jax import numpy as np
from jax import random as jax_random
import numpy as onp
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax
from tensor2tensor.trax.stax import stax_base as stax

DEBUG_LOGGING = False
GAMMA = 0.99
LAMBDA = 0.95
EPOCHS = 50  # 100
NUM_OPTIMIZER_STEPS = 100
PRINT_EVERY_OPTIMIZER_STEP = 20
BATCH_TRAJECTORIES = 32
POLICY = "categorical-sampling"


def policy_net(rng_key,
               batch_observations_shape,
               num_actions,
               bottom_layers=None):
  """A policy net function."""
  # Use the bottom_layers as the bottom part of the network and just add the
  # required layers on top of it.
  if bottom_layers is None:
    bottom_layers = []
  bottom_layers.extend([stax.Dense(num_actions), stax.Softmax])

  net_init, net_apply = stax.serial(*bottom_layers)

  _, net_params = net_init(rng_key, batch_observations_shape)
  return net_params, net_apply


def value_net(rng_key,
              batch_observations_shape,
              num_actions,
              bottom_layers=None):
  """A value net function."""
  del num_actions

  if bottom_layers is None:
    bottom_layers = []
  bottom_layers.extend([
      stax.Dense(1),
  ])

  net_init, net_apply = stax.serial(*bottom_layers)

  _, net_params = net_init(rng_key, batch_observations_shape)
  return net_params, net_apply


def optimizer_fun(net_params):
  opt_init, opt_update = trax_opt.adam(
      step_size=1e-3, b1=0.9, b2=0.999, eps=1e-08)
  opt_state = opt_init(net_params)
  return opt_state, opt_update


# Should this be collect 'n' trajectories, or
# Run the env for 'n' steps and take completed trajectories, or
# Any other option?
# TODO(afrozm): Replace this with EnvProblem?
def collect_trajectories(env,
                         policy_net_apply,
                         policy_net_params,
                         num_trajectories=1,
                         policy="greedy",
                         epsilon=0.1):
  """Collect trajectories with the given policy net and behaviour."""
  trajectories = []

  for _ in range(num_trajectories):
    rewards = []
    actions = []
    done = False

    observation = env.reset()

    # This is currently shaped (1, 1) + OBS, but new observations will keep
    # getting added to it, making it eventually (1, T+1) + OBS
    observation_history = observation[np.newaxis, np.newaxis, :]

    while not done:
      # Run the policy, to pick an action, shape is (1, t, A) because
      # observation_history is shaped (1, t) + OBS
      predictions = policy_net_apply(policy_net_params, observation_history)

      # We need the predictions for the last time-step, so squeeze the batch
      # dimension and take the last time-step.
      predictions = np.squeeze(predictions, axis=0)[-1]

      # Policy can be run in one of the following ways:
      #  - Greedy
      #  - Epsilon-Greedy
      #  - Categorical-Sampling
      action = None
      if policy == "greedy":
        action = np.argmax(predictions)
      elif policy == "epsilon-greedy":
        # A schedule for epsilon is 1/k where k is the episode number sampled.
        if onp.random.random() < epsilon:
          # Choose an action at random.
          action = onp.random.randint(0, high=len(predictions))
        else:
          # Return the best action.
          action = np.argmax(predictions)
      elif policy == "categorical-sampling":
        action = onp.argwhere(onp.random.multinomial(1, predictions) == 1)
      else:
        raise ValueError("Unknown policy: %s" % policy)

      # NOTE: Assumption, single batch.
      action = int(action)

      observation, reward, done, _ = env.step(action)

      # observation is of shape OBS, so add extra dims and concatenate on the
      # time dimension.
      observation_history = np.concatenate(
          [observation_history, observation[np.newaxis, np.newaxis, :]], axis=1)

      rewards.append(reward)
      actions.append(action)

    # This means we are done
    assert done
    # observation_history is (1, T+1) + OBS, lets squeeze out the batch dim.
    observation_history = np.squeeze(observation_history, axis=0)
    trajectories.append(
        (observation_history, np.stack(actions), np.stack(rewards)))

  return trajectories


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
  elif dtype == np.float32:
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
  t_max = max(r.shape[0] for (_, _, r) in trajectories)

  # t_max is rounded to the next multiple of `boundary`
  boundary = int(boundary)
  bucket_length = boundary * int(np.ceil(float(t_max) / boundary))

  # So all obs will be padded to t_max + 1 and actions and rewards to t_max.
  padded_observations = []
  padded_actions = []
  padded_rewards = []
  padded_lengths = []
  reward_masks = []
  for (o, a, r) in trajectories:
    # Determine the amount to pad, this holds true for obs, actions and rewards.
    num_to_pad = bucket_length + 1 - o.shape[0]
    padded_lengths.append(num_to_pad)
    if num_to_pad == 0:
      padded_observations.append(o)
      padded_actions.append(a)
      padded_rewards.append(r)
      reward_masks.append(onp.ones_like(r, dtype=np.int32))
      continue

    # First pad observations.
    padding_config = [(0, num_to_pad, 0)]
    for _ in range(o.ndim - 1):
      padding_config.append((0, 0, 0))
    padding_config = tuple(padding_config)

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

  return padded_lengths, np.stack(reward_masks), np.stack(
      padded_observations), np.stack(padded_actions), np.stack(padded_rewards)


# TODO(afrozm): JAX-ify this, this is too slow for pong.
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


@functools.partial(jit, static_argnums=(0,))
def value_loss(value_net_apply,
               value_net_params,
               observations,
               rewards,
               reward_mask,
               gamma=0.99):
  """Computes the value loss.

  Args:
    value_net_apply: value net apply function with signature (params, ndarray of
      shape (B, T+1) + OBS) -> ndarray(B, T+1, 1)
    value_net_params: params of value_net_apply.
    observations: np.ndarray of shape (B, T+1) + OBS
    rewards: np.ndarray of shape (B, T) of rewards.
    reward_mask: np.ndarray of shape (B, T), the mask over rewards.
    gamma: float, discount factor.

  Returns:
    The average L2 value loss, averaged over instances where reward_mask is 1.
  """

  B, T = rewards.shape  # pylint: disable=invalid-name
  assert (B, T) == reward_mask.shape
  assert (B, T + 1) == observations.shape[:2]

  r2g = rewards_to_go(rewards, reward_mask, gamma=gamma)  # (B, T)
  # NOTE: observations is (B, T+1) + OBS, value_prediction is (B, T+1, 1)
  value_prediction = value_net_apply(value_net_params, observations)
  assert (B, T + 1, 1) == value_prediction.shape
  value_prediction = np.squeeze(value_prediction, axis=2)  # (B, T+1)
  value_prediction = value_prediction[:, :-1] * reward_mask  # (B, T)
  loss = (value_prediction - r2g)**2
  # Take an average on only the points where mask != 0.
  return np.sum(loss) / np.sum(reward_mask)


# TODO(afrozm): JAX-ify this, this is too slow for pong.
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

  # `d`s are basically one-step TD residuals.
  d = []
  _, T = rewards.shape  # pylint: disable=invalid-name
  for t in range(T):
    d.append(rewards[:, t] + (gamma * predicted_values[:, t + 1]) -
             predicted_values[:, t])

  return np.array(d).T * mask


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
    probab_observations: ndarray of shape `[B, T, A]`, where
      probab_observations[b, t, i] contains the probability of action = i at the
      t^th time-step in the b^th trajectory.
    actions: ndarray of shape `[B, T]`, with each entry in [0, A) denoting which
      action was chosen in the b^th trajectory's t^th time-step.

  Returns:
    `[B, T]` ndarray with the probabilities of the chosen actions.
  """
  b, t = actions.shape
  return probab_observations[np.arange(b)[:, None], np.arange(t), actions]


def compute_probab_ratios(p_old, p_new, actions, reward_mask):
  """Computes the probability ratios for each time-step in a trajectory.

  Args:
    p_old: ndarray of shape [B, T, A] of the probabilities that the policy
      network assigns to all the actions at each time-step in each batch using
      the old parameters.
    p_new: ndarray of shape [B, T, A], same as above, but using new policy
      network parameters.
    actions: ndarray of shape [B, T] where each element is from [0, A).
    reward_mask: ndarray of shape [B, T] masking over probabilities.

  Returns:
    probab_ratios: ndarray of shape [B, T], where
    probab_ratios_{b,t} = p_new_{b,t,action_{b,t}} / p_old_{b,t,action_{b,t}}
  """
  bp_old = chosen_probabs(p_old, actions)
  bp_new = chosen_probabs(p_new, actions)

  # Add a small number to bp_old, where reward_mask is 0, this is just to help
  # never to divide by 0.
  bp_old = bp_old + (0.1 * np.abs(reward_mask - 1))
  probab_ratios = (bp_new * reward_mask) / bp_old
  return probab_ratios


def clipped_probab_ratios(probab_ratios, reward_mask, epsilon=0.2):
  return reward_mask * np.clip(probab_ratios, 1 - epsilon, 1 + epsilon)


def clipped_objective(probab_ratios, advantages, reward_mask, epsilon=0.2):
  c1 = probab_ratios * reward_mask
  c2 = clipped_probab_ratios(probab_ratios, reward_mask, epsilon=epsilon)
  return np.minimum(c1, c2) * advantages


@functools.partial(jit, static_argnums=(0, 3))
def ppo_loss(policy_net_apply,
             new_policy_params,
             old_policy_params,
             value_net_apply,
             value_net_params,
             padded_observations,
             padded_actions,
             padded_rewards,
             reward_mask,
             gamma=0.99,
             lambda_=0.95,
             epsilon=0.2):
  """PPO objective, with an eventual minus sign."""
  # (B, T+1, 1)
  predicted_values = value_net_apply(value_net_params, padded_observations)

  # (B, T)
  td_deltas = deltas(
      np.squeeze(predicted_values, axis=2),  # (B, T)
      padded_rewards,
      reward_mask,
      gamma=gamma)

  # (B, T)
  advantages = gae_advantages(
      td_deltas, reward_mask, lambda_=lambda_, gamma=gamma)

  # probab_actions_{old,new} are both (B, T, A)
  probab_actions_old = policy_net_apply(old_policy_params, padded_observations)
  probab_actions_new = policy_net_apply(new_policy_params, padded_observations)

  # (B, T)
  ratios = compute_probab_ratios(probab_actions_old, probab_actions_new,
                                 padded_actions, reward_mask)

  # (B, T)
  objective = clipped_objective(
      ratios, advantages, reward_mask, epsilon=epsilon)

  # ()
  average_objective = np.sum(objective) / np.sum(reward_mask)

  # Loss is negative objective.
  return -average_objective


@functools.partial(jit, static_argnums=(2, 3, 5))
def ppo_opt_step(i,
                 opt_state,
                 ppo_opt_update,
                 policy_net_apply,
                 old_policy_params,
                 value_net_apply,
                 value_net_params,
                 padded_observations,
                 padded_actions,
                 padded_rewards,
                 reward_mask,
                 gamma=0.99,
                 lambda_=0.95,
                 epsilon=0.1):
  """PPO optimizer step."""
  new_policy_params = trax_opt.get_params(opt_state)
  g = grad(
      ppo_loss, argnums=1)(
          policy_net_apply,
          new_policy_params,
          old_policy_params,
          value_net_apply,
          value_net_params,
          padded_observations,
          padded_actions,
          padded_rewards,
          reward_mask,
          gamma=gamma,
          lambda_=lambda_,
          epsilon=epsilon)
  return ppo_opt_update(i, g, opt_state)


@functools.partial(jit, static_argnums=(2, 3))
def value_opt_step(i,
                   opt_state,
                   opt_update,
                   value_net_apply,
                   padded_observations,
                   padded_rewards,
                   reward_mask,
                   gamma=0.99):
  """Value optimizer step."""
  value_params = trax_opt.get_params(opt_state)
  # Note this partial application here and argnums above in ppo_opt_step.
  g = grad(functools.partial(value_loss, value_net_apply))(
      value_params,
      padded_observations,
      padded_rewards,
      reward_mask,
      gamma=gamma)
  return opt_update(i, g, opt_state)


def get_time(t1, t2=None):
  if t2 is None:
    t2 = time.time()
  return round((t2 - t1) * 1000, 2)


def training_loop(
    env=None,
    env_name="CartPole-v0",
    epochs=EPOCHS,
    policy_net_fun=None,
    value_net_fun=None,
    policy_and_value_net_fun=None,  # TODO(afrozm): Implement.
    policy_optimizer_fun=optimizer_fun,
    value_optimizer_fun=optimizer_fun,
    batch_size=BATCH_TRAJECTORIES,
    num_optimizer_steps=NUM_OPTIMIZER_STEPS,
    print_every_optimizer_steps=PRINT_EVERY_OPTIMIZER_STEP,
    boundary=20,
    random_seed=None):
  """Runs the training loop for PPO, with fixed policy and value nets."""
  jax_rng_key = trax.get_random_number_generator_and_set_seed(random_seed)

  value_losses = []
  ppo_objective = []
  average_rewards = []

  env = env if env is not None else gym.make(env_name)

  # Batch Observations Shape = [-1, -1] + OBS, because we will eventually call
  # policy and value networks on shape [B, T] +_OBS
  batch_observations_shape = (-1, -1) + env.observation_space.shape

  assert isinstance(env.action_space, gym.spaces.Discrete)
  num_actions = env.action_space.n

  # TODO(afrozm): Have a single net for both policy and action.
  assert policy_and_value_net_fun is None

  # Initialize the policy and value functions.
  assert policy_net_fun and value_net_fun
  jax_rng_key, key1, key2 = jax_random.split(jax_rng_key, num=3)

  policy_net_params, policy_net_apply = policy_net_fun(
      key1, batch_observations_shape, num_actions)
  value_net_params, value_net_apply = value_net_fun(key2,
                                                    batch_observations_shape,
                                                    num_actions)

  # Initialize the optimizers.
  assert policy_optimizer_fun and value_optimizer_fun

  ppo_opt_state, ppo_opt_update = policy_optimizer_fun(policy_net_params)
  value_opt_state, value_opt_update = value_optimizer_fun(value_net_params)

  for i in range(epochs):
    t = time.time()
    t0 = t
    logging.vlog(1, "Epoch [% 6d] collecting trajectories.", i)
    trajs = collect_trajectories(
        env,
        policy_net_apply,
        policy_net_params,
        num_trajectories=batch_size,
        policy=POLICY,
        epsilon=(10.0 / (i + 10.0)))  # this is a different epsilon.

    avg_reward = float(sum(np.sum(traj[2]) for traj in trajs)) / len(trajs)
    max_reward = max(np.sum(traj[2]) for traj in trajs)
    min_reward = min(np.sum(traj[2]) for traj in trajs)
    average_rewards.append(avg_reward)

    logging.vlog(1, "Rewards average=[%0.2f], max=[%0.2f], min=[%0.2f]",
                 avg_reward, max_reward, min_reward)
    logging.vlog(1, "Collecting trajectories took %0.2f msec.", get_time(t))
    logging.vlog(1,
                 "Trajectory Length average=[%0.2f], max=[%0.2f], min=[%0.2f]",
                 float(sum(len(traj[0]) for traj in trajs)) / len(trajs),
                 max(len(traj[0]) for traj in trajs),
                 min(len(traj[0]) for traj in trajs))

    t = time.time()
    (_, reward_mask, padded_observations, padded_actions,
     padded_rewards) = pad_trajectories(trajs, boundary=boundary)

    logging.vlog(1, "Padding trajectories took %0.2f msec.", get_time(t))
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

    # Linear annealing from 0.1 to 0.0
    epsilon = 0.1 if epochs == 1 else 0.1 * (1.0 - (i / (epochs - 1)))

    t = time.time()
    cur_value_loss = value_loss(
        value_net_apply,
        value_net_params,
        padded_observations,
        padded_rewards,
        reward_mask,
        gamma=GAMMA)

    logging.vlog(1, "Calculating value loss took %0.2f msec.", get_time(t))
    value_losses.append(cur_value_loss)

    t = time.time()
    cur_ppo_loss = ppo_loss(
        policy_net_apply,
        policy_net_params,
        policy_net_params,
        value_net_apply,
        value_net_params,
        padded_observations,
        padded_actions,
        padded_rewards,
        reward_mask,
        gamma=GAMMA,
        lambda_=LAMBDA,
        epsilon=epsilon)
    # ppo_loss = 11.00110011
    logging.vlog(1, "Calculating PPO loss took %0.2f msec.", get_time(t))
    ppo_objective.append(-cur_ppo_loss)

    # Run optimizers.
    logging.vlog(1, "PPO Optimization")
    t1 = time.time()

    for j in range(num_optimizer_steps):
      t = time.time()
      # Update the optimizer state.
      ppo_opt_state = ppo_opt_step(
          j,
          ppo_opt_state,
          ppo_opt_update,
          policy_net_apply,
          policy_net_params,
          value_net_apply,
          value_net_params,
          padded_observations,
          padded_actions,
          padded_rewards,
          reward_mask,
          gamma=GAMMA,
          lambda_=LAMBDA,
          epsilon=epsilon)
      t2 = time.time()
      # Get the new params.
      new_policy_net_params = trax_opt.get_params(ppo_opt_state)
      if ((j + 1) %
          print_every_optimizer_steps == 0) or (j == num_optimizer_steps - 1):
        new_ppo_loss = ppo_loss(
            policy_net_apply,
            new_policy_net_params,
            policy_net_params,
            value_net_apply,
            value_net_params,
            padded_observations,
            padded_actions,
            padded_rewards,
            reward_mask,
            gamma=GAMMA,
            lambda_=LAMBDA,
            epsilon=epsilon)
        logging.vlog(1, "One PPO grad desc took: %0.2f msec", get_time(t, t2))
        logging.vlog(1, "PPO loss [%10.2f] -> [%10.2f]", cur_ppo_loss,
                     new_ppo_loss)
      # Update the params.
      policy_net_params = new_policy_net_params

    logging.vlog(1, "Total PPO loss reduction [%0.2f]%%",
                 (100 * (cur_ppo_loss - new_ppo_loss) / np.abs(cur_ppo_loss)))

    logging.vlog(1, "Value Optimization")

    for j in range(num_optimizer_steps):
      t = time.time()
      value_opt_state = value_opt_step(
          j,
          value_opt_state,
          value_opt_update,
          value_net_apply,
          padded_observations,
          padded_rewards,
          reward_mask,
          gamma=GAMMA)
      t2 = time.time()
      value_net_params = trax_opt.get_params(value_opt_state)
      if ((j + 1) %
          print_every_optimizer_steps == 0) or (j == num_optimizer_steps - 1):
        new_value_loss = value_loss(
            value_net_apply,
            value_net_params,
            padded_observations,
            padded_rewards,
            reward_mask,
            gamma=GAMMA)
        logging.vlog(1, "One value grad desc took: %0.2f msec", get_time(t, t2))
        logging.vlog(1, "Value loss [%10.2f] -> [%10.2f]", cur_value_loss,
                     new_value_loss)
    logging.vlog(1, "Total value loss reduction [%0.2f]%%",
                 (100 *
                  (cur_value_loss - new_value_loss) / np.abs(cur_value_loss)))

    logging.vlog(1, "Grad desc took %0.2f msec", get_time(t1))

    # Set the optimized params to new params.
    policy_net_params = trax_opt.get_params(ppo_opt_state)
    value_net_params = trax_opt.get_params(value_opt_state)

    logging.info(
        "Epoch [% 6d], average reward [%10.2f], ppo loss [%10.2f], "
        "value loss [%10.2f], took [%10.2f msec]", i, avg_reward, new_ppo_loss,
        new_value_loss, get_time(t0))

  logging.vlog(1, "value_losses: %s", np.stack(value_losses))
  logging.vlog(1, "ppo_objective: %s", np.stack(ppo_objective))
  logging.vlog(1, "average_rewards: %s", average_rewards)

  return ((policy_net_params, value_net_params), average_rewards,
          np.stack(value_losses), np.stack(ppo_objective))
