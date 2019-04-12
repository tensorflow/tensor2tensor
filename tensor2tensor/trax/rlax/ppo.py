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

"""PPO in JAX."""

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
from jax import vmap
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


def policy_net(jax_rng_key,
               batch_observations_shape,
               num_actions,
               bottom_layers=None):
  """A policy net function."""
  key1, _ = jax_random.split(jax_rng_key)

  # Use the bottom_layers as the bottom part of the network and just add the
  # required layers on top of it.
  if bottom_layers is None:
    bottom_layers = []
  bottom_layers.extend([stax.Dense(num_actions), stax.Softmax])

  net_init, net_apply = stax.serial(*bottom_layers)

  _, net_params = net_init(key1, batch_observations_shape)
  return net_params, net_apply


def value_net(jax_rng_key,
              batch_observations_shape,
              num_actions,
              bottom_layers=None):
  """A value net function."""
  del num_actions
  key1, _ = jax_random.split(jax_rng_key)

  if bottom_layers is None:
    bottom_layers = []
  bottom_layers.extend([
      stax.Dense(1),
  ])

  net_init, net_apply = stax.serial(*bottom_layers)

  _, net_params = net_init(key1, batch_observations_shape)
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
    observations = []
    rewards = []
    actions = []
    done = False

    observation = env.reset()
    observations.append(observation)
    while not done:
      # Run the policy, pick an action.
      predictions = policy_net_apply(policy_net_params, observation)

      # Greedy policy.
      action = np.argmax(predictions)
      if policy == "epsilon-greedy":
        # A schedule for epsilon is 1/k where k is the episode number sampled.
        if onp.random.random() < epsilon:
          # Choose an action at random.
          action = onp.random.randint(0, high=len(predictions))
        else:
          # Return the best action.
          action = np.argmax(predictions)
      elif policy == "categorical-sampling":
        action = int(onp.argwhere(onp.random.multinomial(1, predictions) == 1))

      # NOTE: Assumption, single batch.
      action = int(action)

      observation, reward, done, _ = env.step(action)

      observations.append(observation)
      rewards.append(reward)
      actions.append(action)

    # This means we are done
    assert done
    trajectories.append(
        (np.stack(observations), np.stack(actions), np.stack(rewards)))

  return trajectories


# This function can probably be simplified, ask how?
# Can we do something much simpler than lax.pad, maybe np.pad?
# Others?
def pad_trajectories(trajectories, boundary=10):
  """Pad trajectories to a bucket length that is a multiple of boundary."""

  # trajectories is a list of tuples of (observations, actions, rewards)
  # observations's length is one more than actions and rewards
  #
  # i.e. observations = (o_0, o_1, ... o_{T-1}, o_T)
  #           actions = (a_0, a_1, ... a_{T-1})
  #           rewards = (r_0, r_1, ... r_{T-1})

  # Given the above, let's compute max(T) over all trajectories.
  t_max = max(o.shape[0] for (o, a, r) in trajectories)

  # t_max - 1 is rounded to the next multiple of `boundary`
  boundary = int(boundary)
  bucket_length = boundary * int(np.ceil(float(t_max - 1) / boundary))

  # So all obs will be padded to t_max and actions and rewards to t_max - 1.
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
    padding_value = 0.0 if o.dtype == np.float32 else 0
    padded_obs = lax.pad(o, padding_value, padding_config)
    padded_observations.append(padded_obs)

    # Now pad actions and rewards.
    assert a.ndim == 1 and r.ndim == 1
    padding_config = ((0, num_to_pad, 0),)
    action_padding_value = 0.0 if a.dtype == np.float32 else 0
    reward_padding_value = 0.0 if r.dtype == np.float32 else 0
    padded_action = lax.pad(a, action_padding_value, padding_config)
    padded_actions.append(padded_action)
    padded_reward = lax.pad(r, reward_padding_value, padding_config)
    padded_rewards.append(padded_reward)

    # Also create the mask to use later.
    reward_mask = onp.ones_like(r, dtype=np.int32)
    reward_masks.append(lax.pad(reward_mask, 0, padding_config))

  return padded_lengths, np.stack(reward_masks), np.stack(
      padded_observations), np.stack(padded_actions), np.stack(padded_rewards)


# TODO(afrozm): Make this batched by default.
def rewards_to_go(rewards, reward_mask=1.0, gamma=0.99):
  r"""r2g[t] = \sum_{l=0}^{\infty}(\gamma^l * r_{t+l})."""
  time_steps = len(rewards)
  # r2g[t] = r[t] + (gamma * r2g[t+1])

  # First initialize like:
  # r2g[t] = r[t], for t = 0 to T-1
  r2g = list(rewards)

  # Then add the discounted version of the next time-step.
  # i = [T-2 .. 0]
  for i in range(time_steps - 2, -1, -1):
    r2g[i] += gamma * r2g[i + 1]

  # Makes this back into JAX's DeviceArray
  r2g = np.stack(list(r2g))

  return r2g * reward_mask


# TODO(afrozm): Make this batched by default.
@functools.partial(jit, static_argnums=(0,))
def value_loss(value_net_apply,
               value_net_params,
               observations,
               rewards,
               reward_mask=1.0,
               gamma=0.99):
  """L2 loss on the value function's outputs."""

  # Capturing the value_net_apply from the parent function's scope.
  # See: https://github.com/google/jax/issues/183
  def _value_function_loss_trajectory(value_net_params,
                                      observations,
                                      rewards,
                                      reward_mask=1.0,
                                      gamma=0.99):
    """Compute the actual loss for a trajectory."""
    r2g = rewards_to_go(rewards, reward_mask=reward_mask, gamma=gamma)
    v = value_net_apply(value_net_params, observations[:-1])
    v = np.squeeze(v) * reward_mask
    loss = v - r2g
    return np.sum(loss**2)

  batched_value_function_loss_trajectory = vmap(
      _value_function_loss_trajectory, in_axes=(None, 0, 0, 0), out_axes=0)

  return np.mean(
      batched_value_function_loss_trajectory(
          value_net_params, observations, rewards, reward_mask, gamma=gamma))


def deltas(predicted_values, rewards, reward_mask, gamma=0.99):
  r"""\delta_t = \sum_{l = 0}^{\infty}(r_t + \gamma * V(s_{t+1}) - V(s_t))."""
  # predicted_values are application of value net only the observations.
  # B x T+1

  # `d`s are basically one-step TD residuals.
  d = []
  _, T = rewards.shape  # pylint: disable=invalid-name
  for t in range(T):
    d.append(rewards[:, t] + (gamma * predicted_values[:, t + 1]) -
             predicted_values[:, t])

  return np.array(d).T * reward_mask


def gae_advantages(td_deltas, reward_mask, lambda_=0.95, gamma=0.99):
  r"""A_t = \sum_{l=0}^{\infty}(\gamma * \lambda)^{l}(\delta_{t+l})."""
  _, T = td_deltas.shape  # pylint: disable=invalid-name
  gl = lambda_ * gamma

  # [[1, gl, gl**2, ... gl**T-1]]
  # Not jittable, T should be a compile time constant.
  # gl_gp = np.geomspace(1, gl**T, T, endpoint=False).reshape(1, T)
  gl_geometric_progression = [1]
  for _ in range(1, T):
    gl_geometric_progression.append(gl_geometric_progression[-1] * gl)
  gl_gp = np.array(gl_geometric_progression)
  gl_gp = gl_gp.reshape((1, T))

  # td_deltas * gl_gp
  deltas_gl_gp = td_deltas * gl_gp

  # A0 - advantage for 0th time-step, across all batches.
  As = []  # pylint: disable=invalid-name
  A0 = np.sum(deltas_gl_gp, axis=1)  # (B,)  # pylint: disable=invalid-name
  As.append(A0)

  # Now compute the other advantages.
  for t in range(1, T):
    As.append((As[-1] - td_deltas[:, t - 1]) / gl)

  return np.stack(As).T * reward_mask


def chosen_probabs(probab_observations, actions):
  """Picks out the probabilities of the actions along batch and time-steps.

  Args:
    probab_observations: `[B, T, #actions]` ndarray, where
      probab_observations[b, t, i] contains the probability of action = i at the
      t^th time-step in the b^th trajectory.
    actions: `[B, T]` ndarray, with each entry in [0, #actions) denoting which
      action was chosen in the b^th trajectory's t^th time-step.

  Returns:
    `[B, T]` ndarray with the probabilities of the chosen actions.
  """
  b, t = actions.shape
  return probab_observations[np.arange(b)[:, None], np.arange(t), actions]


def probab_ratios(policy_net_apply, old_policy_params, new_policy_params,
                  observations, actions, reward_mask):
  """Calculates the probaility ratios for each time-step in a trajectory."""
  p_old = policy_net_apply(old_policy_params, observations)
  p_new = policy_net_apply(new_policy_params, observations)

  bp_old = chosen_probabs(p_old, actions)
  bp_new = chosen_probabs(p_new, actions)

  # Add a small number to bp_old, where reward_mask is 0, this is just to help
  # never to divide by 0.
  bp_old = bp_old + (0.1 * np.abs(reward_mask - 1))

  ret_val = (bp_new * reward_mask) / bp_old

  return ret_val


def clipped_probab_ratios(bpr, reward_mask, epsilon=0.2):
  return reward_mask * np.clip(bpr, 1 - epsilon, 1 + epsilon)


def clipped_objective(bpr, adv, reward_mask, epsilon=0.2):
  c1 = bpr * adv
  c2 = clipped_probab_ratios(bpr, reward_mask, epsilon=epsilon) * adv
  return np.minimum(c1, c2)


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
  # V(s_t) forall s & t
  value_function = np.squeeze(
      value_net_apply(value_net_params, padded_observations))
  td_deltas = deltas(value_function, padded_rewards, reward_mask, gamma=gamma)
  advantages = gae_advantages(
      td_deltas, reward_mask, lambda_=lambda_, gamma=gamma)
  ratios = probab_ratios(policy_net_apply, old_policy_params, new_policy_params,
                         padded_observations, padded_actions, reward_mask)
  clipped_loss = clipped_objective(
      ratios, advantages, reward_mask, epsilon=epsilon)
  return -np.sum(clipped_loss)


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
    random_seed=None):
  """Runs the training loop for PPO, with fixed policy and value nets."""
  jax_rng_key = trax.get_random_number_generator_and_set_seed(random_seed)

  value_losses = []
  ppo_objective = []
  average_rewards = []

  env = env if env is not None else gym.make(env_name)

  batch_observations_shape = (-1,) + env.observation_space.shape

  assert isinstance(env.action_space, gym.spaces.Discrete)
  num_actions = env.action_space.n

  # TODO(afrozm): Have a single net for both policy and action.
  assert policy_and_value_net_fun is None

  # Initialize the policy and value functions.
  assert policy_net_fun and value_net_fun
  jax_rng_key, key1, key2 = jax_random.split(jax_rng_key, num=3)

  policy_net_params, policy_net_apply = policy_net_fun(
      key1, batch_observations_shape, num_actions)
  value_net_params, value_net_apply = value_net_fun(
      key2, batch_observations_shape, num_actions)

  # Initialize the optimizers.
  assert policy_optimizer_fun and value_optimizer_fun

  ppo_opt_state, ppo_opt_update = policy_optimizer_fun(policy_net_params)
  value_opt_state, value_opt_update = value_optimizer_fun(value_net_params)

  for i in range(epochs):
    t = time.time()
    t0 = t
    trajs = collect_trajectories(
        env,
        policy_net_apply,
        policy_net_params,
        num_trajectories=batch_size,
        policy=POLICY,
        epsilon=(10.0 / (i + 10.0)))  # this is a different epsilon.

    avg_reward = float(sum(np.sum(traj[2]) for traj in trajs)) / len(trajs)
    average_rewards.append(avg_reward)

    logging.debug("Average sum rewards [%0.2f]", avg_reward)
    logging.debug("Collecting trajectories took %0.2f msec.", get_time(t))
    logging.debug("Average Trajectory size [%0.2f]",
                  float(sum(len(traj[0]) for traj in trajs)) / len(trajs))

    t = time.time()
    (_, reward_mask, padded_observations, padded_actions,
     padded_rewards) = pad_trajectories(
         trajs, boundary=20)

    logging.debug("Padding trajectories took %0.2f msec.", get_time(t))
    logging.debug("Padded Actions' shape [%s]", str(padded_actions.shape))

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

    logging.debug("Calculating value loss took %0.2f msec.", get_time(t))
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
    logging.debug("Calculating PPO loss took %0.2f msec.", get_time(t))
    ppo_objective.append(-cur_ppo_loss)

    # Run optimizers.
    logging.debug("PPO Optimization")
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
        logging.debug("One PPO grad desc took: %0.2f msec", get_time(t, t2))
        logging.debug("PPO loss [%10.2f] -> [%10.2f]", cur_ppo_loss,
                      new_ppo_loss)
      # Update the params.
      policy_net_params = new_policy_net_params

    logging.debug("Total PPO loss reduction [%0.2f]%%",
                  (100 * (cur_ppo_loss - new_ppo_loss) / np.abs(cur_ppo_loss)))

    logging.debug("Value Optimization")

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
        logging.debug("One value grad desc took: %0.2f msec", get_time(t, t2))
        logging.debug("Value loss [%10.2f] -> [%10.2f]", cur_value_loss,
                      new_value_loss)
    logging.debug("Total value loss reduction [%0.2f]%%",
                  (100 *
                   (cur_value_loss - new_value_loss) / np.abs(cur_value_loss)))

    logging.debug("Grad desc took %0.2f msec", get_time(t1))

    # Set the optimized params to new params.
    policy_net_params = trax_opt.get_params(ppo_opt_state)
    value_net_params = trax_opt.get_params(value_opt_state)

    logging.info(
        "Epoch [% 6d], average reward [%10.2f], ppo loss [%10.2f], "
        "value loss [%10.2f], took [%10.2f msec]", i, avg_reward, new_ppo_loss,
        new_value_loss, get_time(t0))

  logging.debug("value_losses: %s", np.stack(value_losses))
  logging.debug("ppo_objective: %s", np.stack(ppo_objective))
  logging.debug("average_rewards: %s", average_rewards)

  return ((policy_net_params, value_net_params), average_rewards,
          np.stack(value_losses), np.stack(ppo_objective))
