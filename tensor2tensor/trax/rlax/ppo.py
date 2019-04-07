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
from absl import app
import gym

from jax import grad
from jax import jit
from jax import lax
from jax import numpy as np
from jax import vmap
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense
from jax.experimental.stax import Relu
from jax.experimental.stax import Softmax

import numpy as onp

DEBUG_LOGGING = False
GAMMA = 0.99
LAMBDA = 0.95
EPOCHS = 50  # 100
NUM_OPTIMIZER_STEPS = 100
PRINT_EVERY_OPTIMIZER_STEP = 20
BATCH_TRAJECTORIES = 32
POLICY = "categorical-sampling"


# TODO(afrozm): Have a single net for both policy and value.
def initialize_policy_and_value_nets(num_actions, batch_observations_shape):
  """Setup and initialize the policy and value networks."""
  policy_net_init, policy_net_apply = stax.serial(
      Dense(16),
      Relu,
      Dense(4),
      Relu,
      Dense(num_actions),
      Softmax,
  )

  _, policy_net_params = policy_net_init(
      batch_observations_shape)

  value_net_init, value_net_apply = stax.serial(
      Dense(16),
      Relu,
      Dense(4),
      Relu,
      Dense(1),  # 1 since we want to predict reward using value network.
  )

  _, value_net_params = value_net_init(
      batch_observations_shape)

  return ((policy_net_params, policy_net_apply), (value_net_params,
                                                  value_net_apply))


def initialize_optimizers(policy_net_params, value_net_params):
  """Initialize optimizers for the policy and value params."""
  # ppo_opt_init, ppo_opt_update = optimizers.sgd(step_size=1e-3)
  # val_opt_init, val_opt_update = optimizers.sgd(step_size=1e-3)
  ppo_opt_init, ppo_opt_update = optimizers.adam(
      step_size=1e-3, b1=0.9, b2=0.999, eps=1e-08)
  value_opt_init, value_opt_update = optimizers.adam(
      step_size=1e-3, b1=0.9, b2=0.999, eps=1e-08)

  ppo_opt_state = ppo_opt_init(policy_net_params)
  value_opt_state = value_opt_init(value_net_params)

  return (ppo_opt_state, ppo_opt_update), (value_opt_state, value_opt_update)


# Should this be collect 'n' trajectories, or
# Run the env for 'n' steps and take completed trajectories, or
# Any other option?
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

      if DEBUG_LOGGING:
        print("With predictions: ", predictions, " chose action: ", action)

      # NOTE: Assumption, single batch.
      action = int(action)

      observation, reward, done, _ = env.step(action)

      observations.append(observation)
      rewards.append(reward)
      actions.append(action)

    # This means we are done
    assert done
    trajectories.append((np.stack(observations), np.stack(actions),
                         np.stack(rewards)))

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
  if DEBUG_LOGGING:
    print("t_max: %s" % t_max)

  # t_max - 1 is rounded to the next multiple of `boundary`
  boundary = int(boundary)
  bucket_length = boundary * int(np.ceil(float(t_max - 1) / boundary))
  if DEBUG_LOGGING:
    print("bucket_length: %s" % bucket_length)

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
    if DEBUG_LOGGING:
      print("num_to_pad: %s" % num_to_pad)
    padding_config = [(0, num_to_pad, 0)]
    for _ in range(o.ndim - 1):
      padding_config.append((0, 0, 0))
    padding_config = tuple(padding_config)
    if DEBUG_LOGGING:
      print("padding_config: %s" % str(padding_config))
    padding_value = 0.0 if o.dtype == np.float32 else 0
    if DEBUG_LOGGING:
      print("padding_value: %s" % padding_value)
    padded_obs = lax.pad(o, padding_value, padding_config)
    padded_observations.append(padded_obs)

    # Now pad actions and rewards.
    assert a.ndim == 1 and r.ndim == 1
    padding_config = ((0, num_to_pad, 0),)
    if DEBUG_LOGGING:
      print("action/reward padding_config: %s" % str(padding_config))
    action_padding_value = 0.0 if a.dtype == np.float32 else 0
    reward_padding_value = 0.0 if r.dtype == np.float32 else 0
    if DEBUG_LOGGING:
      print("action_padding_value: %s" % action_padding_value)
    padded_action = lax.pad(a, action_padding_value, padding_config)
    padded_actions.append(padded_action)
    if DEBUG_LOGGING:
      print("reward_padding_value: %s" % reward_padding_value)
    padded_reward = lax.pad(r, reward_padding_value, padding_config)
    padded_rewards.append(padded_reward)

    # Also create the mask to use later.
    reward_mask = onp.ones_like(r, dtype=np.int32)
    reward_masks.append(lax.pad(reward_mask, 0, padding_config))

  return padded_lengths, np.stack(reward_masks), np.stack(
      padded_observations), np.stack(padded_actions), np.stack(padded_rewards)


def rewards_to_go_discounted(rewards, reward_mask=1.0, gamma=0.99):
  r"""r2g[t] = \sum_{l=0}^{\infty}(\gamma^l * r_{t+l})."""
  time_steps = len(rewards)
  # r2g[t] = r[t] + (gamma * r2g[t+1])

  # First initialize like:
  # r2g[t] = r[t], for t = 0 to T-1
  rewards_to_go = list(rewards)

  # Then add the discounted version of the next time-step.
  # i = [T-2 .. 0]
  for i in range(time_steps - 2, -1, -1):
    rewards_to_go[i] += gamma * rewards_to_go[i + 1]

  # Makes this back into JAX's DeviceArray
  rewards_to_go = np.stack(list(rewards_to_go))

  return rewards_to_go * reward_mask


def batched_avg_value_function_loss(value_net_apply,
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
    r2g = rewards_to_go_discounted(
        rewards, reward_mask=reward_mask, gamma=gamma)
    v = value_net_apply(value_net_params, observations[:-1])
    v = np.squeeze(v) * reward_mask
    loss = v - r2g
    return np.sum(loss**2)

  batched_value_function_loss_trajectory = vmap(
      _value_function_loss_trajectory, in_axes=(None, 0, 0, 0), out_axes=0)

  return np.mean(
      batched_value_function_loss_trajectory(
          value_net_params, observations, rewards, reward_mask, gamma=gamma))


def batched_deltas(predicted_values, rewards, reward_mask, gamma=0.99):
  r"""\delta_t = \sum_{l = 0}^{\infty}(r_t + \gamma * V(s_{t+1}) - V(s_t))."""
  # predicted_values are application of value net only the observations.
  # B x T+1

  deltas = []
  _, T = rewards.shape  # pylint: disable=invalid-name
  for t in range(T):
    deltas.append(rewards[:, t] + (gamma * predicted_values[:, t + 1]) -
                  predicted_values[:, t])

  return np.array(deltas).T * reward_mask


def batched_gae_advantages(deltas, reward_mask, lamda=0.95,  # NOTYPO
                           gamma=0.99):
  r"""A_t = \sum_{l=0}^{\infty}(\gamma * \lambda)^{l}(\delta_{t+l})."""
  _, T = deltas.shape  # pylint: disable=invalid-name
  gl = lamda * gamma  # NOTYPO

  # [[1, gl, gl**2, ... gl**T-1]]
  # Not jittable, T should be a compile time constant.
  # gl_gp = np.geomspace(1, gl**T, T, endpoint=False).reshape(1, T)
  gl_geometric_progression = [1]
  for _ in range(1, T):
    gl_geometric_progression.append(gl_geometric_progression[-1] * gl)
  gl_gp = np.array(gl_geometric_progression)
  gl_gp = gl_gp.reshape((1, T))

  # deltas * gl_gp
  deltas_gl_gp = deltas * gl_gp

  # A0 - advantage for 0th time-step, across all batches.
  As = []  # pylint: disable=invalid-name
  A0 = np.sum(deltas_gl_gp, axis=1)  # (B,)  # pylint: disable=invalid-name
  As.append(A0)

  # Now compute the other advantages.
  for t in range(1, T):
    As.append((As[-1] - deltas[:, t - 1]) / gl)

  return np.stack(As).T * reward_mask


def batched_probabs(probab_observations, actions):
  b, t = actions.shape
  return probab_observations[np.arange(b)[:, None], np.arange(t), actions]


def batched_probab_ratios(policy_net_apply, old_policy_params,
                          new_policy_params, observations, actions,
                          reward_mask):
  """Calculates the probaility ratios for each time-step in a trajectory."""
  p_old = policy_net_apply(old_policy_params, observations)
  p_new = policy_net_apply(new_policy_params, observations)

  bp_old = batched_probabs(p_old, actions)
  bp_new = batched_probabs(p_new, actions)

  if DEBUG_LOGGING:
    print("bp_old: ", bp_old)
    print("bp_new: ", bp_new)

  # Add a small number to bp_old, where reward_mask is 0, this is just to help
  # never to divide by 0.
  bp_old = bp_old + (0.1 * np.abs(reward_mask - 1))

  if DEBUG_LOGGING:
    print("masked bp_old: ", bp_old)

  ret_val = (bp_new * reward_mask) / bp_old

  if DEBUG_LOGGING:
    print("ret_val: ", ret_val)

  return ret_val


def batched_clipped_probab_ratios(bpr, reward_mask, epsilon=0.2):
  return reward_mask * np.clip(bpr, 1 - epsilon, 1 + epsilon)


def batched_clipped_objective(bpr, adv, reward_mask, epsilon=0.2):
  c1 = bpr * adv
  c2 = batched_clipped_probab_ratios(bpr, reward_mask, epsilon=epsilon) * adv
  return np.minimum(c1, c2)


def batched_ppo_loss(policy_net_apply,
                     new_policy_params,
                     old_policy_params,
                     value_net_apply,
                     value_net_params,
                     padded_observations,
                     padded_actions,
                     padded_rewards,
                     reward_mask,
                     gamma=0.99,
                     lamda=0.95,  # NOTYPO
                     epsilon=0.2):
  """PPO objective, with an eventual minus sign."""
  # V(s_t) forall s & t
  value_function = np.squeeze(
      value_net_apply(value_net_params, padded_observations))
  deltas = batched_deltas(
      value_function, padded_rewards, reward_mask, gamma=gamma)
  advantages = batched_gae_advantages(
      deltas, reward_mask, lamda=lamda, gamma=gamma)  # NOTYPO
  ratios = batched_probab_ratios(policy_net_apply, old_policy_params,
                                 new_policy_params, padded_observations,
                                 padded_actions, reward_mask)
  clipped_loss = batched_clipped_objective(ratios, advantages, reward_mask,
                                           epsilon=epsilon)
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
                 lamda=0.95,  # NOTYPO
                 epsilon=0.1):
  """PPO optimizer step."""
  new_policy_params = optimizers.get_params(opt_state)
  g = grad(
      batched_ppo_loss, argnums=1)(
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
          lamda=lamda,  # NOTYPO
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
  value_params = optimizers.get_params(opt_state)
  # Note this partial application here and argnums above in ppo_opt_step.
  g = grad(functools.partial(batched_avg_value_function_loss, value_net_apply))(
      value_params,
      padded_observations,
      padded_rewards,
      reward_mask,
      gamma=gamma)
  return opt_update(i, g, opt_state)


def main(unused_argv):
  onp.random.seed(0)

  value_losses = []
  ppo_objective = []
  average_rewards = []

  env = gym.make("CartPole-v0")

  print("Initial observation: ", env.reset())

  for i in range(100):
    random_action = env.action_space.sample()
    obs, rew, done, _ = env.step(random_action)
    print("[%s] reward [%s], done [%s] and obs [%s]" % (i, rew, done, obs))
    if done:
      print("Done, so exiting, step: ", i)
      env.close()
      break

  print("action_space.shape", env.action_space.shape)
  print("observation_space.shape", env.observation_space.shape)

  batch_observations_shape = (-1,) + env.observation_space.shape

  assert isinstance(env.action_space, gym.spaces.Discrete)
  num_actions = env.action_space.n

  print("batch_observations_shape: ", batch_observations_shape)
  print("num_actions: ", num_actions)

  ((policy_net_params, policy_net_apply),
   (value_net_params, value_net_apply)) = initialize_policy_and_value_nets(
       num_actions, batch_observations_shape)

  (ppo_opt_state, ppo_opt_update), (value_opt_state,
                                    value_opt_update) = initialize_optimizers(
                                        policy_net_params, value_net_params)

  for i in range(EPOCHS):
    t = time.time()
    t0 = t
    trajs = collect_trajectories(
        env,
        policy_net_apply,
        policy_net_params,
        num_trajectories=BATCH_TRAJECTORIES,
        policy=POLICY,
        epsilon=(10.0 / (i + 10.0)))  # this is a different epsilon.
    print("Took ", round((time.time() - t) * 1000, 2),
          "msec to collect trajectories.")

    print("Average Trajectory size: ",
          float(sum(len(traj[0]) for traj in trajs)) / len(trajs))
    avg_reward = float(sum(np.sum(traj[2]) for traj in trajs)) / len(trajs)
    average_rewards.append(avg_reward)
    print("Average sum rewards: ", avg_reward)

    if (avg_reward > 190.0) and (i % 5 == 0):
      print("policy_net_params:\n", policy_net_params)
      print("value_net_params:\n", value_net_params)

    t = time.time()
    (_, reward_mask, padded_observations, padded_actions,
     padded_rewards) = pad_trajectories(trajs, boundary=20)
    print("Took ", round((time.time() - t) * 1000, 2),
          "msec to pad trajectories.")

    print("Padded Observations' shape: ", padded_observations.shape)
    print("Padded Actions' shape:      ", padded_actions.shape)
    print("Padded Rewards' shape:      ", padded_rewards.shape)

    # Linear annealing from 0.1 to 0.0
    epsilon = 0.1 if EPOCHS == 1 else 0.1 * (1.0 - (i / (EPOCHS - 1)))

    t = time.time()
    val_loss = jit(
        batched_avg_value_function_loss, static_argnums=(0,))(
            value_net_apply,
            value_net_params,
            padded_observations,
            padded_rewards,
            reward_mask,
            gamma=GAMMA)

    print("Took ", round((time.time() - t) * 1000, 2),
          "msec to calculate value loss = ", val_loss)
    value_losses.append(val_loss)

    t = time.time()
    ppo_loss = jit(
        batched_ppo_loss, static_argnums=(0,
                                          3))(policy_net_apply,
                                              policy_net_params,
                                              policy_net_params,
                                              value_net_apply,
                                              value_net_params,
                                              padded_observations,
                                              padded_actions,
                                              padded_rewards,
                                              reward_mask,
                                              gamma=GAMMA,
                                              lamda=LAMBDA,  # NOTYPO
                                              epsilon=epsilon)
    # ppo_loss = 11.00110011
    print("Took ", round((time.time() - t) * 1000, 2),
          "msec to calculate ppo loss = ", ppo_loss)
    ppo_objective.append(-ppo_loss)

    # Run optimizers.
    t1 = time.time()

    print("PPO objective optimization.")

    for j in range(NUM_OPTIMIZER_STEPS):
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
          lamda=LAMBDA,  # NOTYPO
          epsilon=epsilon)
      t2 = time.time()
      # Get the new params.
      new_policy_net_params = optimizers.get_params(ppo_opt_state)
      if ((j + 1) %
          PRINT_EVERY_OPTIMIZER_STEP == 0) or (j == NUM_OPTIMIZER_STEPS - 1):
        new_ppo_loss = jit(
            batched_ppo_loss, static_argnums=(0,
                                              3))(policy_net_apply,
                                                  new_policy_net_params,
                                                  policy_net_params,
                                                  value_net_apply,
                                                  value_net_params,
                                                  padded_observations,
                                                  padded_actions,
                                                  padded_rewards,
                                                  reward_mask,
                                                  gamma=GAMMA,
                                                  lamda=LAMBDA,  # NOTYPO
                                                  epsilon=epsilon)
        print("Took ", round((t2 - t) * 1000, 2),
              "msec to do one step ppo grad desc")
        print("New ppo loss[", j, "]: ", new_ppo_loss, " vs old ppo loss: ",
              ppo_loss)
      # Update the params.
      policy_net_params = new_policy_net_params

    print("Total ppo loss reduction: ",
          100 * (ppo_loss - new_ppo_loss) / np.abs(ppo_loss), "%")

    print("Value optimization.")

    for j in range(NUM_OPTIMIZER_STEPS):
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
      value_net_params = optimizers.get_params(value_opt_state)
      if ((j + 1) %
          PRINT_EVERY_OPTIMIZER_STEP == 0) or (j == NUM_OPTIMIZER_STEPS - 1):
        new_val_loss = jit(
            batched_avg_value_function_loss, static_argnums=(0,))(
                value_net_apply,
                value_net_params,
                padded_observations,
                padded_rewards,
                reward_mask,
                gamma=GAMMA)
        print("Took ", round((t2 - t) * 1000, 2),
              "msec to do one step value grad desc")
        print("New value loss[", j, "]: ", new_val_loss, " vs old value loss: ",
              val_loss)
    print("Total value loss reduction: ",
          100 * (val_loss - new_val_loss) / val_loss, "%")

    print("Took ", round((time.time() - t1) * 1000, 2), "msec to do grad desc")

    # Set the optimized params to new params.
    policy_net_params = optimizers.get_params(ppo_opt_state)
    value_net_params = optimizers.get_params(value_opt_state)

    print("Epoch [%s] took [%s]msec." % (i, round(
        (time.time() - t0) * 1000, 2)))
    print()

  print("value_losses: ", np.stack(value_losses))
  print("ppo_objective: ", np.stack(ppo_objective))
  print("average_rewards: ", average_rewards)


if __name__ == "__main__":
  app.run(main)
