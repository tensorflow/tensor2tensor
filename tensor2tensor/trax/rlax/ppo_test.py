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

"""Tests for tensor2tensor.trax.rlax.ppo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
from jax import random as jax_random
import numpy as np
from tensor2tensor.trax import layers
from tensor2tensor.trax import trax
from tensor2tensor.trax.rlax import fake_env
from tensor2tensor.trax.rlax import ppo
from tensorflow import test


class PpoTest(test.TestCase):

  def setUp(self):
    self.rng_key = trax.get_random_number_generator_and_set_seed(0)

  def test_policy_net(self):
    observation_shape = (3, 4)
    num_actions = 2
    policy_params, policy_apply = ppo.policy_net(
        self.rng_key,
        (-1, -1) + observation_shape,
        num_actions,
        # flatten except batch and time
        # step dimensions.
        [layers.Flatten(num_axis_to_keep=2)])

    # Generate a batch of observations.
    batch = 2
    time_steps = 10
    batch_of_observations = np.random.uniform(
        size=(batch, time_steps) + observation_shape)

    # Apply the policy net on observations
    policy_output = policy_apply(batch_of_observations, policy_params)

    # Verify certain expectations on the output.
    self.assertEqual((batch, time_steps, num_actions), policy_output.shape)

    # Also exp of last axis normalizes to 1, since these are log-probabilities.
    sum_actions = np.sum(np.exp(policy_output), axis=-1)
    self.assertAllClose(np.ones_like(sum_actions), sum_actions)

  def test_value_net(self):
    observation_shape = (3, 4, 5)
    num_actions = 2
    value_params, value_apply = ppo.value_net(
        self.rng_key,
        (-1, -1) + observation_shape,
        num_actions, [layers.Flatten(num_axis_to_keep=2)])
    batch = 2
    time_steps = 10
    batch_of_observations = np.random.uniform(
        size=(batch, time_steps) + observation_shape)
    value_output = value_apply(batch_of_observations, value_params)

    # NOTE: The extra dimension at the end because of Dense(1).
    self.assertEqual((batch, time_steps, 1), value_output.shape)

  def test_policy_and_value_net(self):
    observation_shape = (3, 4, 5)
    batch_observation_shape = (-1, -1) + observation_shape
    num_actions = 2
    pnv_params, pnv_apply = ppo.policy_and_value_net(
        self.rng_key, batch_observation_shape, num_actions,
        [layers.Flatten(num_axis_to_keep=2)])
    batch = 2
    time_steps = 10
    batch_of_observations = np.random.uniform(
        size=(batch, time_steps) + observation_shape)
    pnv_output = pnv_apply(batch_of_observations, pnv_params)

    # Output is a list, first is probab of actions and the next is value output.
    self.assertEqual(2, len(pnv_output))
    self.assertEqual((batch, time_steps, num_actions), pnv_output[0].shape)
    self.assertEqual((batch, time_steps, 1), pnv_output[1].shape)

  def test_collect_trajectories(self):
    observation_shape = (2, 3, 4)
    num_actions = 2
    policy_params, policy_apply = ppo.policy_net(
        self.rng_key,
        (-1, -1) + observation_shape,
        num_actions,
        # flatten except batch and time
        # step dimensions.
        [layers.Flatten(num_axis_to_keep=2)])

    # We'll get done at time-step #5, starting from 0, therefore in 6 steps.
    done_time_step = 5
    env = fake_env.FakeEnv(
        observation_shape, num_actions, done_time_step=done_time_step)

    num_trajectories = 5
    trajectories = ppo.collect_trajectories(
        env,
        policy_apply,
        policy_params,
        num_trajectories,
        policy="categorical-sampling")

    # Number of trajectories is as expected.
    self.assertEqual(num_trajectories, len(trajectories))

    # Shapes of observations, actions and rewards are as expected.
    for observations, actions, rewards in trajectories:
      # observations are one more in number than rewards or actions.
      self.assertEqual((done_time_step + 2,) + observation_shape,
                       observations.shape)
      self.assertEqual((done_time_step + 1,), actions.shape)
      self.assertEqual((done_time_step + 1,), rewards.shape)

  def test_collect_trajectories_max_timestep(self):
    observation_shape = (2, 3, 4)
    num_actions = 2
    policy_params, policy_apply = ppo.policy_net(
        self.rng_key,
        (-1, -1) + observation_shape,
        num_actions,
        # flatten except batch and time
        # step dimensions.
        [layers.Flatten(num_axis_to_keep=2)])

    # We'll get done at time-step #5, starting from 0, therefore in 6 steps.
    done_time_step = 5
    env = fake_env.FakeEnv(
        observation_shape, num_actions, done_time_step=done_time_step)

    num_trajectories = 5

    # Let's collect trajectories only till `max_timestep`.
    max_timestep = 3

    # we're testing when we early stop the trajectory.
    assert max_timestep < done_time_step

    trajectories = ppo.collect_trajectories(
        env,
        policy_apply,
        policy_params,
        num_trajectories,
        policy="categorical-sampling",
        max_timestep=max_timestep)

    # Number of trajectories is as expected.
    self.assertEqual(num_trajectories, len(trajectories))

    # Shapes of observations, actions and rewards are as expected.
    for observations, actions, rewards in trajectories:
      # observations are one more in number than rewards or actions.
      self.assertEqual((max_timestep,) + observation_shape,
                       observations.shape)
      self.assertEqual((max_timestep - 1,), actions.shape)
      self.assertEqual((max_timestep - 1,), rewards.shape)

  def test_pad_trajectories(self):
    observation_shape = (2, 3, 4)
    trajectories = []
    num_trajectories = 7
    num_actions = 10

    # Time-steps are between [min_allowable_time_step, max_allowable_time_step]
    max_allowable_time_step = 19
    min_allowable_time_step = 5

    # The actual max we see in the data.
    max_time_step = -1

    # Bucket length.
    bucket_length = 15

    # Make `num_trajectories` random trajectories.
    for i in range(num_trajectories):
      time_steps = np.random.randint(min_allowable_time_step,
                                     max_allowable_time_step + 1)
      if time_steps > max_time_step:
        max_time_step = time_steps
      observations = np.random.randint(
          0, 255, size=(time_steps + 1,) + observation_shape).astype(np.uint8)
      rewards = np.random.uniform(size=(time_steps,)).astype(np.float32)
      actions = np.random.randint(
          0, num_actions, size=(time_steps,)).astype(np.int32)
      trajectories.append((observations, rewards, actions))

    # Now pad these trajectories.
    padded_trajectories = ppo.pad_trajectories(
        trajectories, boundary=bucket_length)

    # Expected padding.
    i = 1
    while i * bucket_length < max_time_step:
      i += 1
    expected_padding = i * bucket_length

    # Get the padded objects.
    (pad_lengths, reward_mask, padded_observations, padded_actions,
     padded_rewards) = padded_trajectories

    # Expectations on the padded shapes.
    self.assertEqual(padded_observations.shape, (
        num_trajectories,
        expected_padding + 1,
    ) + observation_shape)
    self.assertEqual(padded_actions.shape, (num_trajectories, expected_padding))
    self.assertEqual(padded_rewards.shape, (num_trajectories, expected_padding))
    self.assertEqual(reward_mask.shape, (num_trajectories, expected_padding))

    # Assert that the padding lengths and reward mask are consistent.
    self.assertAllEqual(
        np.full((num_trajectories,), expected_padding),
        np.array(np.sum(reward_mask, axis=1)) + pad_lengths)

  def test_rewards_to_go(self):
    rewards = np.array([
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rewards_mask = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    gamma = 0.5

    rewards_to_go = ppo.rewards_to_go(rewards, rewards_mask, gamma)

    self.assertAllEqual(
        np.array([
            [5, 8, 12, 16, 16, 0, 0, 0],
            [1.984375, 1.96875, 1.9375, 1.875, 1.75, 1.5, 1.0, 0],
        ]), rewards_to_go)

  def test_rewards_to_go_really_long_sequences(self):
    T = 1200  # pylint: disable=invalid-name

    rewards = np.random.uniform(1e-3, 1e-2, (1, T))

    # Make a mask, clear out a fixed number `L` of 1s from the end.
    L = 36  # pylint: disable=invalid-name
    assert L < T
    rewards_mask = np.ones_like(rewards)
    rewards_mask[0, L:] = 0

    gamma = 0.94

    actual_r2g = ppo.rewards_to_go(rewards, rewards_mask, gamma).reshape(-1)

    # Let's compute r2g the slow way.
    masked_rewards = (rewards_mask * rewards).reshape(-1)
    expected_r2g = np.zeros_like(masked_rewards)
    for t in range(T):
      for j in range(t, T):
        expected_r2g[t] += (gamma**(j-t)) * masked_rewards[j]

    self.assertAllClose(expected_r2g, actual_r2g)

  def test_value_loss(self):
    rewards = np.array([
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rewards_mask = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    gamma = 0.5

    # Random observations and a value function that returns a constant value.
    # NOTE: Observations have an extra time-step.
    B, T = rewards.shape  # pylint: disable=invalid-name
    observation_shape = (210, 160, 3)  # atari pong
    random_observations = np.random.uniform(size=(B, T + 1) + observation_shape)

    def value_net_apply(observations, params):
      del params
      # pylint: disable=invalid-name
      B, T_p_1, OBS = (observations.shape[0], observations.shape[1],
                       observations.shape[2:])
      del OBS
      return np.ones((B, T_p_1, 1))
      # pylint: enable=invalid-name

    with jax.disable_jit():
      value_loss = ppo.value_loss(
          value_net_apply, [],
          random_observations,
          rewards,
          rewards_mask,
          gamma=gamma)

    self.assertNear(53.3637084961, value_loss, 1e-6)

  def test_deltas(self):
    rewards = np.array([
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rewards_mask = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    B, T = rewards.shape  # pylint: disable=invalid-name

    # Say, all predicted values are 1.
    predicted_values = np.ones((B, T + 1))

    gamma = 1.0

    td_residuals = ppo.deltas(predicted_values, rewards, rewards_mask, gamma)

    # With V(s) being the same for all s, td_residuals should be
    # equal to the rewards + (\gamma - 1)*v(s), masked in the right places.
    truncated_pv = predicted_values[:, :-1]
    masked_rewards = rewards * rewards_mask
    expected_residuals = (masked_rewards +
                          (gamma - 1) * truncated_pv) * rewards_mask
    self.assertAllEqual(expected_residuals, td_residuals)

    gamma = 0.5
    td_residuals = ppo.deltas(predicted_values, rewards, rewards_mask, gamma)
    expected_residuals = (masked_rewards +
                          (gamma - 1) * truncated_pv) * rewards_mask
    self.assertAllEqual(expected_residuals, td_residuals)

  def test_gae_advantages(self):
    td_deltas = np.array([
        [1, 2, 4, 8, 16, 32, 64, 128],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ])

    rewards_mask = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ])

    gamma = 0.5
    lambda_ = 1.0

    expected_gae_advantages = np.array([
        [5, 8, 12, 16, 16, 0, 0, 0],
        [1.984375, 1.96875, 1.9375, 1.875, 1.75, 1.5, 1.0, 0],
    ])

    gae_advantages = ppo.gae_advantages(td_deltas * rewards_mask, rewards_mask,
                                        lambda_, gamma)
    self.assertAllEqual(expected_gae_advantages, gae_advantages)

    gamma = 1.0
    lambda_ = 0.5

    gae_advantages = ppo.gae_advantages(td_deltas * rewards_mask, rewards_mask,
                                        lambda_, gamma)
    self.assertAllEqual(expected_gae_advantages, gae_advantages)

  def test_chosen_probabs(self):
    # Shape (2, 2+1, 3)
    probab_observations = np.array(
        [[[0.1, 0.2, 0.7], [0.4, 0.1, 0.5], [0.2, 0.4, 0.4]],
         [[0.3, 0.1, 0.6], [0.1, 0.1, 0.8], [0.2, 0.4, 0.4]]]
    )

    # Shape (2, 2)
    actions = np.array([[1, 2], [0, 1]])

    chosen_probabs = ppo.chosen_probabs(probab_observations, actions)

    self.assertAllEqual(np.array([[0.2, 0.5], [0.3, 0.1]]), chosen_probabs)

  def test_compute_probab_ratios(self):
    p_old = np.array([[
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
        [np.log(0.4), np.log(0.1), np.log(0.4), np.log(0.1)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
    ], [
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.1), np.log(0.4), np.log(0.4)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.2), np.log(0.6), np.log(0.1)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
    ]])

    p_new = np.array([[
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.4), np.log(0.1), np.log(0.1), np.log(0.3)],
        [np.log(0.1), np.log(0.2), np.log(0.1), np.log(0.6)],
        [np.log(0.3), np.log(0.1), np.log(0.5), np.log(0.1)],
        [np.log(0.1), np.log(0.2), np.log(0.1), np.log(0.6)],
    ], [
        [np.log(0.1), np.log(0.2), np.log(0.1), np.log(0.6)],
        [np.log(0.1), np.log(0.1), np.log(0.2), np.log(0.6)],
        [np.log(0.3), np.log(0.1), np.log(0.3), np.log(0.3)],
        [np.log(0.1), np.log(0.2), np.log(0.1), np.log(0.6)],
        [np.log(0.1), np.log(0.2), np.log(0.1), np.log(0.6)],
    ]])

    actions = np.array([[1, 2, 0, 1], [0, 3, 3, 0]])

    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])

    probab_ratios = ppo.compute_probab_ratios(p_old, p_new, actions, mask)

    self.assertAllClose(
        np.array([
            [0.1 / 0.2, 0.1 / 0.4, 0.0, 0.0],
            [0.1 / 0.3, 0.6 / 0.4, 0.3 / 0.1, 0.0],
        ]), probab_ratios)

  def test_clipped_probab_ratios(self):
    probab_ratios = np.array([
        [1.5, 1.0, 0.5, 0.7],
        [2.5, 2.0, 0.1, 1.0],
    ])

    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])

    clipped_probab_ratios = ppo.clipped_probab_ratios(probab_ratios, mask, 0.1)

    self.assertAllClose(
        np.array([
            [1.1, 1.0, 0, 0],
            [1.1, 1.1, 0.9, 0],
        ]), clipped_probab_ratios)

  def test_clipped_objective(self):
    probab_ratios = np.array([
        [1.5, 2.0, 0.5, 0.7],
        [2.5, 2.0, 0.1, 1.0],
    ])

    advantages = np.array([
        [0.1, 0.1, 0.5, 0.7],
        [2.0, 2.0, 2.0, 2.0],
    ])

    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])

    epsilon = 0.1

    unused_clipped_probab_ratios = np.array([
        [1.1, 1.1, 0.9, 0.9],
        [1.1, 1.1, 0.9, 1.0],
    ])

    minimums = np.array([
        [1.1, 1.1, 0.5, 0.7],
        [1.1, 1.1, 0.1, 1.0],
    ])

    # advantages * minimums * mask
    objective = np.array([
        [0.11, 0.11, 0.0, 0.0],
        [2.2, 2.2, 0.2, 0.0],
    ])

    # Assert that we computed things correctly in this test.
    self.assertAllClose(advantages * mask * minimums, objective)

    self.assertAllClose(
        objective,
        ppo.clipped_objective(probab_ratios, advantages, mask, epsilon))

  def test_ppo_loss(self):
    self.rng_key, key1, key2, key3 = jax_random.split(self.rng_key, num=4)

    B, T, A, OBS = 2, 10, 2, (28, 28, 3)  # pylint: disable=invalid-name
    batch_observation_shape = (-1, -1) + OBS

    old_policy_params, _ = ppo.policy_net(key1, batch_observation_shape, A,
                                          [layers.Flatten(num_axis_to_keep=2)])

    new_policy_params, policy_apply = ppo.policy_net(
        key2,
        batch_observation_shape, A,
        [layers.Flatten(num_axis_to_keep=2)])

    value_params, value_apply = ppo.value_net(
        key3, batch_observation_shape, A,
        [layers.Flatten(num_axis_to_keep=2)])

    # Generate a batch of observations.

    observations = np.random.uniform(size=(B, T + 1) + OBS)
    actions = np.random.randint(0, A, size=(B, T))
    rewards = np.random.uniform(0, 1, size=(B, T))
    mask = np.ones_like(rewards)

    # Just test that this computes at all.
    _ = ppo.ppo_loss(policy_apply, new_policy_params, old_policy_params,
                     value_apply, value_params, observations, actions, rewards,
                     mask)


if __name__ == "__main__":
  test.main()
