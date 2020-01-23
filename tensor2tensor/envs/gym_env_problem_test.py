# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Tests for tensor2tensor.envs.gym_env_problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gym
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.envs import env_problem
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import gym_env_problem
from tensor2tensor.layers import modalities
import tensorflow.compat.v1 as tf


class GymEnvProblemTest(tf.test.TestCase):

  def setUp(self):
    self.tmp_dir = os.path.join(tf.test.get_temp_dir(), "tmp_dir")
    tf.gfile.MakeDirs(self.tmp_dir)

  def tearDown(self):
    tf.gfile.DeleteRecursively(self.tmp_dir)

  def test_setup(self):
    ep = gym_env_problem.GymEnvProblem(
        base_env_name="CartPole-v0", batch_size=5)
    # Checks that environments were created and they are `batch_size` in number.
    ep.assert_common_preconditions()

    # Expectations on the observation space.
    observation_space = ep.observation_space
    self.assertIsInstance(observation_space, Box)
    self.assertEqual(observation_space.shape, (4,))
    self.assertEqual(observation_space.dtype, np.float32)

    # Expectations on the action space.
    action_space = ep.action_space
    self.assertTrue(isinstance(action_space, Discrete))
    self.assertEqual(action_space.shape, ())
    self.assertEqual(action_space.dtype, np.int64)
    self.assertEqual(ep.num_actions, 2)

    # Reward range is infinite here.
    self.assertFalse(ep.is_reward_range_finite)

  def test_reward_range(self):
    # Passing reward_range=None means take the reward range of the underlying
    # environment as the reward range.
    ep = gym_env_problem.GymEnvProblem(
        base_env_name="FrozenLake-v0", batch_size=5, reward_range=None)
    ep.assert_common_preconditions()

    # Assert reward range is finite here.
    self.assertTrue(ep.is_reward_range_finite)

    # Assert that it is as expected of the underlying environment, since reward_
    self.assertEqual(0, ep.reward_range[0])
    self.assertEqual(1, ep.reward_range[1])

  def test_default_processed_rewards_discrete(self):
    # This differs in the above because it has a Tuple observation space.
    ep = gym_env_problem.GymEnvProblem(
        base_env_name="KellyCoinflip-v0", batch_size=5, reward_range=None)
    ep.assert_common_preconditions()

    # Assert reward range is finite here.
    self.assertTrue(ep.is_reward_range_finite)

    # Assert that it is as expected of the underlying environment.
    reward_range = ep.reward_range
    self.assertEqual(0, reward_range[0])

    # Google's version of Gym has maxWealth, vs max_wealth externally.
    max_wealth = getattr(ep._envs[0], "maxWealth",
                         getattr(ep._envs[0], "max_wealth", None))
    self.assertIsNotNone(max_wealth)
    self.assertEqual(max_wealth, reward_range[1])

    # Check that the processed rewards are discrete.
    self.assertTrue(ep.is_processed_rewards_discrete)

    # Assert on the number of rewards.
    self.assertEqual(ep.num_rewards, reward_range[1] - reward_range[0] + 1)

  def test_interaction_with_env(self):
    batch_size = 5
    reward_range = (-1, 1)
    ep = gym_env_problem.GymEnvProblem(
        base_env_name="KellyCoinflip-v0",
        batch_size=batch_size,
        reward_range=reward_range)

    # Resets all environments.
    ep.reset()

    # Let's play a few steps.
    nsteps = 100
    num_trajectories_completed = 0
    num_timesteps_completed = 0
    # If batch_done_at_step[i] = j then it means that i^th env last got done at
    # step = j.
    batch_done_at_step = np.full(batch_size, -1)
    for i in range(nsteps):
      # Sample batch_size actions from the action space and stack them (since
      # that is the expected type).
      actions = np.stack([ep.action_space.sample() for _ in range(batch_size)])

      _, _, dones, _ = ep.step(actions)

      # Do the book-keeping on number of trajectories completed and expect that
      # it matches ep's completed number.

      num_done = sum(dones)
      num_trajectories_completed += num_done

      self.assertEqual(num_trajectories_completed,
                       len(ep.trajectories.completed_trajectories))

      # Get the indices where we are done ...
      done_indices = env_problem_utils.done_indices(dones)

      # ... and reset those.
      ep.reset(indices=done_indices)

      # If nothing got done, go on to the next step.
      if done_indices.size == 0:
        # i.e. this is an empty array.
        continue

      # See when these indices were last done and calculate how many time-steps
      # each one took to get done.
      num_timesteps_completed += sum(i + 1 - batch_done_at_step[done_indices])
      batch_done_at_step[done_indices] = i

      # This should also match the number of time-steps completed given by ep.
      num_timesteps_completed_ep = sum(
          ct.num_time_steps for ct in ep.trajectories.completed_trajectories)
      self.assertEqual(num_timesteps_completed, num_timesteps_completed_ep)

    # Reset the trajectories.
    ep.trajectories.reset_batch_trajectories()
    self.assertEqual(0, len(ep.trajectories.completed_trajectories))

  def read_tfrecord_dataset(self, filenames, ep):
    # Read the dataset at `filenames` into a tf.data.Dataset and returns the
    # number of time-steps (just the number of records in the dataset) and the
    # number of trajectories.

    last_timestep = -1
    num_time_steps = 0
    num_trajectories = 0
    for ex in generator_utils.tfrecord_iterator(
        filenames, example_spec=ep.example_reading_spec()[0]):
      num_time_steps += 1
      this_timestep = ex[env_problem.TIMESTEP_FIELD][0]
      if 1 + last_timestep != this_timestep:
        num_trajectories += 1
        self.assertEqual(0, this_timestep)
      last_timestep = this_timestep
    num_trajectories += 1

    return num_trajectories, num_time_steps

  def play_env(self,
               env=None,
               nsteps=100,
               base_env_name=None,
               batch_size=5,
               reward_range=None):
    """Creates `GymEnvProblem` with the given arguments and plays it randomly.

    Args:
      env: optional env.
      nsteps: plays the env randomly for nsteps.
      base_env_name: passed to GymEnvProblem's init.
      batch_size: passed to GymEnvProblem's init.
      reward_range: passed to GymEnvProblem's init.

    Returns:
      tuple of gym_env_problem, number of trajectories done,
      number of trajectories done in the last step.
    """

    if env is None:
      env = gym_env_problem.GymEnvProblem(
          base_env_name=base_env_name,
          batch_size=batch_size,
          reward_range=reward_range)
      # Usually done by a registered subclass, we do this manually in the test.
      env.name = base_env_name

    # Reset all environments.
    env.reset()

    # Play for some steps to generate data.
    num_dones = 0
    num_dones_in_last_step = 0
    for _ in range(nsteps):
      # Sample actions.
      actions = np.stack([env.action_space.sample() for _ in range(batch_size)])
      # Step through it.
      _, _, dones, _ = env.step(actions)
      # Get the indices where we are done ...
      done_indices = env_problem_utils.done_indices(dones)
      # ... and reset those.
      env.reset(indices=done_indices)
      # count the number of dones we got, in this step and overall.
      num_dones_in_last_step = sum(dones)
      num_dones += num_dones_in_last_step

    return env, num_dones, num_dones_in_last_step

  def test_generate_data(self):
    base_env_name = "CartPole-v0"
    batch_size = 5
    reward_range = (-1, 1)
    nsteps = 100
    ep, num_dones, num_dones_in_last_step = self.play_env(
        base_env_name=base_env_name,
        batch_size=batch_size,
        reward_range=reward_range,
        nsteps=nsteps)

    # This is because every num_dones starts a new trajectory, and a further
    # batch_size are active at the last step when we call generate_data, but
    # the ones that got done in the last step (these have only one time-step in
    # their trajectory) will be skipped.
    expected_num_trajectories = num_dones + batch_size - num_dones_in_last_step

    # Similar logic as above, nsteps * batch_size overall `step` calls are made.
    expected_num_time_steps = (
        nsteps * batch_size) + num_dones + batch_size - num_dones_in_last_step

    # Dump the completed data to disk.
    ep.generate_data(self.tmp_dir, self.tmp_dir)

    # Read the written files and assert on the number of time steps.
    training_filenames = ep.training_filepaths(
        self.tmp_dir, ep.num_shards[problem.DatasetSplit.TRAIN], True)
    dev_filenames = ep.dev_filepaths(
        self.tmp_dir, ep.num_shards[problem.DatasetSplit.EVAL], True)

    training_trajectories, training_timesteps = self.read_tfrecord_dataset(
        training_filenames, ep)
    dev_trajectories, dev_timesteps = self.read_tfrecord_dataset(
        dev_filenames, ep)

    # This tests what we wrote on disk matches with what we computed.
    self.assertEqual(expected_num_time_steps,
                     training_timesteps + dev_timesteps)
    self.assertEqual(expected_num_trajectories,
                     training_trajectories + dev_trajectories)

  def test_problem_dataset_works(self):

    # We need to derive this class to set the required methods.
    class TestEnv(gym_env_problem.GymEnvProblem):
      name = "TestEnv"

      @property
      def input_modality(self):
        return modalities.ModalityType.REAL_L2_LOSS

      @property
      def input_vocab_size(self):
        return None

      @property
      def target_modality(self):
        return modalities.ModalityType.SYMBOL_WEIGHTS_ALL

      @property
      def target_vocab_size(self):
        return 2

      @property
      def action_modality(self):
        return modalities.ModalityType.SYMBOL_WEIGHTS_ALL

      @property
      def reward_modality(self):
        return modalities.ModalityType.SYMBOL_WEIGHTS_ALL

    base_env_name = "CartPole-v0"
    batch_size = 5
    reward_range = (-1, 1)

    env = TestEnv(
        base_env_name=base_env_name,
        batch_size=batch_size,
        reward_range=reward_range)

    nsteps = 100
    ep, _, _ = self.play_env(env=env, nsteps=nsteps)

    # Dump the completed data to disk.
    ep.generate_data(self.tmp_dir, self.tmp_dir)

    # Read the actual files and count the trajectories and time-steps.
    dev_filenames = ep.dev_filepaths(
        self.tmp_dir, ep.num_shards[problem.DatasetSplit.EVAL], True)
    dev_trajectories, dev_timesteps = self.read_tfrecord_dataset(
        dev_filenames, ep)

    # Count them using a tf.data.Dataset.
    dev_dataset = ep.dataset(tf.estimator.ModeKeys.EVAL, data_dir=self.tmp_dir)

    last_timestep = -1
    dev_timesteps_ds = 0
    dev_trajectories_ds = 0
    iterator = dev_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as session:
      while True:
        try:
          tf_example_dict = session.run(next_element)

          # We have a time-step.
          dev_timesteps_ds += 1

          this_timestep = tf_example_dict[env_problem.TIMESTEP_FIELD][
              0]  # [0] since every value in tf_example_dict is an array/list.
          if 1 + last_timestep != this_timestep:
            dev_trajectories_ds += 1
            self.assertEqual(0, this_timestep)
          last_timestep = this_timestep
        except tf.errors.OutOfRangeError:
          dev_trajectories_ds += 1
          break

    # Make sure that they agree.
    self.assertEqual(dev_trajectories, dev_trajectories_ds)
    self.assertEqual(dev_timesteps, dev_timesteps_ds)

  def test_resets_properly(self):
    base_env_name = "CartPole-v0"
    batch_size = 5
    reward_range = (-1, 1)
    nsteps = 100

    env = gym_env_problem.GymEnvProblem(
        base_env_name=base_env_name,
        batch_size=batch_size,
        reward_range=reward_range)
    env.name = base_env_name

    num_dones = 0
    while num_dones == 0:
      env, num_dones, _ = self.play_env(env=env,
                                        nsteps=nsteps,
                                        batch_size=batch_size,
                                        reward_range=reward_range)

    # Some completed trajectories have been generated.
    self.assertGreater(env.trajectories.num_completed_trajectories, 0)

    # This should clear the env completely of any state.
    env.reset()

    # Assert that there aren't any completed trajectories in the env now.
    self.assertEqual(env.trajectories.num_completed_trajectories, 0)

  def test_per_env_kwargs(self):

    # Creating a dummy class where we specify the action at which the env
    # returns done.
    class TestPerEnvKwargsEnv(gym.Env):
      """Test environment with the `done action` specified."""

      action_space = Discrete(3)
      observation_space = Box(low=-1.0, high=1.0, shape=())

      def __init__(self, done_action=0):
        self._done_action = done_action

      def _generate_ob(self):
        return self.observation_space.sample()

      def step(self, action):
        done = self._done_action == action
        reward = 1 if done else 0
        return (self._generate_ob(), reward, done, {})

      def reset(self):
        return self._generate_ob()

    # Registering it with gym.
    test_env_name = "TestPerEnvKwargsEnv-v0"
    gym.envs.register(id=test_env_name, entry_point=TestPerEnvKwargsEnv)

    # Creating a batch of those with different done actions.
    base_env_name = test_env_name
    batch_size = 2
    reward_range = (-1, 1)
    per_env_kwargs = [{"done_action": 1}, {"done_action": 2}]

    env = gym_env_problem.GymEnvProblem(
        base_env_name=base_env_name,
        batch_size=batch_size,
        reward_range=reward_range,
        per_env_kwargs=per_env_kwargs)

    _ = env.reset()

    # Finally querying the done actions.

    _, _, d, _ = env.step(np.array([0, 0]))
    self.assertFalse(d[0])
    self.assertFalse(d[1])

    _, _, d, _ = env.step(np.array([1, 2]))
    self.assertTrue(d[0])
    self.assertTrue(d[1])

if __name__ == "__main__":
  tf.test.main()
