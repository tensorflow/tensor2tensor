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

"""Gym env tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import gym
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from tensor2tensor.data_generators import gym_env
from tensor2tensor.data_generators import problem
from tensor2tensor.rl.gym_utils import make_gym_env

import tensorflow.compat.v1 as tf


class TestEnv(gym.Env):
  """Test environment.

  Odd frames are "done".
  """

  action_space = Discrete(1)
  # TODO(afrozm): Gym's Box has a bug for uint8 type, which doesn't allow
  # sampling, send them a PR. Till that time let this be np.int64
  observation_space = Box(
      low=0, high=255, shape=(2, 6, 3), dtype=np.int64
  )

  def __init__(self):
    self._counter = 0

  def _generate_ob(self):
    return self.observation_space.sample()

  def step(self, action):
    done = self._counter % 2 == 1
    self._counter += 1
    reward = 5 if done else -5
    return (self._generate_ob(), reward, done, {})

  def reset(self):
    return self._generate_ob()

TEST_ENV_NAME = "T2TTestEnv-v1"

gym.envs.register(id=TEST_ENV_NAME, entry_point=TestEnv)


class GymEnvTest(tf.test.TestCase):

  splits = (problem.DatasetSplit.TRAIN, problem.DatasetSplit.EVAL)

  # TODO(koz4k): Tests for loading:
  # - loaded epoch is read-only
  # - partial write detection (should raise on loading)

  def setUp(self):
    self.out_dir = tf.test.get_temp_dir()
    shutil.rmtree(self.out_dir)
    os.mkdir(self.out_dir)
    np.random.seed(0)

  def init_batch_and_play(self, env_name, steps_per_epoch=1, epochs=(0,),
                          generate_data=False, batch_size=2, **kwargs):
    env = gym_env.T2TGymEnv(env_name, batch_size=batch_size, **kwargs)
    obs = []
    rewards = []
    num_dones = 0
    for epoch in epochs:
      env.start_new_epoch(epoch, self.out_dir)
      _, epoch_obs, epoch_rewards, epoch_num_dones = \
          self.play(env, steps_per_epoch)
      epoch_obs.append(env.reset())
      if generate_data:
        env.generate_data(self.out_dir)
      obs.extend(epoch_obs)
      rewards.extend(epoch_rewards)
      num_dones += epoch_num_dones
    return env, obs, rewards, num_dones

  def play(self, env, n_steps):
    obs = []
    rewards = []
    obs.append(env.reset())
    num_dones = 0
    for _ in range(n_steps):
      step_obs, step_rewards, dones = env.step(actions=[0, 0])
      obs.append(step_obs)
      rewards.append(step_rewards)
      for (i, done) in enumerate(dones):
        if done:
          env.reset([i])
          num_dones += 1
    return env, obs, rewards, num_dones

  def test_splits_dataset(self):
    env, _, _, _ = self.init_batch_and_play(
        TEST_ENV_NAME, steps_per_epoch=20, generate_data=True
    )

    for split in self.splits:
      self.assertTrue(env.current_epoch_rollouts(split))

  def test_split_preserves_number_of_rollouts(self):
    batch_size = 2
    env, _, _, num_dones = self.init_batch_and_play(
        TEST_ENV_NAME, steps_per_epoch=20, generate_data=True,
        batch_size=batch_size
    )

    num_rollouts_after_split = sum(
        len(env.current_epoch_rollouts(split)) for split in self.splits
    )
    # After the end of epoch all environments are reset, which increases number
    # of rollouts by batch size. Number of rollouts could be increased by one
    # in case a rollout is broken on a boundary between the dataset splits.
    self.assertGreaterEqual(num_rollouts_after_split, num_dones + batch_size)
    self.assertLessEqual(num_rollouts_after_split, num_dones + batch_size + 1)

  def test_split_preserves_number_of_frames(self):
    batch_size = 2
    env, _, _, num_dones = self.init_batch_and_play(
        TEST_ENV_NAME, steps_per_epoch=20, generate_data=True,
        batch_size=batch_size
    )

    num_frames = sum(
        len(rollout)
        for split in self.splits
        for rollout in env.current_epoch_rollouts(split)
    )
    # There are 3 frames in every rollout: the initial one and two returned by
    # step(). Additionally there are batch_size observations coming from final
    # reset at the end of epoch.
    self.assertEqual(num_frames, 3 * num_dones + batch_size)

  def test_generates_data(self):
    # This test needs base env which outputs done after two steps.
    self.init_batch_and_play(
        TEST_ENV_NAME, steps_per_epoch=20, generate_data=True
    )

    filenames = os.listdir(self.out_dir)
    self.assertTrue(filenames)
    for filename in filenames:
      path = os.path.join(self.out_dir, filename)
      records = list(tf.python_io.tf_record_iterator(path))
      self.assertTrue(records)

  def test_shards_per_epoch(self):
    def num_ending_with(filenames, suffix):
      return sum(
          1 for filename in filenames if filename.endswith(suffix)
      )

    env = gym_env.T2TGymEnv(TEST_ENV_NAME, batch_size=2)
    env.start_new_epoch(0, self.out_dir)
    self.play(env, n_steps=20)
    env.generate_data(self.out_dir)

    filenames = os.listdir(self.out_dir)
    num_shards_per_epoch = len(filenames)
    self.assertEqual(num_ending_with(filenames, ".0"), num_shards_per_epoch)

    env.start_new_epoch(1, self.out_dir)
    self.play(env, n_steps=20)
    env.generate_data(self.out_dir)

    filenames = os.listdir(self.out_dir)
    self.assertEqual(len(filenames), 2 * num_shards_per_epoch)
    for suffix in (".0", ".1"):
      self.assertEqual(num_ending_with(filenames, suffix), num_shards_per_epoch)

  def test_frame_numbers_are_continuous(self):
    env, _, _, _ = self.init_batch_and_play(
        TEST_ENV_NAME, steps_per_epoch=20, generate_data=True
    )

    frame_numbers = [
        tf.train.Example.FromString(
            record
        ).features.feature["frame_number"].int64_list.value[0]
        for (_, paths) in env.splits_and_paths(self.out_dir)
        for path in paths
        for record in tf.python_io.tf_record_iterator(path)
    ]
    last_frame_number = -1
    for frame_number in frame_numbers:
      # Every consecutive frame number should be either zero (first frame in
      # a new rollout) or one bigger than the last one (next frame in the same
      # rollout).
      if frame_number > 0:
        self.assertEqual(frame_number, last_frame_number + 1)
      last_frame_number = frame_number

  def test_clipping(self):
    _, _, rewards, _ = self.init_batch_and_play(TEST_ENV_NAME,
                                                steps_per_epoch=2)
    self.assertTrue(np.max(rewards) == 1)
    self.assertTrue(np.min(rewards) == -1)

  def test_resize(self):
    env_name = TEST_ENV_NAME
    orig_env = make_gym_env(env_name)
    resize_height_factor = 2
    resize_width_factor = 3
    orig_height, orig_width = orig_env.observation_space.shape[:2]
    env, obs, _, _ = self.init_batch_and_play(
        env_name, steps_per_epoch=1,
        resize_height_factor=resize_height_factor,
        resize_width_factor=resize_width_factor)
    for obs_batch in obs:
      ob = obs_batch[0]
      self.assertEqual(ob.shape, env.observation_space.shape)
      height, width = ob.shape[:2]
      self.assertEqual(height, orig_height // resize_height_factor)
      self.assertEqual(width, orig_width // resize_width_factor)

  def test_no_resize_option(self):
    env_name = TEST_ENV_NAME
    orig_env = make_gym_env(env_name)
    resize_height_factor = 2
    resize_width_factor = 3
    orig_height, orig_width = orig_env.observation_space.shape[:2]
    env, obs, _, _ = self.init_batch_and_play(
        env_name, steps_per_epoch=1,
        resize_height_factor=resize_height_factor,
        resize_width_factor=resize_width_factor,
        should_derive_observation_space=False)
    for obs_batch in obs:
      ob = obs_batch[0]
      self.assertEqual(ob.shape, env.observation_space.shape)
      height, width = ob.shape[:2]
      self.assertEqual(height, orig_height)
      self.assertEqual(width, orig_width)

  def assert_channels(self, env, obs, n_channels):
    self.assertEqual(env.observation_space.shape[2], n_channels)
    self.assertEqual(env.num_channels, n_channels)
    for obs_batch in obs:
      ob = obs_batch[0]
      self.assertEqual(ob.shape[2], n_channels)

  def test_channels(self):
    env_name = TEST_ENV_NAME
    env, obs, _, _ = self.init_batch_and_play(env_name, grayscale=True)
    self.assert_channels(env, obs, n_channels=1)

    env, obs, _, _ = self.init_batch_and_play(env_name, grayscale=False)
    self.assert_channels(env, obs, n_channels=3)

  def test_generating_and_loading_preserves_rollouts(self):
    env_name = TEST_ENV_NAME
    from_env = gym_env.T2TGymEnv(env_name, batch_size=1)
    from_env.start_new_epoch(0, self.out_dir)
    self.play(from_env, n_steps=20)
    from_env.generate_data(self.out_dir)

    to_env = gym_env.T2TGymEnv(env_name, batch_size=1)
    to_env.start_new_epoch(0, self.out_dir)

    self.assertEqual(
        from_env.current_epoch_rollouts(), to_env.current_epoch_rollouts()
    )

if __name__ == "__main__":
  tf.test.main()
