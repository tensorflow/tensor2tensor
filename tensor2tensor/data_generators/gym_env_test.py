# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

import tensorflow as tf


class TestEnv(gym.Env):
  """Test environment.

  Odd frames are "done".
  """

  action_space = Discrete(1)
  observation_space = Box(
      low=0, high=255, shape=(2, 6, 3), dtype=np.uint8
  )

  def __init__(self):
    self._counter = 0

  def _generate_ob(self):
    return np.zeros(
        self.observation_space.shape, self.observation_space.dtype
    )

  def step(self, action):
    done = self._counter % 2 == 1
    self._counter += 1
    reward = 5 if done else -5
    return (self._generate_ob(), reward, done, {})

  def reset(self):
    return self._generate_ob()


class GymEnvTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.out_dir = tf.test.get_temp_dir()
    shutil.rmtree(cls.out_dir)
    os.mkdir(cls.out_dir)

  def init_batch_and_play(self, env_lambda, n_steps=1, **kwargs):
    raw_envs = [env_lambda(), env_lambda()]
    env = gym_env.T2TGymEnv(raw_envs, **kwargs)
    obs = list()
    rewards = list()
    obs.append(env.reset())
    for _ in range(n_steps):
      step_obs, step_rewards, dones = env.step(actions=[0, 0])
      obs.append(step_obs)
      rewards.append(step_rewards)
      for (i, done) in enumerate(dones):
        if done:
          env.reset([i])
    return env, obs, rewards

  def test_generates(self):
    # This test needs base env which outputs done after two steps.
    env_lambda = TestEnv
    env, _, _ = self.init_batch_and_play(env_lambda, n_steps=20)
    env.generate_data(self.out_dir, tmp_dir=None)

    filenames = os.listdir(self.out_dir)
    self.assertTrue(filenames)
    path = os.path.join(self.out_dir, filenames[0])
    records = list(tf.python_io.tf_record_iterator(path))
    self.assertTrue(records)

  def test_clipping(self):
    # This test needs base env with rewards out of [-1,1] range.
    env_lambda = TestEnv
    # TODO(lukaszkaiser): turn clipping on by default after refactor.
    # _, _, rewards = self.init_batch_and_play(env_lambda, n_steps=2)
    # self.assertTrue(np.max(rewards) == 1)
    # self.assertTrue(np.min(rewards) == -1)

    _, _, unclipped_rewards = self.init_batch_and_play(env_lambda, n_steps=2)
    self.assertTrue(np.max(unclipped_rewards) > 1)
    self.assertTrue(np.min(unclipped_rewards) < -1)

  def test_resize(self):
    env_lambda = TestEnv
    orig_env = env_lambda()
    resize_height_factor = 2
    resize_width_factor = 3
    orig_height, orig_width = orig_env.observation_space.shape[:2]
    env, obs, _ = self.init_batch_and_play(
        env_lambda, n_steps=1,
        resize_height_factor=resize_height_factor,
        resize_width_factor=resize_width_factor)
    for obs_batch in obs:
      ob = obs_batch[0]
      self.assertEqual(ob.shape, env.observation_space.shape)
      height, width = ob.shape[:2]
      self.assertEqual(height, orig_height // resize_height_factor)
      self.assertEqual(width, orig_width // resize_width_factor)

  def assert_channels(self, env, obs, n_channels):
    self.assertEqual(env.observation_space.shape[2], n_channels)
    self.assertEqual(env.num_channels, n_channels)
    for obs_batch in obs:
      ob = obs_batch[0]
      self.assertEqual(ob.shape[2], n_channels)

  def test_channels(self):
    env_lambda = TestEnv
    env, obs, _ = self.init_batch_and_play(env_lambda, grayscale=True)
    self.assert_channels(env, obs, n_channels=1)

    env, obs, _ = self.init_batch_and_play(env_lambda, grayscale=False)
    self.assert_channels(env, obs, n_channels=3)


if __name__ == "__main__":
  tf.test.main()
