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

"""Tests for tensor2tensor.rl.gym_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from tensor2tensor.rl import gym_utils
import tensorflow as tf


class GymUtilsTest(tf.test.TestCase):

  # Just make an environment and expect to get one.
  def test_making_simple_env(self):
    env = gym_utils.make_gym_env("CartPole-v0")
    self.assertTrue(isinstance(env, gym.Env))

  # Make a time-wrapped environment and expect to get one.
  def test_making_timewrapped_env(self):
    env = gym_utils.make_gym_env("CartPole-v0", rl_env_max_episode_steps=1000)
    self.assertTrue(isinstance(env, gym.Env))
    self.assertTrue(isinstance(env, gym.wrappers.TimeLimit))
    self.assertEquals(1000, env._max_episode_steps)

  # Make a time-wrapped environment with unlimited limit.
  def test_unlimited_env(self):
    env = gym_utils.make_gym_env("CartPole-v0", rl_env_max_episode_steps=None)
    self.assertTrue(isinstance(env, gym.Env))
    self.assertTrue(isinstance(env, gym.wrappers.TimeLimit))
    self.assertTrue(env._max_episode_steps is None)


if __name__ == "__main__":
  tf.test.main()
