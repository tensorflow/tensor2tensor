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

"""Tests for tensor2tensor.rl.gym_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import gym
from gym import spaces
import numpy as np
from tensor2tensor.rl import gym_utils
import tensorflow.compat.v1 as tf


class SimpleEnv(gym.Env):
  """A simple environment with a 3x3 observation space, is done on action=1."""

  def __init__(self):
    self.reward_range = (-1.0, 1.0)
    self.action_space = spaces.Discrete(2)
    self.observation_space = spaces.Box(low=0, high=255, shape=(3, 3))

  def reset(self):
    return self.observation_space.low

  def step(self, action):
    if action == 0:
      return self.reset(), -1.0, False, {}
    else:
      return self.observation_space.high, +1.0, True, {}

  def render(self, mode="human"):
    del mode  # Unused
    return np.zeros([640, 480, 3], np.uint8)


class SimpleContinuousActionsEnv(gym.Env):
  """A simple environment with a 3x3 observation space, is done on action=1."""

  def __init__(self, dimensions):
    self.reward_range = (-1.0, 1.0)
    self.action_space = spaces.Box(low=-1, high=1, shape=(dimensions,))
    self.observation_space = spaces.Box(low=0, high=255, shape=(3, 3))

  def reset(self):
    return self.observation_space.low

  def step(self, action):
    if action == 0:
      return self.reset(), -1.0, False, {}
    else:
      return self.observation_space.high, +1.0, True, {}

  def render(self, mode="human"):
    del mode  # Unused
    return np.zeros([640, 480, 3], np.uint8)


class EnvWithOptions(SimpleEnv):
  """A simple env that takes arguments on init."""

  def __init__(self, done_action=0):
    super(EnvWithOptions, self).__init__()
    self.action_space = spaces.Discrete(3)
    self._done_action = done_action

  def step(self, action):
    if action == self._done_action:
      return self.observation_space.high, +1.0, True, {}
    return self.reset(), -1.0, False, {}


class GymUtilsTest(tf.test.TestCase):

  # Just make an environment and expect to get one.
  def test_making_simple_env(self):
    env = gym_utils.make_gym_env("CartPole-v0")
    self.assertIsInstance(env, gym.Env)

  # Make a time-wrapped environment and expect to get one.
  def test_making_timewrapped_env(self):
    env = gym_utils.make_gym_env("CartPole-v0", rl_env_max_episode_steps=1000)
    self.assertIsInstance(env, gym.Env)
    self.assertIsInstance(env, gym.wrappers.TimeLimit)
    self.assertEqual(1000, env._max_episode_steps)

  # Make an instance of the environment without a TimeLimit
  def test_unlimited_env(self):
    env = gym_utils.make_gym_env("CartPole-v0", rl_env_max_episode_steps=None)
    self.assertIsInstance(env, gym.Env)
    self.assertNotIsInstance(env, gym.wrappers.TimeLimit)

  def test_rendered_env(self):
    env = gym_utils.RenderedEnv(SimpleEnv(), resize_to=(64, 12))
    obs, _, _, _ = env.step(1)
    self.assertTrue(np.allclose(np.zeros([64, 12, 3], np.uint8), obs))

    env = gym_utils.RenderedEnv(SimpleEnv(), resize_to=(64, 12),
                                output_dtype=np.float32)
    obs, _, _, _ = env.step(1)
    self.assertTrue(np.allclose(np.zeros([64, 12, 3], np.float32), obs))

  def test_rendered_env_continuous_1d(self):
    env = gym_utils.RenderedEnv(
        SimpleContinuousActionsEnv(dimensions=1),
        resize_to=(64, 12))
    obs, _, _, _ = env.step(0.5)
    self.assertTrue(np.allclose(np.zeros([64, 12, 3], np.uint8), obs))

    env = gym_utils.RenderedEnv(
        SimpleContinuousActionsEnv(dimensions=1),
        resize_to=(64, 12),
        output_dtype=np.float32)
    obs, _, _, _ = env.step(1)
    self.assertTrue(np.allclose(np.zeros([64, 12, 3], np.float32), obs))

  def test_rendered_env_continuous_2d(self):
    env = gym_utils.RenderedEnv(
        SimpleContinuousActionsEnv(dimensions=2),
        resize_to=(64, 12))
    obs, _, _, _ = env.step(0.5)
    self.assertTrue(np.allclose(np.zeros([64, 12, 3], np.uint8), obs))

    env = gym_utils.RenderedEnv(
        SimpleContinuousActionsEnv(dimensions=2),
        resize_to=(64, 12),
        output_dtype=np.float32)
    obs, _, _, _ = env.step(1)
    self.assertTrue(np.allclose(np.zeros([64, 12, 3], np.float32), obs))

  def test_correct_number_of_discrete_actions_1d(self):
    """The env should become discrete whenever we pass num_action."""
    env_discrete = gym_utils.ActionDiscretizeWrapper(
        gym_utils.RenderedEnv(SimpleContinuousActionsEnv(dimensions=1)),
        num_actions=4)

    expected_action_space = gym.spaces.MultiDiscrete([4,])
    self.assertEqual(env_discrete.action_space, expected_action_space)

  def test_correct_number_of_discrete_actions_2d(self):
    env_discrete = gym_utils.ActionDiscretizeWrapper(
        gym_utils.RenderedEnv(SimpleContinuousActionsEnv(dimensions=2)),
        num_actions=4)

    expected_action_space = gym.spaces.MultiDiscrete([4, 4])
    self.assertEqual(env_discrete.action_space, expected_action_space)

  def test_action_mapping_1d(self):
    """Testing discretization with a mock environment.

    In the mock call we get access to the argument of the
    SimpleContinuousActionsEnv.step method which we check against
    precomputed values of continuous actions.
    """
    num_actions = 4

    with unittest.mock.patch.object(
        gym_utils.RenderedEnv, "step", autospec=True) as mock_step_method:
      env = gym_utils.RenderedEnv(SimpleContinuousActionsEnv(dimensions=1))
      expected_continuous_actions = np.linspace(
          np.min(env.action_space.low),
          np.min(env.action_space.high),
          num=num_actions).flatten()

      env_discrete = gym_utils.ActionDiscretizeWrapper(env, num_actions)
      for discrete_action in range(num_actions):
        env_discrete.step([discrete_action])
        mock_step_method.assert_called_with(
            unittest.mock.ANY,
            expected_continuous_actions[discrete_action])

  def test_action_mapping_2d(self):
    num_actions = 8

    def expected_continuous_actions(discrete_action):
      if discrete_action == [0, 0]:
        return np.array([-1, -1])
      elif discrete_action == [0, 3]:
        return np.array([-1, -0.14285714])
      elif discrete_action == [4, 4]:
        return np.array([0.14285714, 0.14285714])
      elif discrete_action == [7, 7]:
        return np.array([1, 1])

    discrete_actions = [[0, 0], [0, 3], [4, 4], [7, 7]]

    with unittest.mock.patch.object(
        gym_utils.RenderedEnv, "step", autospec=True) as mock_step_method:
      env = gym_utils.RenderedEnv(SimpleContinuousActionsEnv(dimensions=2))

      env_discrete = gym_utils.ActionDiscretizeWrapper(env, num_actions)
      for discrete_action in discrete_actions:
        env_discrete.step(discrete_action)
        mock_args, _ = mock_step_method.call_args
        np.testing.assert_array_almost_equal(
            mock_args[1], expected_continuous_actions(discrete_action))

  def test_gym_registration(self):
    reg_id, env = gym_utils.register_gym_env(
        "tensor2tensor.rl.gym_utils_test:SimpleEnv")

    self.assertEqual("T2TEnv-SimpleEnv-v0", reg_id)

    # Most basic check.
    self.assertIsInstance(env, gym.Env)

    # Just make sure we got the same environment.
    self.assertTrue(
        np.allclose(env.reset(), np.zeros(shape=(3, 3), dtype=np.uint8)))

    _, _, done, _ = env.step(1)
    self.assertTrue(done)

  def test_gym_registration_continuous(self):
    reg_id, env = gym_utils.register_gym_env(
        "tensor2tensor.rl.gym_utils_test:SimpleContinuousActionsEnv",
        kwargs={"dimensions": 2})

    self.assertEqual("T2TEnv-SimpleContinuousActionsEnv-v0", reg_id)

    # Most basic check.
    self.assertIsInstance(env, gym.Env)

    # Just make sure we got the same environment.
    self.assertTrue(
        np.allclose(env.reset(), np.zeros(shape=(3, 3), dtype=np.uint8)))

    _, _, done, _ = env.step(1)
    self.assertTrue(done)

  def test_gym_registration_with_kwargs(self):
    reg_id, env = gym_utils.register_gym_env(
        "tensor2tensor.rl.gym_utils_test:EnvWithOptions",
        kwargs={"done_action": 2})

    self.assertEqual("T2TEnv-EnvWithOptions-v0", reg_id)

    # Obligatory reset.
    env.reset()

    # Make sure that on action = 0, 1 we are not done, but on 2 we are.
    _, _, done, _ = env.step(0)
    self.assertFalse(done)

    _, _, done, _ = env.step(1)
    self.assertFalse(done)

    _, _, done, _ = env.step(2)
    self.assertTrue(done)

    # Now lets try to change the env -- note we have to change the version.
    reg_id, env = gym_utils.register_gym_env(
        "tensor2tensor.rl.gym_utils_test:EnvWithOptions",
        version="v1",
        kwargs={"done_action": 1})

    self.assertEqual("T2TEnv-EnvWithOptions-v1", reg_id)

    # Obligatory reset.
    env.reset()

    # Make sure that on action = 0, 2 we are not done, but on 1 we are.
    _, _, done, _ = env.step(0)
    self.assertFalse(done)

    _, _, done, _ = env.step(2)
    self.assertFalse(done)

    _, _, done, _ = env.step(1)
    self.assertTrue(done)

if __name__ == "__main__":
  tf.test.main()
