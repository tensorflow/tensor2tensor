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

"""Utilities for interacting with Gym classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf


class MaxAndSkipEnv(gym.Wrapper):
  """Same wrapper as in OpenAI baselines for comparability of results."""

  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame."""
    gym.Wrapper.__init__(self, env)
    observation_space = env.observation_space
    # Most recent raw observations (for max pooling across time steps).
    self._obs_buffer = np.zeros((2,) + observation_space.shape,
                                dtype=observation_space.dtype)
    self._skip = skip

  def __str__(self):
    return "MaxAndSkip<%s>" % str(self.env)

  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2: self._obs_buffer[0] = obs
      if i == self._skip - 1: self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame doesn't matter.
    max_frame = self._obs_buffer.max(axis=0)
    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


def make_gym_env(name, rl_env_max_episode_steps=-1, maxskip_env=False):
  """Create a gym env optionally with a time limit and maxskip wrapper.

  NOTE: The returned env may already be wrapped with TimeLimit!

  Args:
    name: `str` - base name of the gym env to make.
    rl_env_max_episode_steps: `int` or None - Using any value < 0 returns the
      env as-in, otherwise we impose the requested timelimit. Setting this to
      None returns a wrapped env that doesn't have a step limit.
    maxskip_env: whether to also use MaxAndSkip wrapper before time limit.

  Returns:
    An instance of `gym.Env` or `gym.wrappers.TimeLimit` with the requested
    step limit.
  """

  # rl_env_max_episode_steps is None or int.
  assert ((not rl_env_max_episode_steps) or
          isinstance(rl_env_max_episode_steps, int))

  env = gym.make(name)

  # If nothing to do, then return the env.
  if rl_env_max_episode_steps and rl_env_max_episode_steps < 0:
    if maxskip_env:
      if isinstance(env, gym.wrappers.TimeLimit):
        # Unwrap time limit and put it above MaxAndSkip for consistency.
        max_episode_steps = env._max_episode_steps  # pylint: disable=protected-access
        env = MaxAndSkipEnv(env.env)
        return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
      return MaxAndSkipEnv(env)
    return env

  # Sometimes (mostly?) the env is already wrapped in a TimeLimit wrapper, in
  # which case unwrap it and wrap with the proper time limit requested.
  if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

  if maxskip_env:
    env = MaxAndSkipEnv(env)

  return gym.wrappers.TimeLimit(env, max_episode_steps=rl_env_max_episode_steps)


def register_gym_env(class_entry_point, version="v0"):
  """Registers the class in Gym and returns the registered name and the env."""

  split_on_colon = class_entry_point.split(":")
  assert len(split_on_colon) == 2

  class_name = split_on_colon[1]
  # We have to add the version to conform to gym's API.
  env_name = "T2TEnv-{}-{}".format(class_name, version)
  gym.envs.register(id=env_name, entry_point=class_entry_point)

  tf.logging.info("Entry Point [%s] registered with id [%s]",
                  class_entry_point, env_name)

  return env_name, gym.make(env_name)
