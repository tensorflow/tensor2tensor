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
from PIL import Image
import tensorflow as tf


class StickyActionEnv(gym.Wrapper):
  """Based on openai/atari-reset implementation."""

  def __init__(self, env, p=0.25):
    gym.Wrapper.__init__(self, env)
    self.p = p
    self.last_action = 0

  def step(self, action):
    if np.random.uniform() < self.p:
      action = self.last_action
    self.last_action = action
    obs, reward, done, info = self.env.step(action)
    return obs, reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class MaxAndSkipEnv(gym.Wrapper):
  """Same wrapper as in OpenAI baselines for comparability of results."""

  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame."""
    gym.Wrapper.__init__(self, env)
    observation_space = env.observation_space
    # Most recent raw observations (for max pooling across time steps).
    self._obs_buffer = np.zeros(
        (2,) + observation_space.shape, dtype=observation_space.dtype)
    self._skip = skip

  def __str__(self):
    return "MaxAndSkip<%s>" % str(self.env)

  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame doesn't matter.
    max_frame = self._obs_buffer.max(axis=0)
    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class RenderedEnv(gym.Wrapper):
  """Simple Env wrapper to override observations with rendered rgb values."""

  def __init__(self, env, mode="rgb_array", low=0, high=255, resize_to=None):
    gym.Wrapper.__init__(self, env)
    # Get a sample frame to correctly set observation space
    self.mode = mode
    sample_frame = self.render(mode=self.mode)
    assert sample_frame is not None
    self.should_resize = False
    if resize_to is None:
      self.observation_space = gym.spaces.Box(
          low=low,
          high=high,
          shape=sample_frame.shape,
          dtype=sample_frame.dtype)
    else:
      assert len(resize_to) == 2
      self.should_resize = True
      self.observation_space = gym.spaces.Box(
          low=low,
          high=high,
          shape=list(resize_to) + list(sample_frame.shape[-1:]),
          dtype=sample_frame.dtype)

  def step(self, action):
    _, reward, done, info = self.env.step(action)
    obs = self.env.render(mode=self.mode)
    if self.should_resize:
      img = Image.fromarray(obs)
      img = img.resize(
          self.observation_space.shape[:-1], resample=Image.ANTIALIAS)
      obs = np.array(img)
    return obs, reward, done, info

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    return self.env.render(mode=self.mode)


def remove_time_limit_wrapper(env):
  """Removes top level TimeLimit Wrapper.

  Removes TimeLimit Wrapper from top level if exists, throws error if any other
  TimeLimit Wrapper is present in stack.

  Args:
    env: environment

  Returns:
    the env with removed time limit wrapper.
  """
  if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env
  env_ = env
  while isinstance(env_, gym.Wrapper):
    if isinstance(env_, gym.wrappers.TimeLimit):
      raise ValueError("Can remove only top-level TimeLimit gym.Wrapper.")
    env_ = env_.env
  return env


def gym_env_wrapper(env, rl_env_max_episode_steps, maxskip_env, rendered_env,
                    rendered_env_resize_to, sticky_actions):
  """Wraps a gym environment. see make_gym_env for details."""
  # rl_env_max_episode_steps is None or int.
  assert ((not rl_env_max_episode_steps) or
          isinstance(rl_env_max_episode_steps, int))

  wrap_with_time_limit = ((not rl_env_max_episode_steps) or
                          rl_env_max_episode_steps >= 0)

  if wrap_with_time_limit:
    env = remove_time_limit_wrapper(env)

  if sticky_actions:
    env = StickyActionEnv(env)

  if maxskip_env:
    env = MaxAndSkipEnv(env)  # pylint: disable=redefined-variable-type

  if rendered_env:
    env = RenderedEnv(env, resize_to=rendered_env_resize_to)

  if wrap_with_time_limit:
    env = gym.wrappers.TimeLimit(
        env, max_episode_steps=rl_env_max_episode_steps)
  return env


def make_gym_env(name,
                 rl_env_max_episode_steps=-1,
                 maxskip_env=False,
                 rendered_env=False,
                 rendered_env_resize_to=None,
                 sticky_actions=False):
  """Create a gym env optionally with a time limit and maxskip wrapper.

  NOTE: The returned env may already be wrapped with TimeLimit!

  Args:
    name: `str` - base name of the gym env to make.
    rl_env_max_episode_steps: `int` or None - Using any value < 0 returns the
      env as-in, otherwise we impose the requested timelimit. Setting this to
      None returns a wrapped env that doesn't have a step limit.
    maxskip_env: whether to also use MaxAndSkip wrapper before time limit.
    rendered_env: whether to force render for observations. Use this for
      environments that are not natively rendering the scene for observations.
    rendered_env_resize_to: a list of [height, width] to change the original
      resolution of the native environment render.
    sticky_actions: whether to use sticky_actions before MaxAndSkip wrapper.

  Returns:
    An instance of `gym.Env` or `gym.Wrapper`.
  """
  env = gym.make(name)
  return gym_env_wrapper(env, rl_env_max_episode_steps, maxskip_env,
                         rendered_env, rendered_env_resize_to, sticky_actions)


def register_gym_env(class_entry_point, version="v0", kwargs=None):
  """Registers the class in Gym and returns the registered name and the env."""

  split_on_colon = class_entry_point.split(":")
  assert len(split_on_colon) == 2

  class_name = split_on_colon[1]
  # We have to add the version to conform to gym's API.
  env_name = "T2TEnv-{}-{}".format(class_name, version)
  gym.envs.register(id=env_name, entry_point=class_entry_point, kwargs=kwargs)

  tf.logging.info("Entry Point [%s] registered with id [%s]", class_entry_point,
                  env_name)

  return env_name, gym.make(env_name)
