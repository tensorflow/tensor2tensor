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

"""Utilities for interacting with Gym classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import logging
import gym
import gym.wrappers
import numpy as np
from PIL import Image


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


class ActionDiscretizeWrapper(gym.ActionWrapper):
  """Wraps an environment with continuous actions and discretizes them.

  This is a simplified adaptation of ActionDiscretizeWrapper
  from tf_agents.
  """

  def __init__(self, env, num_actions):
    """Constructs a wrapper for discretizing the action space.

    Args:
      env: environment to wrap.
      num_actions: A np.array of the same shape as the environment's
        action_spec. Elements in the array specify the number of actions to
        discretize to for each dimension.

    Raises:
      ValueError: IF the action_spec shape and the limits shape are not equal.
    """

    if not isinstance(env.action_space, gym.spaces.box.Box):
      raise ValueError(
          "The action space is {}, but gym.spaces.box.Box is expected".format(
              env.action_space))

    gym.Wrapper.__init__(self, env)

    # We convert a scalar num_actions to array [num_actions, num_actions, ...]
    self._num_actions = np.broadcast_to(num_actions, env.action_space.shape)

    if env.action_space.shape != self._num_actions.shape:
      raise ValueError("Spec {} and limit shape do not match. Got {}".format(
          env.action_space.shape, self._num_actions.shape))
    self.action_space = gym.spaces.MultiDiscrete(nvec=self._num_actions)
    self._action_map = self._discretize_env(env)

  def _discretize_env(self, env):
    """Generates a discrete bounded spec and a linspace for the given limits.

    Args:
      env: An array to discretize.

    Returns:
      Tuple with the discrete_spec along with a list of lists mapping actions.
    Raises:
      ValueError: If not all limits value are >=2 or maximum or minimum of boxes
      is equal to +- infinity.
    """
    if not np.all(self._num_actions >= 2):
      raise ValueError("num_actions should all be at least size 2.")

    if (math.isinf(np.min(env.action_space.low)) or
        math.isinf(np.max(env.action_space.high))):
      raise ValueError(
          """Minimum of boxes is {} and maximum of boxes is {},
          but we expect that finite values are provided.""".
          format(np.min(env.action_space.low),
                 np.max(env.action_space.high)))

    limits = np.broadcast_to(self._num_actions,
                             env.action_space.shape)
    minimum = np.broadcast_to(np.min(env.action_space.low),
                              env.action_space.shape)
    maximum = np.broadcast_to(np.max(env.action_space.high),
                              env.action_space.shape)

    action_map = [
        np.linspace(env_min, env_max, num=n_actions)
        for env_min, env_max, n_actions in zip(
            np.nditer(minimum), np.nditer(maximum), np.nditer(limits))
    ]

    return action_map

  def _map_actions(self, action):
    """Maps the given discrete action to the corresponding continuous action.

    Args:
      action: Discrete action to map.

    Returns:
      Numpy array with the mapped continuous actions.
    Raises:
      ValueError: If the given action's shpe does not match the action_spec
      shape.
    """
    action = np.asarray(action)
    if action.shape != self.action_space.shape:
      raise ValueError(
          "Received action with incorrect shape. Got {}, expected {}".format(
              action.shape, self.action_space.shape))

    mapped_action = [self._action_map[i][a]
                     for i, a in enumerate(action.flatten())]
    return np.reshape(mapped_action, newshape=action.shape)

  def action(self, action):
    """Steps the environment while remapping the actions.

    Args:
      action: Action to take.

    Returns:
      The next time_step from the environment.
    """
    return self._map_actions(action)

  def reverse_action(self, action):
    raise NotImplementedError


class RenderedEnv(gym.Wrapper):
  """Simple Env wrapper to override observations with rendered rgb values."""

  def __init__(self,
               env,
               mode="rgb_array",
               low=0,
               high=255,
               resize_to=None,
               output_dtype=None):
    gym.Wrapper.__init__(self, env)
    # Get a sample frame to correctly set observation space
    self.mode = mode
    sample_frame = self.render(mode=self.mode)
    assert sample_frame is not None
    self.should_resize = False
    self.output_dtype = output_dtype
    if resize_to is None:
      self.observation_space = gym.spaces.Box(
          low=low,
          high=high,
          shape=sample_frame.shape,
          dtype=sample_frame.dtype)
    else:
      assert len(resize_to) == 2
      self.should_resize = True
      num_channels = sample_frame.shape[-1]
      self.observation_space = gym.spaces.Box(
          low=low,
          high=high,
          shape=list(resize_to) + [num_channels],
          dtype=sample_frame.dtype)

  def _maybe_resize(self, obs):
    if not self.should_resize:
      return obs
    height, width = self.observation_space.shape[:2]
    img = Image.fromarray(obs)
    img = img.resize([width, height], resample=Image.ANTIALIAS)
    if self.output_dtype is None:
      return np.array(img)
    return np.array(img).astype(self.output_dtype)

  def step(self, action):
    _, reward, done, info = self.env.step(action)
    obs = self._maybe_resize(self.env.render(mode=self.mode))
    return obs, reward, done, info

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    obs = self._maybe_resize(self.env.render(mode=self.mode))
    return obs


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
                    rendered_env_resize_to, sticky_actions, output_dtype,
                    num_actions):
  """Wraps a gym environment. see make_gym_env for details."""
  # rl_env_max_episode_steps is None or int.
  assert ((not rl_env_max_episode_steps) or
          isinstance(rl_env_max_episode_steps, int))

  wrap_with_time_limit = ((not rl_env_max_episode_steps) or
                          rl_env_max_episode_steps >= 0)

  if wrap_with_time_limit:
    env = remove_time_limit_wrapper(env)

  if num_actions is not None:
    logging.log_first_n(
        logging.INFO, "Number of discretized actions: %d", 1, num_actions)
    env = ActionDiscretizeWrapper(env, num_actions=num_actions)

  if sticky_actions:
    env = StickyActionEnv(env)

  if maxskip_env:
    env = MaxAndSkipEnv(env)  # pylint: disable=redefined-variable-type

  if rendered_env:
    env = RenderedEnv(
        env, resize_to=rendered_env_resize_to, output_dtype=output_dtype)

  if wrap_with_time_limit and rl_env_max_episode_steps is not None:
    env = gym.wrappers.TimeLimit(
        env, max_episode_steps=rl_env_max_episode_steps)
  return env


def make_gym_env(name,
                 rl_env_max_episode_steps=-1,
                 maxskip_env=False,
                 rendered_env=False,
                 rendered_env_resize_to=None,
                 sticky_actions=False,
                 output_dtype=None,
                 num_actions=None):
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
    output_dtype: numpy datatype that we want the observation to be in, if None
      this defaults to the env's observation dtype. Useful for TPUs since they
      don't support uint8 which is a default observation type for a lot of envs.
    num_actions: None if we do not need discretization and the number of
      discrete actions per continuous action.

  Returns:
    An instance of `gym.Env` or `gym.Wrapper`.
  """
  env = gym.make(name)
  return gym_env_wrapper(env, rl_env_max_episode_steps, maxskip_env,
                         rendered_env, rendered_env_resize_to, sticky_actions,
                         output_dtype, num_actions)


def register_gym_env(class_entry_point, version="v0", kwargs=None):
  """Registers the class in Gym and returns the registered name and the env."""

  split_on_colon = class_entry_point.split(":")
  assert len(split_on_colon) == 2

  class_name = split_on_colon[1]
  # We have to add the version to conform to gym's API.
  env_name = "T2TEnv-{}-{}".format(class_name, version)
  gym.envs.register(id=env_name, entry_point=class_entry_point, kwargs=kwargs)

  logging.info(
      "Entry Point [%s] registered with id [%s]", class_entry_point, env_name)

  return env_name, gym.make(env_name)
