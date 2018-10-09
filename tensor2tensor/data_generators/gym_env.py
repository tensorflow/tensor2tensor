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

"""RL environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from PIL import Image
from gym.spaces import Box


Frame = collections.namedtuple(
    "Frame", ("observation", "action", "reward", "done")
)


class T2TEnv(object):
  """Abstract class representing a batch of environments.

  Attributes:
    history: List of finished rollouts, where rollout is a list of Frames.
    batch_size: Number of environments played simultaneously.
    observation_space: Gym observation space. Should be overridden in derived
      classes.
    action_space: Gym action space. Should be overridden in derived classes.

  Args:
    batch_size: Number of environments in a batch.
  """

  observation_space = None
  action_space = None

  def __init__(self, batch_size):
    self.clear_history()
    self.batch_size = batch_size
    self._current_rollouts = [[] for _ in range(batch_size)]
    self._current_observations = [None for _ in range(batch_size)]

  def __str__(self):
    """Returns a string representation of the environment for debug purposes."""
    raise NotImplementedError

  def clear_history(self):
    """Clears the rollout history."""
    self.history = []

  def _preprocess_observations(self, obs):
    """Transforms a batch of observations.

    Can be overridden in derived classes.

    Args:
      obs: A batch of observations.

    Returns:
      Transformed batch of observations.
    """
    return obs

  def _preprocess_rewards(self, rewards):
    """Transforms a batch of rewards.

    Can be overridden in derived classes.

    Args:
      rewards: A batch of rewards.

    Returns:
      Transformed batch of rewards.
    """
    return rewards

  def _step(self, actions):
    """Makes a step in all environments without recording history.

    Should be overridden in derived classes.

    Should not do any preprocessing of the observations and rewards; this
    should be done in _preprocess_*.

    Args:
      actions: Batch of actions.

    Returns:
      (obs, rewards, dones) - batches of observations, rewards and done flags
      respectively.
    """
    raise NotImplementedError

  def step(self, actions):
    """Makes a step in all environments.

    Does any preprocessing and records frames.

    Args:
      actions: Batch of actions.

    Returns:
      (obs, rewards, dones) - batches of observations, rewards and done flags
      respectively.
    """
    (obs, rewards, dones) = self._step(actions)
    obs = self._preprocess_observations(obs)
    rewards = self._preprocess_rewards(rewards)
    # oard = (observation, action, reward, done)
    for (rollout, oard) in zip(self._current_rollouts, zip(
        self._current_observations, actions, rewards, dones
    )):
      rollout.append(Frame(*oard))
    self._current_observations = obs
    return (obs, rewards, dones)

  def _reset(self, indices):
    """Resets environments at given indices without recording history.

    Args:
      indices: Indices of environments to reset.

    Returns:
      Batch of initial observations of reset environments.
    """
    raise NotImplementedError

  def reset(self, indices=None):
    """Resets environments at given indices.

    Does any preprocessing and adds finished rollouts to history.

    Args:
      indices: Indices of environments to reset.

    Returns:
      Batch of initial observations of reset environments.
    """
    if indices is None:
      indices = np.arange(self.batch_size)
    new_obs = self._reset(indices)
    new_obs = self._preprocess_observations(new_obs)
    for (index, ob) in zip(indices, new_obs):
      rollout = self._current_rollouts[index]
      if rollout and rollout[-1].done:
        self.history.append(rollout)
        self._current_rollouts[index] = []
      self._current_observations[index] = ob
    return new_obs

  def close(self):
    """Cleanups any resources.

    Can be overridden in derived classes.
    """
    pass


class T2TGymEnv(T2TEnv):
  """Class representing a batch of Gym environments."""

  def __init__(self, envs, clip_rewards=False, grayscale=False,
               resize_height_factor=1, resize_width_factor=1):
    super(T2TGymEnv, self).__init__(len(envs))
    self.clip_rewards = clip_rewards
    self.grayscale = grayscale
    self.resize_height_factor = resize_height_factor
    self.resize_width_factor = resize_width_factor
    if not envs:
      raise ValueError("Must have at least one environment.")
    self._envs = envs

    orig_observ_space = envs[0].observation_space
    if not all(env.observation_space == orig_observ_space
               for env in self._envs):
      raise ValueError("All environments must use the same observation space.")

    self.observation_space = self._derive_observation_space(orig_observ_space)

    self.action_space = envs[0].action_space
    if not all(env.action_space == self.action_space for env in self._envs):
      raise ValueError("All environments must use the same action space.")

  def _derive_observation_space(self, orig_observ_space):
    height, width, channels = orig_observ_space.shape
    if self.grayscale:
      channels = 1
    resized_height = height // self.resize_height_factor
    resized_width = width // self.resize_width_factor
    shape = (resized_height, resized_width, channels)
    return Box(low=orig_observ_space.low.min(),
               high=orig_observ_space.high.max(), shape=shape,
               dtype=orig_observ_space.dtype)

  def __str__(self):
    return "T2TGymEnv(%s)" % ", ".join([str(env) for env in self._envs])

  def _preprocess_observations(self, obs):
    assert obs.dtype == np.uint8, "Image.fromarray() requires np.uint8 dtype" \
                                  " to work as expected"
    preprocessed = list()
    for ob in obs:
      im = Image.fromarray(ob, mode='RGB')
      height, width = self.observation_space.shape[:2]
      im = im.resize(size=(width, height), resample=Image.BOX)
      if self.grayscale:
        im = im.convert(mode='L')
      preprocessed.append(np.array(im))
    return np.stack(preprocessed)

  def _preprocess_rewards(self, rewards):
    if self.clip_rewards:
      rewards = np.clip(rewards, -1, 1)
    return rewards

  def _step(self, actions):
    (obs, rewards, dones, _) = zip(*[
        env.step(action) for (env, action) in zip(self._envs, actions)
    ])
    return tuple(map(np.stack, (obs, rewards, dones)))

  def _reset(self, indices):
    return np.stack([self._envs[index].reset() for index in indices])

  def close(self):
    for env in self._envs:
      env.close()
