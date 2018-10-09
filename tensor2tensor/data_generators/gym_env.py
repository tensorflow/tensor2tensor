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

import tensorflow as tf


Frame = collections.namedtuple(
    # Order of elements reflects time progression within a frame.
    "Frame", ("observation", "reward", "done", "action")
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
    self._current_frames = [None for _ in range(batch_size)]

    with tf.Graph().as_default():
      self._image_t = tf.placeholder(dtype=tf.uint8, shape=(None, None, None))
      self._encoded_image_t = tf.image.encode_png(self._image_t)
      self._encode_session = tf.Session()

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

  def _encode_observations(self, observations):
    """Encodes observations as PNG."""
    return [
        self._encode_session.run(
            self._encoded_image_t, feed_dict={self._image_t: observation}
        )
        for observation in observations
    ]

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
    encoded_obs = self._encode_observations(obs)
    for (rollout, frame, action) in zip(
        self._current_rollouts, self._current_frames, actions
    ):
      rollout.append(frame._replace(action=action))

    # ord = (observation, reward, done)
    self._current_frames = [
        Frame(*ord, action=None)
        for ord in zip(encoded_obs, rewards, dones)
    ]
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
    encoded_obs = self._encode_observations(new_obs)
    for (index, ob) in zip(indices, encoded_obs):
      frame = self._current_frames[index]
      if frame is not None and frame.done:
        rollout = self._current_rollouts[index]
        rollout.append(frame._replace(action=0))
        self.history.append(rollout)
        self._current_rollouts[index] = []
      self._current_frames[index] = Frame(
          observation=ob, reward=0, done=False, action=None
      )
    return new_obs

  def close(self):
    """Cleanups any resources.

    Can be overridden in derived classes.
    """
    self._encode_session.close()


class T2TGymEnv(T2TEnv):
  """Class representing a batch of Gym environments."""

  def __init__(self, envs):
    super(T2TGymEnv, self).__init__(len(envs))

    if not envs:
      raise ValueError("Must have at least one environment.")
    self._envs = envs

    self.observation_space = envs[0].observation_space
    if not all(env.observation_space == self.observation_space for env in envs):
      raise ValueError("All environments must use the same observation space.")

    self.action_space = envs[0].action_space
    if not all(env.action_space == self.action_space for env in envs):
      raise ValueError("All environments must use the same action space.")

  def __str__(self):
    return "T2TGymEnv(%s)" % ", ".join([str(env) for env in self._envs])

  def _preprocess_observations(self, obs):
    # TODO(lukaszkaiser): Implement.
    return obs

  def _preprocess_rewards(self, rewards):
    # TODO(lukaszkaiser): Implement.
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
