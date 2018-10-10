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
import math
import random

from gym.spaces import Box
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.utils import metrics

import tensorflow as tf


Frame = collections.namedtuple(
    # Order of elements reflects time progression within a frame.
    "Frame", ("observation", "reward", "unclipped_reward", "done", "action")
)


class T2TEnv(video_utils.VideoProblem):
  """Abstract class representing a batch of environments.

  Attributes:
    history: List of finished rollouts, where rollout is a list of Frames.
    batch_size: Number of environments played simultaneously.
    observation_space: Gym observation space. Should be overridden in derived
      classes.
    action_space: Gym action space. Should be overridden in derived classes.
    reward_range: Tuple (min, max) representing the range of rewards. Limits
      should be integer (discrete rewards).
    name: Problem name for generating filenames. Should be overridden in
      derived classes.

  Args:
    batch_size: Number of environments in a batch.
  """

  observation_space = None
  action_space = None
  reward_range = (-1, 1)
  name = None

  def __init__(self, batch_size):
    super(T2TEnv, self).__init__()

    self.clear_history()
    self.batch_size = batch_size
    self._current_rollouts = [[] for _ in range(batch_size)]
    self._current_frames = [None for _ in range(batch_size)]

    with tf.Graph().as_default() as tf_graph:
      self._tf_graph = tf_graph
      self._image_t = tf.placeholder(dtype=tf.uint8, shape=(None, None, None))
      self._encoded_image_t = tf.image.encode_png(self._image_t)
      self._session = tf.Session()

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

  def _encode_observations(self, observations):
    """Encodes observations as PNG."""
    return [
        self._session.run(
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
    (obs, unclipped_rewards, dones) = self._step(actions)
    obs = self._preprocess_observations(obs)
    (min_reward, max_reward) = self.reward_range
    rewards = np.around(np.clip(unclipped_rewards, min_reward, max_reward))
    encoded_obs = self._encode_observations(obs)
    for (rollout, frame, action) in zip(
        self._current_rollouts, self._current_frames, actions
    ):
      rollout.append(frame._replace(action=action))

    # orud = (observation, reward, unclipped_reward, done)
    self._current_frames = [
        Frame(*orud, action=None)
        for orud in zip(encoded_obs, rewards, unclipped_rewards, dones)
    ]
    # TODO(lukaszkaiser): changed unclipped_reward to reward once we've
    # removed the current setup with RewardClippingWrapper and so on.
    return (obs, unclipped_rewards, dones)

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
          observation=ob, reward=0, unclipped_reward=0, done=False, action=None
      )
    return new_obs

  def close(self):
    """Cleanups any resources.

    Can be overridden in derived classes.
    """
    self._session.close()

  @property
  def num_channels(self):
    """Number of color channels in each frame."""
    raise NotImplementedError

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ,
        metrics.Metrics.IMAGE_RMSE
    ]
    return eval_metrics

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    field_names = ("frame_number", "action", "reward", "done")
    data_fields = {
        name: tf.FixedLenFeature([1], tf.int64) for name in field_names
    }
    decoders = {
        name: tf.contrib.slim.tfexample_decoder.Tensor(tensor_key=name)
        for name in field_names
    }
    return (data_fields, decoders)

  @property
  def frame_height(self):
    return self.observation_space.shape[0]

  @property
  def frame_width(self):
    return self.observation_space.shape[1]

  @property
  def num_actions(self):
    return self.action_space.n

  @property
  def num_rewards(self):
    (min_reward, max_reward) = self.reward_range
    return max_reward - min_reward + 1

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    def make_modality(name):
      return {
          "{}s".format(name): ("video", 256),
          "{}_reward".format(name): ("symbol:weights_all", self.num_rewards),
          "{}_action".format(name): ("symbol:weights_all", self.num_actions)
      }
    p.input_modality = make_modality("input")
    p.target_modality = make_modality("target")
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def _generate_frames(self, rollouts):
    for rollout in rollouts:
      for (frame_number, frame) in enumerate(rollout):
        yield {
            "frame_number": [frame_number],
            "image/encoded": [frame.observation],
            "image/format": ["png"],
            "image/height": [self.frame_height],
            "image/width": [self.frame_width],
            "action": [int(frame.action)],
            "reward": [int(frame.reward - self.reward_range[0])],
            "done": [int(frame.done)]
        }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    """Saves the rollout history to disk."""
    # Suffle rollouts globally taking advantage of the fact that we have
    # everything in memory.
    shuffled_history = self.history[:]
    random.shuffle(shuffled_history)

    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    # We set shuffled=True as we don't want to shuffle on disk later.
    splits_and_paths = [
        (split["split"], path)
        for split in self.dataset_splits
        for path in filepath_fns[split["split"]](
            data_dir, split["shards"], shuffled=True
        )
    ]

    # Split entire rollouts into shards so that no rollout is broken on shard
    # boundary.
    shard_size = int(math.ceil(len(shuffled_history)) / len(splits_and_paths))
    for (i, (split, path)) in enumerate(splits_and_paths):
      rollouts = shuffled_history[i * shard_size : (i + 1) * shard_size]
      generator_utils.generate_files(
          self._generate_frames(rollouts), [path], cycle_every_n=float("inf")
      )


class T2TGymEnv(T2TEnv):
  """Class representing a batch of Gym environments."""

  name = "t2t_gym_env"

  def __init__(self, envs, grayscale=False,
               resize_height_factor=1, resize_width_factor=1):
    super(T2TGymEnv, self).__init__(len(envs))
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

    with self._tf_graph.as_default():
      self._resize = dict()
      orig_height, orig_width = orig_observ_space.shape[:2]
      self._img_batch_t = tf.placeholder(
          dtype=tf.uint8, shape=(None, orig_height, orig_width, 3))
      height, width = self.observation_space.shape[:2]
      resized = tf.image.resize_images(self._img_batch_t,
                                       [height, width],
                                       tf.image.ResizeMethod.AREA)
      if self.grayscale:
        resized = tf.image.rgb_to_grayscale(resized)
      self._resized_img_batch_t = resized

  @property
  def num_channels(self):
    return self.observation_space.shape[2]

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
    return self._session.run(self._resized_img_batch_t,
                             feed_dict={self._img_batch_t: obs})

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
