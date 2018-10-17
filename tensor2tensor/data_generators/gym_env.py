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
import itertools

import gym
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


class _Noncopyable(object):

  def __init__(self, obj):
    self.obj = obj

  def __deepcopy__(self, memo):
    return self


def make_gym_env(name, timesteps_limit=-1):
  env = gym.make(name)
  if timesteps_limit != -1:
    # Replace TimeLimit Wrapper with one of proper time step limit.
    if isinstance(env, gym.wrappers.TimeLimit):
      env = env.env
    env = gym.wrappers.TimeLimit(env,
                                 max_episode_steps=timesteps_limit)
  return env


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

  def __init__(self, batch_size, *args, **kwargs):
    super(T2TEnv, self).__init__(*args, **kwargs)

    self.clear_history()
    self.batch_size = batch_size
    self._current_batch_frames = [None for _ in range(batch_size)]
    self._current_batch_rollouts = [[] for _ in range(batch_size)]
    self._current_epoch_rollouts = []
    self._rollouts_by_epoch_and_split = collections.OrderedDict()
    self.current_epoch = None
    with tf.Graph().as_default() as tf_graph:
      self._tf_graph = _Noncopyable(tf_graph)
      self._image_p = _Noncopyable(
          tf.placeholder(dtype=tf.uint8, shape=(None, None, None))
      )
      self._encoded_image_t = _Noncopyable(
          tf.image.encode_png(self._image_p.obj)
      )
      self._session = _Noncopyable(tf.Session())

  def __str__(self):
    """Returns a string representation of the environment for debug purposes."""
    raise NotImplementedError

  def clear_history(self):
    """Clears the rollout history."""
    self._rollouts_by_epoch_and_split = collections.OrderedDict()

  def start_new_epoch(self, epoch, load_data_dir):
    if not isinstance(epoch, int):
      raise ValueError("Epoch should be integer, got {}".format(epoch))
    if epoch in self._rollouts_by_epoch_and_split:
      raise ValueError("Epoch {} already registered".format(epoch))
    self.current_epoch = epoch
    self._current_epoch_rollouts = []
    self._rollouts_by_epoch_and_split[epoch] = collections.defaultdict(list)
    self._load_epoch_data(load_data_dir)

  def current_epoch_rollouts(self, split=None):
    rollouts_by_split = self._rollouts_by_epoch_and_split[self.current_epoch]
    if not rollouts_by_split:
      if split is not None:
        raise ValueError(
            "generate_data() should first be called in the current epoch"
        )
      else:
        return self._current_epoch_rollouts
    if split is not None:
      return rollouts_by_split[split]
    else:
      return [
          rollout
          for rollouts in rollouts_by_split.values()
          for rollout in rollouts
      ]

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
        self._session.obj.run(
            self._encoded_image_t.obj,
            feed_dict={self._image_p.obj: observation}
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

    Raises:
      ValueError: when the data for current epoch has already been loaded.
    """
    if self._rollouts_by_epoch_and_split[self.current_epoch]:
      raise ValueError(
          "Data for current epoch has already been loaded from disk."
      )
    (obs, unclipped_rewards, dones) = self._step(actions)
    obs = self._preprocess_observations(obs)
    (min_reward, max_reward) = self.reward_range
    rewards = np.around(np.clip(unclipped_rewards, min_reward, max_reward))
    unclipped_rewards = unclipped_rewards.astype(np.float64)
    encoded_obs = self._encode_observations(obs)
    for (rollout, frame, action) in zip(
        self._current_batch_rollouts, self._current_batch_frames, actions
    ):
      rollout.append(frame._replace(action=action))

    # orud = (observation, reward, unclipped_reward, done)
    self._current_batch_frames = [
        Frame(*orud, action=None)
        for orud in zip(encoded_obs, rewards, unclipped_rewards, dones)
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

    Does any preprocessing and adds rollouts to history.

    Args:
      indices: Indices of environments to reset.

    Returns:
      Batch of initial observations of reset environments.
    """
    if self.current_epoch is None:
      raise ValueError(
          "No current epoch. start_new_epoch() should first be called."
      )

    if indices is None:
      indices = np.arange(self.batch_size)
    new_obs = self._reset(indices)
    new_obs = self._preprocess_observations(new_obs)
    encoded_obs = self._encode_observations(new_obs)
    for (index, ob) in zip(indices, encoded_obs):
      frame = self._current_batch_frames[index]
      if frame is not None:
        rollout = self._current_batch_rollouts[index]
        rollout.append(frame._replace(action=0))
        self._current_epoch_rollouts.append(rollout)
        self._current_batch_rollouts[index] = []
      self._current_batch_frames[index] = Frame(
          observation=ob, reward=0, unclipped_reward=0, done=False, action=None
      )
    return new_obs

  def close(self):
    """Cleanups any resources.

    Can be overridden in derived classes.
    """
    self._session.obj.close()

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
  def only_keep_videos_from_0th_frame(self):
    return False

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
            "epoch": [self.current_epoch],
            "image/encoded": [frame.observation],
            "image/format": ["png"],
            "image/height": [self.frame_height],
            "image/width": [self.frame_width],
            "action": [int(frame.action)],
            "reward": [int(frame.reward - self.reward_range[0])],
            "unclipped_reward": [float(frame.unclipped_reward)],
            "done": [int(frame.done)]
        }

  @staticmethod
  def _calc_num_frames(rollouts):
    return sum(len(rollout) for rollout in rollouts)

  def _split_current_epoch(self):
    """Splits frames in the current epoch according to self.dataset_splits.

    Rollouts can be broken on shard boundary. This is desirable when we have
    few long rollouts and we want to make sure we have data in the dev set.
    """
    num_frames = self._calc_num_frames(self._current_epoch_rollouts)
    num_shards = sum(split["shards"] for split in self.dataset_splits)
    shard_size = num_frames // num_shards

    splits = self.dataset_splits
    num_saved_frames = 0
    split_index = 0
    split_begin_index = 0
    rollouts_by_split = collections.defaultdict(list)

    def split_size(split_index):
      return splits[split_index]["shards"] * shard_size

    for rollout in self._current_epoch_rollouts:
      num_saved_frames_current_rollout = 0
      # Split the rollout into chunks corresponding to dataset splits. In most
      # cases there should be only one chunk. On dataset split boundary there
      # will be two. If a rollout is longer then the size of a dataset split,
      # there might be more.
      while num_saved_frames_current_rollout < len(rollout):
        max_chunk_length = (
            split_begin_index + split_size(split_index) - num_saved_frames
        )
        if split_index == len(splits) - 1:
          # Put the remainder in the last split to preserve the ordering.
          max_chunk_length = len(rollout)
        rollout_chunk = rollout[
            num_saved_frames_current_rollout:
            (num_saved_frames_current_rollout + max_chunk_length)
        ]
        rollouts_by_split[splits[split_index]["split"]].append(rollout_chunk)
        num_saved_frames_current_rollout += len(rollout_chunk)
        num_saved_frames += len(rollout_chunk)

        if num_saved_frames == split_begin_index + split_size(split_index):
          split_begin_index += split_size(split_index)
          split_index = min(split_index + 1, len(splits) - 1)

    self._rollouts_by_epoch_and_split[self.current_epoch] = rollouts_by_split
    self._current_epoch_rollouts = []

  def splits_and_paths(self, data_dir):
    """List of pairs (split, paths) for the current epoch."""
    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    def append_epoch(paths):
      return [
          "{}.{}".format(path, self.current_epoch)
          for path in paths
      ]

    # We set shuffled=True as we don't want to shuffle on disk later.
    return [
        (split["split"], append_epoch(filepath_fns[split["split"]](
            data_dir, split["shards"], shuffled=True
        )))
        for split in self.dataset_splits
    ]

  def filepattern(self, data_dir, mode, shard=None, only_last=False):
    filepattern = super(T2TEnv, self).filepattern(
        data_dir, mode, shard
    )
    if only_last:
      filepattern += ".{}".format(self.current_epoch)
    return filepattern

  def generate_data(self, data_dir, tmp_dir=None, task_id=-1):
    """Saves the current epoch rollouts to disk, split into train/dev sets."""
    if not self._rollouts_by_epoch_and_split[self.current_epoch]:
      # Data not loaded from disk.
      self._split_current_epoch()

    rollouts_by_split = self._rollouts_by_epoch_and_split[self.current_epoch]
    splits_and_paths = self.splits_and_paths(data_dir)

    for (split, paths) in splits_and_paths:
      rollouts = rollouts_by_split[split]
      num_frames = self._calc_num_frames(rollouts)
      shard_size = num_frames // len(paths)

      frame_gen = self._generate_frames(rollouts)
      for (path_index, path) in enumerate(paths):
        limit = shard_size
        # Put the remainder in the last shard to preserve the ordering.
        if path_index == len(paths) - 1:
          limit = None
        generator_utils.generate_files(
            itertools.islice(frame_gen, limit), [path],
            cycle_every_n=float("inf")
        )

  def _load_epoch_data(self, data_dir):
    any_files_found = False
    all_files_found = True
    any_shard_empty = False

    for split, paths in self.splits_and_paths(data_dir):
      try:
        any_shard_empty |= self._load_epoch_split(split, paths)
        any_files_found = True
      except tf.errors.NotFoundError:
        all_files_found = False
    if any_shard_empty or (not all_files_found and any_files_found):
      raise ValueError("Some data is missing, the experiment might've been "
                       "interupted during generating data.")

  def _load_epoch_split(self, split, paths):
    epoch = self.current_epoch
    last_frame_number = -1
    any_shard_empty = False
    current_rollout = []

    for path in paths:
      this_shard_empty = True
      for example in tf.python_io.tf_record_iterator(path):
        this_shard_empty = False

        result = tf.train.Example.FromString(example)
        feature = result.features.feature

        def get_feature_value(key, list_name):
          return getattr(feature[key], list_name).value[0]  # pylint: disable=cell-var-from-loop

        fields = {
            key: get_feature_value(key, list_name)
            for (key, list_name) in [
                ("image/encoded", "bytes_list"), ("reward", "int64_list"),
                ("unclipped_reward", "float_list"), ("done", "int64_list"),
                ("action", "int64_list")
            ]
        }
        fields["reward"] += self.reward_range[0]
        fields["done"] = bool(fields["done"])
        fields["observation"] = fields["image/encoded"]
        del fields["image/encoded"]

        frame = Frame(**fields)
        frame_number = get_feature_value("frame_number", "int64_list")
        if frame_number == last_frame_number + 1:
          current_rollout.append(frame)
        else:
          self._rollouts_by_epoch_and_split[epoch][split].append(
              current_rollout)
          current_rollout = [frame]
        last_frame_number = frame_number

      any_shard_empty |= this_shard_empty

    self._rollouts_by_epoch_and_split[epoch][split].append(
        current_rollout
    )
    return any_shard_empty


class T2TGymEnv(T2TEnv):
  """Class representing a batch of Gym environments.

  Do not register it, instead create subclass with hardcoded __init__
  arguments and register this subclass.
  """

  def __init__(self, base_env_name, batch_size, grayscale=False,
               resize_height_factor=1, resize_width_factor=1,
               base_env_timesteps_limit=-1, *args, **kwargs):
    super(T2TGymEnv, self).__init__(batch_size, *args, **kwargs)
    self.grayscale = grayscale
    self.resize_height_factor = resize_height_factor
    self.resize_width_factor = resize_width_factor
    if not self.name:
      # Set problem name if not registered.
      self.name = "t2t_gym_env_{}".format(base_env_name)

    self._envs = [make_gym_env(base_env_name, base_env_timesteps_limit)
                  for _ in range(self.batch_size)]

    orig_observ_space = self._envs[0].observation_space
    if not all(env.observation_space == orig_observ_space
               for env in self._envs):
      raise ValueError("All environments must use the same observation space.")

    self.observation_space = self._derive_observation_space(orig_observ_space)

    self.action_space = self._envs[0].action_space
    if not all(env.action_space == self.action_space for env in self._envs):
      raise ValueError("All environments must use the same action space.")

    with self._tf_graph.obj.as_default():
      self._resize = dict()
      orig_height, orig_width = orig_observ_space.shape[:2]
      self._img_batch_t = _Noncopyable(tf.placeholder(
          dtype=tf.uint8, shape=(None, orig_height, orig_width, 3)))
      height, width = self.observation_space.shape[:2]
      resized = tf.image.resize_images(self._img_batch_t.obj,
                                       [height, width],
                                       tf.image.ResizeMethod.AREA)
      resized = tf.cast(resized, tf.as_dtype(self.observation_space.dtype))
      if self.grayscale:
        resized = tf.image.rgb_to_grayscale(resized)
      self._resized_img_batch_t = _Noncopyable(resized)

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
    return self._session.obj.run(self._resized_img_batch_t.obj,
                                 feed_dict={self._img_batch_t.obj: obs})

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
