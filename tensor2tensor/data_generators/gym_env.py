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

"""RL environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import random

from gym.spaces import Box
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import modalities
from tensor2tensor.rl import gym_utils
from tensor2tensor.utils import contrib
from tensor2tensor.utils import metrics
from tensor2tensor.utils import misc_utils
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


Frame = collections.namedtuple(
    # Order of elements reflects time progression within a frame.
    "Frame", ("observation", "reward", "unclipped_reward", "done", "action")
)


# pylint: disable=g-complex-comprehension
class Observation(object):
  """Encoded observations.

  Args:
    data: Encoded observation.
    decode_fn: Function for decoding observation.
  """

  def __init__(self, data, decode_fn):
    self.data = data
    self._decode = decode_fn

  def __eq__(self, other):
    """Equality comparison based on encoded data."""
    if isinstance(other, Observation):
      return self.data == other.data
    else:
      return False

  def __ne__(self, other):
    """For consistency with __eq__."""
    return not self == other

  def decode(self):
    """Decode the observation."""
    return self._decode(self.data)


class _Noncopyable(object):

  def __init__(self, obj):
    self.obj = obj

  def __deepcopy__(self, memo):
    return self


class EnvSimulationProblem(video_utils.VideoProblem):
  """Base Problem class for use with world models.

  Attributes:
    action_space: Gym action space. Should be overridden in derived classes.
    reward_range: Tuple (min, max) representing the range of rewards. Limits
      should be integer (discrete rewards).
  """

  action_space = None
  reward_range = (-1, 1)

  @property
  def num_actions(self):
    return self.action_space.n

  @property
  def num_rewards(self):
    (min_reward, max_reward) = self.reward_range
    return max_reward - min_reward + 1

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {
        "inputs": modalities.ModalityType.VIDEO,
        "input_reward": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "input_action": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "targets": modalities.ModalityType.VIDEO,
        "target_reward": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "target_action": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
    }
    p.vocab_size = {
        "inputs": 256,
        "input_reward": self.num_rewards,
        "input_action": self.num_actions,
        "targets": 256,
        "target_reward": self.num_rewards,
        "target_action": self.num_actions,
    }
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE


class T2TEnv(EnvSimulationProblem):
  """Abstract class representing a batch of environments.

  Attributes:
    history: List of finished rollouts, where rollout is a list of Frames.
    batch_size: Number of environments played simultaneously.
    observation_space: Gym observation space. Should be overridden in derived
      classes.
    name: Problem name for generating filenames. Should be overridden in
      derived classes.

  Args:
    batch_size: Number of environments in a batch.
    store_rollouts: Whether to store collected rollouts in memory and later on
      disk. Defaults to True.
  """

  observation_space = None
  name = None

  def __init__(self, batch_size, *args, **kwargs):
    self._store_rollouts = kwargs.pop("store_rollouts", True)

    super(T2TEnv, self).__init__(*args, **kwargs)

    self.batch_size = batch_size
    self._rollouts_by_epoch_and_split = collections.OrderedDict()
    self.current_epoch = None
    self._should_preprocess_on_reset = True
    with tf.Graph().as_default() as tf_graph:
      self._tf_graph = _Noncopyable(tf_graph)
      self._decoded_image_p = _Noncopyable(
          tf.placeholder(dtype=tf.uint8, shape=(None, None, None))
      )
      self._encoded_image_t = _Noncopyable(
          tf.image.encode_png(self._decoded_image_p.obj)
      )
      self._encoded_image_p = _Noncopyable(tf.placeholder(tf.string))
      self._decoded_image_t = _Noncopyable(
          tf.image.decode_png(self._encoded_image_p.obj)
      )
      self._session = _Noncopyable(tf.Session())

  def __str__(self):
    """Returns a string representation of the environment for debug purposes."""
    raise NotImplementedError

  def start_new_epoch(self, epoch, load_data_dir=None):
    if not isinstance(epoch, int):
      raise ValueError("Epoch should be integer, got {}".format(epoch))
    if epoch in self._rollouts_by_epoch_and_split:
      raise ValueError("Epoch {} already registered".format(epoch))
    self.current_epoch = epoch
    self._current_epoch_rollouts = []
    self._rollouts_by_epoch_and_split[epoch] = collections.defaultdict(list)
    self._current_batch_frames = [None for _ in range(self.batch_size)]
    self._current_batch_rollouts = [[] for _ in range(self.batch_size)]
    if load_data_dir is not None:
      self._load_epoch_data(load_data_dir)

  def current_epoch_rollouts(self, split=None, minimal_rollout_frames=0):
    # TODO(kc): order of rollouts (by splits) is a bit uncontrolled
    # (rollouts_by_split.values() reads dict values), is it a problem?
    rollouts_by_split = self._rollouts_by_epoch_and_split[self.current_epoch]
    if not rollouts_by_split:
      if split is not None:
        raise ValueError(
            "Data is not splitted into train/dev/test. If data created by "
            "environment interaction (NOT loaded from disk) you should call "
            "generate_data() first. Note that generate_data() will write to "
            "disk and can corrupt your experiment data."
        )
      else:
        rollouts = self._current_epoch_rollouts
    else:
      if split is not None:
        rollouts = rollouts_by_split[split]
      else:
        rollouts = [
            rollout
            for rollouts in rollouts_by_split.values()
            for rollout in rollouts
        ]
    return [rollout for rollout in rollouts
            if len(rollout) >= minimal_rollout_frames]

  def _preprocess_observations(self, obs):
    """Transforms a batch of observations.

    Can be overridden in derived classes.

    Args:
      obs: A batch of observations.

    Returns:
      Transformed batch of observations.
    """
    return obs

  def _decode_png(self, encoded_observation):
    """Decodes a single observation from PNG."""
    return self._session.obj.run(
        self._decoded_image_t.obj,
        feed_dict={self._encoded_image_p.obj: encoded_observation}
    )

  def _encode_observations(self, observations):
    """Encodes observations as PNG."""
    return [
        Observation(
            self._session.obj.run(
                self._encoded_image_t.obj,
                feed_dict={self._decoded_image_p.obj: observation}
            ),
            self._decode_png
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
    if self._store_rollouts and \
        self._rollouts_by_epoch_and_split[self.current_epoch]:
      raise ValueError(
          "Data for current epoch has already been loaded from disk."
      )
    (obs, unclipped_rewards, dones) = self._step(actions)
    obs = self._preprocess_observations(obs)
    (min_reward, max_reward) = self.reward_range
    rewards = np.around(np.clip(unclipped_rewards, min_reward, max_reward))
    if self._store_rollouts:
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

    Raises:
      ValueError: when there's no current epoch.
    """
    if self._store_rollouts and self.current_epoch is None:
      raise ValueError(
          "No current epoch. start_new_epoch() should first be called."
      )

    if indices is None:
      indices = np.arange(self.batch_size)
    new_obs = self._reset(indices)
    if self._should_preprocess_on_reset:
      new_obs = self._preprocess_observations(new_obs)
    if self._store_rollouts:
      encoded_obs = self._encode_observations(new_obs)
      for (index, ob) in zip(indices, encoded_obs):
        frame = self._current_batch_frames[index]
        if frame is not None:
          rollout = self._current_batch_rollouts[index]
          rollout.append(frame._replace(action=0))
          self._current_epoch_rollouts.append(rollout)
          self._current_batch_rollouts[index] = []
        self._current_batch_frames[index] = Frame(
            observation=ob, reward=0, unclipped_reward=0, done=False,
            action=None
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
        name: contrib.slim().tfexample_decoder.Tensor(tensor_key=name)
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

  def _generate_frames(self, rollouts):
    for rollout in rollouts:
      for (frame_number, frame) in enumerate(rollout):
        yield {
            "frame_number": [frame_number],
            "epoch": [self.current_epoch],
            "image/encoded": [frame.observation.data],
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
        fields["observation"] = Observation(
            fields["image/encoded"], self._decode_png
        )
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

  noop_action = 0

  def __init__(self, base_env_name=None, batch_size=1, grayscale=False,
               resize_height_factor=2, resize_width_factor=2,
               rl_env_max_episode_steps=-1, max_num_noops=0,
               maxskip_envs=False, sticky_actions=False,
               should_derive_observation_space=True,
               **kwargs):
    if base_env_name is None:
      base_env_name = self.base_env_name
    self._base_env_name = base_env_name
    super(T2TGymEnv, self).__init__(batch_size, **kwargs)
    # TODO(afrozm): Find a proper way of doing this. Refactor or cleanup.
    self.should_derive_observation_space = should_derive_observation_space
    self.grayscale = grayscale
    self.resize_height_factor = resize_height_factor
    self.resize_width_factor = resize_width_factor
    self.rl_env_max_episode_steps = rl_env_max_episode_steps
    self.maxskip_envs = maxskip_envs
    self.sticky_actions = sticky_actions
    self._initial_state = None
    self._initial_frames = None
    if not self.name:
      # Set problem name if not registered.
      self.name = "Gym%s" % base_env_name

    self._envs = [
        gym_utils.make_gym_env(
            base_env_name, rl_env_max_episode_steps=rl_env_max_episode_steps,
            maxskip_env=maxskip_envs, sticky_actions=sticky_actions)
        for _ in range(self.batch_size)]

    # max_num_noops works only with atari envs.
    if max_num_noops > 0:
      assert self._envs[0].unwrapped.get_action_meanings()[
          self.noop_action
      ] == "NOOP"
    self.max_num_noops = max_num_noops

    orig_observ_space = self._envs[0].observation_space
    if not all(env.observation_space == orig_observ_space
               for env in self._envs):
      raise ValueError("All environments must use the same observation space.")

    self.observation_space = orig_observ_space
    if self.should_derive_observation_space:
      self.observation_space = self._derive_observation_space(orig_observ_space)

    self.action_space = self._envs[0].action_space
    if not all(env.action_space == self.action_space for env in self._envs):
      raise ValueError("All environments must use the same action space.")

    if self.should_derive_observation_space:
      with self._tf_graph.obj.as_default():
        self._resize = {}
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

  # TODO(afrozm): Find a place for this. Till then use self._envs[0]'s hparams.
  def hparams(self, defaults, unused_model_hparams):
    if hasattr(self._envs[0], "hparams"):
      tf.logging.info("Retuning the env's hparams from T2TGymEnv.")
      return self._envs[0].hparams(defaults, unused_model_hparams)

    # Otherwise just call the super-class' hparams.
    tf.logging.info("Retuning the T2TGymEnv's superclass' hparams.")
    super(T2TGymEnv, self).hparams(defaults, unused_model_hparams)

  def new_like(self, **kwargs):
    env_kwargs = {
        "base_env_name": self.base_env_name,
        "batch_size": self.batch_size,
        "grayscale": self.grayscale,
        "resize_height_factor": self.resize_height_factor,
        "resize_width_factor": self.resize_width_factor,
        "rl_env_max_episode_steps": self.rl_env_max_episode_steps,
        "max_num_noops": self.max_num_noops,
        "maxskip_envs": self.maxskip_envs,
    }
    env_kwargs.update(kwargs)
    return T2TGymEnv(**env_kwargs)

  @property
  def base_env_name(self):
    return self._base_env_name

  @property
  def num_channels(self):
    return self.observation_space.shape[2]

  # TODO(afrozm): Why is this separated out from _preprocess_observations?
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

  def _encode_observations(self, observations):
    if not self.should_derive_observation_space:
      return observations
    return super(T2TGymEnv, self)._encode_observations(observations)

  def _preprocess_observations(self, observations):
    # TODO(afrozm): Clean this up.
    if not self.should_derive_observation_space:
      return observations
    return self._session.obj.run(
        self._resized_img_batch_t.obj,
        feed_dict={self._img_batch_t.obj: observations})

  @property
  def state(self):
    """Gets the current state."""
    return [env.unwrapped.clone_full_state() for env in self._envs]

  def set_initial_state(self, initial_state, initial_frames):
    """Sets the state that will be used on next reset."""
    self._initial_state = initial_state
    self._initial_frames = initial_frames[:, -1, ...]
    self._should_preprocess_on_reset = False

  def _step(self, actions):
    (obs, rewards, dones, _) = zip(*[
        env.step(action) for (env, action) in zip(self._envs, actions)
    ])
    return tuple(map(np.stack, (obs, rewards, dones)))

  def _reset(self, indices):
    def reset_with_initial_state(env, index):
      """Resets environment taking self._initial_state into account."""
      obs = env.reset()
      if self._initial_state is None:
        return obs
      else:
        env.unwrapped.restore_full_state(self._initial_state[index])
        return self._initial_frames[index, ...]

    def reset_with_noops(env, index):
      """Resets environment and applies random number of NOOP actions on it."""
      obs = reset_with_initial_state(env, index)
      try:
        num_noops = random.randint(1, self.max_num_noops)
      except ValueError:
        num_noops = 0

      for _ in range(num_noops):
        (obs, _, done, _) = env.step(self.noop_action)
        if done:
          obs = reset_with_initial_state(env, index)

      return obs

    return np.stack([
        reset_with_noops(self._envs[index], index) for index in indices
    ])

  def close(self):
    for env in self._envs:
      env.close()


class DummyWorldModelProblem(EnvSimulationProblem):
  """Dummy Problem for world model prediction."""

  def __init__(self, action_space, reward_range, frame_height, frame_width):
    super(DummyWorldModelProblem, self).__init__()
    self.action_space = action_space
    self.reward_range = reward_range
    self._frame_height = frame_height
    self._frame_width = frame_width

  @property
  def frame_height(self):
    """Height of each frame."""
    return self._frame_height

  @property
  def frame_width(self):
    """Width of each frame."""
    return self._frame_width


# Atari registration.

# Game list from our list of ROMs
# Removed because XDeterministic-v4 did not exist:
# * adventure
# * defender
# * kaboom
ATARI_GAMES = [
    "air_raid", "alien", "amidar", "assault", "asterix", "asteroids",
    "atlantis", "bank_heist", "battle_zone", "beam_rider", "berzerk", "bowling",
    "boxing", "breakout", "carnival", "centipede", "chopper_command",
    "crazy_climber", "demon_attack", "double_dunk", "elevator_action", "enduro",
    "fishing_derby", "freeway", "frostbite", "gopher", "gravitar", "hero",
    "ice_hockey", "jamesbond", "journey_escape", "kangaroo", "krull",
    "kung_fu_master", "montezuma_revenge", "ms_pacman", "name_this_game",
    "phoenix", "pitfall", "pong", "pooyan", "private_eye", "qbert", "riverraid",
    "road_runner", "robotank", "seaquest", "skiing", "solaris",
    "space_invaders", "star_gunner", "tennis", "time_pilot", "tutankham",
    "up_n_down", "venture", "video_pinball", "wizard_of_wor", "yars_revenge",
    "zaxxon"
]

# List from paper:
# https://arxiv.org/pdf/1805.11593.pdf
# plus frostbite.
ATARI_GAMES_WITH_HUMAN_SCORE = [
    "alien", "amidar", "assault", "asterix", "asteroids",
    "atlantis", "bank_heist", "battle_zone", "beam_rider", "bowling",
    "boxing", "breakout", "chopper_command",
    "crazy_climber", "demon_attack", "double_dunk", "enduro",
    "fishing_derby", "freeway", "frostbite", "gopher", "gravitar", "hero",
    "ice_hockey", "jamesbond", "kangaroo", "krull",
    "kung_fu_master", "montezuma_revenge", "ms_pacman", "name_this_game",
    "pitfall", "pong", "private_eye", "qbert", "riverraid",
    "road_runner", "seaquest", "solaris",
    "up_n_down", "video_pinball", "yars_revenge",
]


# Blacklist a few games where it makes little sense to run on for now.
ATARI_GAMES_WITH_HUMAN_SCORE_NICE = [
    g for g in ATARI_GAMES_WITH_HUMAN_SCORE if g not in [
        "solaris", "pitfall", "montezuma_revenge", "enduro",
        "video_pinball", "double_dunk"]]


ATARI_WHITELIST_GAMES = [
    "amidar",
    "bank_heist",
    "berzerk",
    "boxing",
    "crazy_climber",
    "freeway",
    "frostbite",
    "gopher",
    "kung_fu_master",
    "ms_pacman",
    "pong",
    "qbert",
    "seaquest",
]


# Games on which model-free does better than model-based at this point.
ATARI_CURIOUS_GAMES = [
    "bank_heist",
    "boxing",
    "enduro",
    "kangaroo",
    "road_runner",
    "up_n_down",
]


# Games on which based should work.
ATARI_DEBUG_GAMES = [
    "crazy_climber",
    "freeway",
    "pong",
]


# Different ATARI game modes in OpenAI Gym. Full list here:
# https://github.com/openai/gym/blob/master/gym/envs/__init__.py
ATARI_GAME_MODES = [
    "Deterministic-v0",  # 0.25 repeat action probability, 4 frame skip.
    "Deterministic-v4",  # 0.00 repeat action probability, 4 frame skip.
    "NoFrameskip-v0",    # 0.25 repeat action probability, 1 frame skip.
    "NoFrameskip-v4",    # 0.00 repeat action probability, 1 frame skip.
    "-v0",               # 0.25 repeat action probability, (2 to 5) frame skip.
    "-v4"                # 0.00 repeat action probability, (2 to 5) frame skip.
]


def register_game(game_name, game_mode="NoFrameskip-v4"):
  """Create and register problems for the game.

  Args:
    game_name: str, one of the games in ATARI_GAMES, e.g. "bank_heist".
    game_mode: the frame skip and sticky keys config.

  Raises:
    ValueError: if game_name or game_mode are wrong.
  """
  if game_name not in ATARI_GAMES:
    raise ValueError("Game %s not in ATARI_GAMES" % game_name)
  if game_mode not in ATARI_GAME_MODES:
    raise ValueError("Unknown ATARI game mode: %s." % game_mode)
  camel_game_name = misc_utils.snakecase_to_camelcase(game_name) + game_mode
  # Create and register the Problem
  cls = type("Gym%sRandom" % camel_game_name,
             (T2TGymEnv,), {"base_env_name": camel_game_name})
  registry.register_problem(cls)


# Register the atari games with all of the possible modes.
for atari_game in ATARI_GAMES:
  for atari_game_mode in ATARI_GAME_MODES:
    register_game(atari_game, game_mode=atari_game_mode)
