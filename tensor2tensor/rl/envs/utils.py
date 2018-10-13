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

"""Utilities for using batched environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import six
import tensorflow as tf


class EvalVideoWrapper(gym.Wrapper):
  """Wrapper for recording videos during eval phase.

  This wrapper is designed to record videos via gym.wrappers.Monitor and
  simplifying its usage in t2t collect phase.
  It alleviate the limitation of Monitor, which doesn't allow reset on an
  active environment.

  EvalVideoWrapper assumes that only every second trajectory (after every
  second reset) will be used by the caller:
  - on the "active" runs it behaves as gym.wrappers.Monitor,
  - on the "inactive" runs it doesn't call underlying environment and only
    returns last seen observation.
  Videos are only generated during the active runs.
  """

  def __init__(self, env):
    super(EvalVideoWrapper, self).__init__(env)
    self._reset_counter = 0
    self._active = False
    self._last_returned = None

  def _step(self, action):
    if self._active:
      self._last_returned = self.env.step(action)
    if self._last_returned is None:
      raise Exception("Environment stepped before proper reset.")
    return self._last_returned

  def _reset(self, **kwargs):
    self._reset_counter += 1
    if self._reset_counter % 2 == 1:
      self._active = True
      return self.env.reset(**kwargs)

    self._active = False
    self._last_returned = (self._last_returned[0],
                           self._last_returned[1],
                           False,  # done = False
                           self._last_returned[3])
    return self._last_returned[0]


def get_observation_space(environment_spec):
  """Get observation space associated with environment spec.

  Args:
     environment_spec:  EnvironmentSpec object

  Returns:
    OpenAi Gym observation space
  """
  return environment_spec.env_lambda().observation_space


def get_action_space(environment_spec):
  """Get action space associated with environment spec.

  Args:
     environment_spec:  Object consisting one of batch_env.action_space, or
     env_lambda().action_space

  Returns:
    OpenAi Gym action space
  """
  if "batch_env" in environment_spec:
    action_space = environment_spec.batch_env.action_space
  else:
    action_space = environment_spec.env_lambda().action_space
  return action_space


def get_policy(observations, hparams):
  """Get a policy network.

  Args:
    observations: Tensor with observations
    hparams: parameters

  Returns:
    Tensor with policy and value function output
  """
  policy_network_lambda = hparams.policy_network
  action_space = get_action_space(hparams.environment_spec)
  return policy_network_lambda(action_space, hparams, observations)


def parse_shape(space):
  """Get a tensor shape from a OpenAI Gym space.

  Args:
    space: Gym space.

  Returns:
    Shape tuple.
  """
  if isinstance(space, gym.spaces.Discrete):
    return ()
  if isinstance(space, gym.spaces.Box):
    return space.shape
  raise NotImplementedError()


def parse_dtype(space):
  """Get a tensor dtype from a OpenAI Gym space.

  Args:
    space: Gym space.

  Returns:
    TensorFlow data type.
  """
  if isinstance(space, gym.spaces.Discrete):
    return tf.int32
  if isinstance(space, gym.spaces.Box):
    return tf.as_dtype(space.dtype)
  raise NotImplementedError()


class InitialFrameChooser(object):
  """Class for choosing the initial frame for simulation from the dataset.

  Can also store a sequence of later frames, which is used for comparison in
  world model evaluation.

  Attributes:
    batch_size (int): Batch size, should be set before calling choose().
    trajectory (dict): Dict of Variables storing a sequence of frames after the
        chosen one.
  """

  def __init__(self, environment_spec, mode, trajectory_length=1):
    self._initial_frames_problem = environment_spec.initial_frames_problem
    self._simulation_random_starts = environment_spec.simulation_random_starts
    self._flip_first_random_for_beginning = \
        environment_spec.simulation_flip_first_random_for_beginning
    self._num_initial_frames = environment_spec.video_num_input_frames

    def dataset_kwargs_lambda():
      video_num_input_frames = environment_spec.video_num_input_frames
      video_num_input_frames += trajectory_length - 1
      dataset_hparams = tf.contrib.training.HParams(
          video_num_input_frames=video_num_input_frames,
          video_num_target_frames=environment_spec.video_num_target_frames,
          environment_spec=environment_spec
      )
      return {
          "mode": mode,
          "data_dir": tf.flags.FLAGS.data_dir,
          "hparams": dataset_hparams,
          "only_last": True
      }

    self._dataset_kwargs_lambda = dataset_kwargs_lambda
    self._start_frames = None

  @property
  def batch_size(self):
    return self._batch_size

  @batch_size.setter
  def batch_size(self, batch_size):
    self._batch_size = batch_size
    self._iterator = \
        self._create_initial_frame_dataset().make_initializable_iterator()

    def fix_and_shorten(shape):
      shape = shape.as_list()
      shape[0] = batch_size
      shape[1] -= self._num_initial_frames - 1
      return shape

    shapes = self._extract_input(self._iterator.output_shapes)
    types = self._extract_input(self._iterator.output_types)
    self.trajectory = {
        key: tf.Variable(
            tf.zeros(fix_and_shorten(shape), types[key]),
            trainable=False
        )
        for (key, shape) in six.iteritems(shapes)
    }

  def initialize(self, sess):
    sess.run(self._iterator.initializer)

  def choose(self):
    """Returns a dict of tensors of the chosen initial frame.

    Also assigns the first trajectory_length frames after the initial frames to
    self.trajectory.
    """
    if self._flip_first_random_for_beginning and self._start_frames is None:
      ordered_dataset = self._create_dataset(shuffle_files=False)
      # Later flip the first random frame in PPO batch for the true beginning.
      self._start_frames = self._extract_input(
          ordered_dataset.make_one_shot_iterator().get_next()
      )

    all_frames = self._extract_input(self._iterator.get_next())
    if self._start_frames is not None:
      all_frames = {
          key: tf.concat([
              tf.expand_dims(self._start_frames[key], axis=0),
              value[1:, ...]
          ], axis=0)
          for (key, value) in six.iteritems(all_frames)
      }
    scatter_ops = [
        tf.scatter_update(
            self.trajectory[key], tf.range(tf.shape(value)[0]),
            value[:, (self._num_initial_frames - 1):, ...]
        )
        for (key, value) in six.iteritems(all_frames)
    ]

    with tf.control_dependencies(scatter_ops):
      return {
          key: value[:, :self._num_initial_frames, ...]
          for (key, value) in six.iteritems(all_frames)
      }

  def _create_dataset(self, **extra_dataset_kwargs):
    dataset_kwargs = self._dataset_kwargs_lambda()
    dataset_kwargs.update(extra_dataset_kwargs)
    return self._initial_frames_problem.dataset(**dataset_kwargs)

  def _create_initial_frame_dataset(self):
    """Returns the dataset that consecutive initial frames will be taken from.
    """
    dataset = self._create_dataset(
        shuffle_files=self._simulation_random_starts
    )
    if self._simulation_random_starts:
      dataset = dataset.shuffle(buffer_size=1000)
    return dataset.repeat().batch(self._batch_size)

  def _extract_input(self, frame):
    input_frame = {"inputs": frame["inputs"]}
    input_frame.update({
        key[len("input_"):]: value
        for (key, value) in six.iteritems(frame)
        if key.startswith("input_")
    })
    return input_frame
