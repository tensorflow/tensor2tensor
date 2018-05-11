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
"""Batch of environments inside the TensorFlow graph."""

# The code was based on Danijar Hafner's code from tf.agents:
# https://github.com/tensorflow/agents/blob/master/agents/tools/in_graph_batch_env.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


class HistoryBuffer(object):
  """History Buffer."""

  def __init__(self, input_data_iterator, problem):
    self.input_data_iterator = input_data_iterator
    self.autoencoder_model = None
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      if FLAGS.autoencoder_path:
        # Feeds for autoencoding.
        problem.setup_autoencoder()
        self.autoencoder_model = problem.autoencoder_model
        self.autoencoder_model.set_mode(tf.estimator.ModeKeys.EVAL)
    initial_frames = self.get_initial_observations()
    self._history_buff = None
    initial_shape = common_layers.shape_list(initial_frames)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      history_buff = tf.get_variable("history_observ", initial_shape,
                                     initializer=tf.zeros_initializer,
                                     trainable=False)
    self._history_buff = history_buff
    self._assigned = False

  def get_initial_observations(self):
    initial_frames = self.input_data_iterator.get_next()["inputs"]
    if self.autoencoder_model:
      autoencoded = self.autoencoder_model.encode(
        tf.expand_dims(initial_frames, axis=1))
      autoencoded_shape = common_layers.shape_list(autoencoded)
      autoencoded = tf.reshape(  # Make 8-bit groups.
        autoencoded, autoencoded_shape[:-1] + [3, 8])
      initial_frames = discretization.bit_to_int(autoencoded, 8)
      initial_frames = tf.to_float(initial_frames)
    else:
      initial_frames = tf.cast(initial_frames, tf.float32)
    return initial_frames

  def get_all_elements(self):
    if self._assigned:
      return self._history_buff.read_value()
    initial_frames = self.get_initial_observations()
    with tf.control_dependencies([self.history_buff.assign(initial_frames)]):
      self._assigned = True
      return tf.identity(initial_frames)

  def move_by_one_element(self, element):
    last_removed = self.get_all_elements()[1:, ...]
    element = tf.expand_dims(element, dim=0)
    moved = tf.concat([last_removed, element], axis=0)
    with tf.control_dependencies([moved]):
      with tf.control_dependencies([self._history_buff.assign(moved)]):
        self._assigned = True
        return self._history_buff.read_value()

  def reset(self):
    with tf.control_dependencies([self._history_buff.assign(
        self.get_initial_observations())]):
      self._assigned = True
      return self._history_buff.read_value()


class SimulatedBatchEnv(InGraphBatchEnv):
  """Batch of environments inside the TensorFlow graph.

  The batch of environments will be stepped and reset inside of the graph using
  a tf.py_func(). The current batch of observations, actions, rewards, and done
  flags are held in according variables.
  """

  def __init__(self, environment_lambda, length, problem):
    """Batch of environments inside the TensorFlow graph."""
    self.length = length
    self._num_frames = problem.num_input_frames

    # TODO(piotrmilos): For the moment we are fine with that.
    assert self.length == 1, "Currently SimulatedBatchEnv support only one env"
    initialization_env = environment_lambda()
    hparams = trainer_lib.create_hparams(
        FLAGS.hparams_set, problem_name=FLAGS.problem)
    hparams.force_full_predict = True
    self._model = registry.model(FLAGS.model)(
        hparams, tf.estimator.ModeKeys.PREDICT)

    self.action_space = initialization_env.action_space
    self.action_shape = list(initialization_env.action_space.shape)
    self.action_dtype = tf.int32

    dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, FLAGS.data_dir)
    input_data_iterator = dataset.make_one_shot_iterator()

    self.history_buffer = HistoryBuffer(input_data_iterator, problem=problem)

    height, width, channels = initialization_env.observation_space.shape
    # TODO(lukaszkaiser): remove this and just use Problem.frame_height.
    if FLAGS.autoencoder_path:
      height = problem.frame_height
      width = problem.frame_width
    shape = (self.length, height, width, channels)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self._observ = tf.get_variable(
          "observ", shape, initializer=tf.zeros_initializer, trainable=False)

  def __len__(self):
    """Number of combined environments."""
    return self.length

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):
      action = tf.expand_dims(action, axis=0)
      actions = [action] * self._num_frames
      actions = tf.concat(actions, axis=0)
      history = self.history_buffer.get_all_elements()
      inputs = {"inputs": tf.expand_dims(history, axis=0),  # Add batch.
                "input_action": tf.expand_dims(actions, axis=0)}
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        model_output = self._model.infer(inputs)
      observ = model_output["targets"]
      observ = tf.cast(observ[:, 0, :, :, :], tf.float32)
      # TODO(lukaszkaiser): instead of -1 use min_reward in the line below.
      reward = model_output["target_reward"][:, 0, 0, 0] - 1
      reward = tf.cast(reward, tf.float32)
      # Some wrappers need explicit shape, so we reshape here.
      reward = tf.reshape(reward, shape=(self.length,))
      done = tf.constant(False, tf.bool, shape=(self.length,))

      with tf.control_dependencies([observ]):
        with tf.control_dependencies(
            [self._observ.assign(observ),
             self.history_buffer.move_by_one_element(observ[0, ...])]):
          return tf.identity(reward), tf.identity(done)

  def reset(self, indices=None):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset.

    Returns:
      Batch tensor of the new observations.
    """
    return tf.cond(
        tf.cast(tf.shape(indices)[0], tf.bool),
        lambda: self._reset_non_empty(indices), lambda: 0.0)

  def _reset_non_empty(self, indices):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset; defaults to all.

    Returns:
      Batch tensor of the new observations.
    """
    with tf.control_dependencies([self.history_buffer.reset()]):
      with tf.control_dependencies([self._observ.assign(
              self.history_buffer.get_all_elements()[-1:, ...])]):
        return tf.identity(self._observ.read_value())

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ
