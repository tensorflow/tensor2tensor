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

from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


class HistoryBuffer():

  def __init__(self, initial_frames):
    self._history_buff = None
    self.initial_frames = initial_frames
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self._history_buff = tf.get_variable(
            "history_observ", initializer=self.initial_frames, trainable=False)

  def get_all_elements(self):
    return self._history_buff.read_value()

  def move_by_one_element(self, element):
    last_removed = self._history_buff.read_value()[1:, ...]
    element = tf.expand_dims(element, dim=0)
    moved = tf.concat([last_removed, element], axis=0)
    with tf.control_dependencies([moved]):
      with tf.control_dependencies([self._history_buff.assign(moved)]):
        return self._history_buff.read_value()

  def reset(self):
    with tf.control_dependencies([self._history_buff.assign(self.initial_frames)]):
      return self._history_buff.read_value()


class SimulatedBatchEnv(InGraphBatchEnv):
  """Batch of environments inside the TensorFlow graph.

  The batch of environments will be stepped and reset inside of the graph using
  a tf.py_func(). The current batch of observations, actions, rewards, and done
  flags are held in according variables.
  """
  NUMBER_OF_HISTORY_FRAMES = 4

  def __init__(self, environment_lambda, length):
    """Batch of environments inside the TensorFlow graph."""
    self.length = length

    #TODO(piotrmilos): For the moment we are fine with that.
    assert self.length==1, "Currently SimulatedBatchEnv support only one env"
    initialization_env = environment_lambda()
    hparams = trainer_lib.create_hparams(
        FLAGS.hparams_set, problem_name=FLAGS.problem, data_dir="UNUSED")
    hparams.force_full_predict = True
    self._model = registry.model(FLAGS.model)(
        hparams, tf.estimator.ModeKeys.PREDICT)

    self.action_space = initialization_env.action_space
    self.action_shape = list(initialization_env.action_space.shape)
    self.action_dtype = tf.int32

    obs = []
    if hasattr(initialization_env.env, "get_starting_data"):
      obs, _, _ = initialization_env.env.get_starting_data()
    else:
      # piotrmilos
      # Ancient method for environments not supporting get_starting_data
      # This is probably not compatibile with NUMBER_OF_HISTORY_FRAMES!=2
      # Should be removed at some point
      num_frames = SimulatedBatchEnv.NUMBER_OF_HISTORY_FRAMES
      initialization_env.reset()
      skip_frames = 20
      for _ in range(skip_frames):
        initialization_env.step(0)
      for i in range(num_frames):
        obs.append(initialization_env.step(0)[0])

    initial_frames = tf.stack(obs)
    initial_frames = tf.cast(initial_frames, tf.float32)

    self.history_buffer = HistoryBuffer(initial_frames)

    shape = (self.length,) + initialization_env.observation_space.shape
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self._observ = tf.get_variable(
        "observ", shape, initializer=tf.zeros_initializer, trainable=False)

  def __len__(self):
    """Number of combined environments."""
    return self.length

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):
      action = tf.expand_dims(action, axis=0)
      actions = [action]*SimulatedBatchEnv.NUMBER_OF_HISTORY_FRAMES
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
          with tf.control_dependencies([self._observ.assign(observ),
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
      with tf.control_dependencies([self._observ.assign(0.0*self._observ)]):
        return tf.identity(self._observ.read_value())


  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ
