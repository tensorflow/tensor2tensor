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

from tensor2tensor.rl.envs import utils
from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv
import tensorflow as tf


class PyFuncBatchEnv(InGraphBatchEnv):
  """Batch of environments inside the TensorFlow graph.

  The batch of environments will be stepped and reset inside of the graph using
  a tf.py_func(). The current batch of observations, actions, rewards, and done
  flags are held in according variables.
  """

  def __init__(self, batch_env):
    """Batch of environments inside the TensorFlow graph.

    Args:
      batch_env: Batch environment.
    """
    self._batch_env = batch_env
    observ_shape = utils.parse_shape(self._batch_env.observation_space)
    observ_dtype = utils.parse_dtype(self._batch_env.observation_space)
    self.action_shape = list(utils.parse_shape(self._batch_env.action_space))
    self.action_dtype = utils.parse_dtype(self._batch_env.action_space)
    with tf.variable_scope('env_temporary'):
      self._observ = tf.Variable(
          tf.zeros((len(self._batch_env),) + observ_shape, observ_dtype),
          name='observ', trainable=False)

  def __getattr__(self, name):
    """Forward unimplemented attributes to one of the original environments.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name in one of the original environments.
    """
    return getattr(self._batch_env, name)

  def initialize(self, sess):
    pass

  def __len__(self):
    """Number of combined environments."""
    return len(self._batch_env)

  def __getitem__(self, index):
    """Access an underlying environment by index."""
    return self._batch_env[index]

  def simulate(self, action):
    """Step the batch of environments.

    The results of the step can be accessed from the variables defined below.

    Args:
      action: Tensor holding the batch of actions to apply.

    Returns:
      Operation.
    """
    with tf.name_scope('environment/simulate'):
      if action.dtype in (tf.float16, tf.float32, tf.float64):
        action = tf.check_numerics(action, 'action')
      observ_dtype = utils.parse_dtype(self._batch_env.observation_space)
      observ, reward, done = tf.py_func(
          lambda a: self._batch_env.step(a)[:3], [action],
          [observ_dtype, tf.float32, tf.bool], name='step')
      observ = tf.check_numerics(observ, 'observ')
      reward = tf.check_numerics(reward, 'reward')
      reward.set_shape((len(self),))
      done.set_shape((len(self),))
      with tf.control_dependencies([self._observ.assign(observ)]):
        return tf.identity(reward), tf.identity(done)

  def _reset_non_empty(self, indices):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset; defaults to all.

    Returns:
      Batch tensor of the new observations.
    """
    observ_dtype = utils.parse_dtype(self._batch_env.observation_space)
    observ = tf.py_func(
        self._batch_env.reset, [indices], observ_dtype, name='reset')
    observ = tf.check_numerics(observ, 'observ')
    with tf.control_dependencies([
        tf.scatter_update(self._observ, indices, observ)]):
      return tf.identity(observ)

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  def close(self):
    """Send close messages to the external process and join them."""
    self._batch_env.close()
