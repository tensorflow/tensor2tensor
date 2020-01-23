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

"""Batch of environments inside the TensorFlow graph."""

# The code was based on Danijar Hafner's code from tf.agents:
# https://github.com/tensorflow/agents/blob/master/agents/tools/in_graph_batch_env.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv
import tensorflow.compat.v1 as tf


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
    super(PyFuncBatchEnv, self).__init__(batch_env.observation_space,
                                         batch_env.action_space)
    self._batch_env = batch_env
    with tf.variable_scope("env_temporary"):
      self._observ = tf.Variable(
          tf.zeros((self._batch_env.batch_size,) + self.observ_shape,
                   self.observ_dtype),
          name="observ", trainable=False)

  def __str__(self):
    return "PyFuncEnv(%s)" % str(self._batch_env)

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
    return self._batch_env.batch_size

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
    with tf.name_scope("environment/simulate"):
      if action.dtype in (tf.float16, tf.float32, tf.float64):
        action = tf.check_numerics(action, "action")
      def step(action):
        step_response = self._batch_env.step(action)
        # Current env doesn't return `info`, but EnvProblem does.
        # TODO(afrozm): The proper way to do this is to make T2TGymEnv return
        # an empty info return value.
        if len(step_response) == 3:
          (observ, reward, done) = step_response
        else:
          (observ, reward, done, _) = step_response
        return (observ, reward.astype(np.float32), done)
      observ, reward, done = tf.py_func(
          step, [action],
          [self.observ_dtype, tf.float32, tf.bool], name="step")
      reward = tf.check_numerics(reward, "reward")
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
    observ = tf.py_func(
        self._batch_env.reset, [indices], self.observ_dtype, name="reset")
    observ.set_shape(indices.get_shape().concatenate(self.observ_shape))
    with tf.control_dependencies([
        tf.scatter_update(self._observ, indices, observ)]):
      return tf.identity(observ)

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ.read_value()

  def close(self):
    """Send close messages to the external process and join them."""
    self._batch_env.close()
