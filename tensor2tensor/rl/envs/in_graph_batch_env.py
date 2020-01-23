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

import gym

import tensorflow.compat.v1 as tf


class InGraphBatchEnv(object):
  """Abstract class for batch of environments inside the TensorFlow graph.
  """

  def __init__(self, observ_space, action_space):
    self.observ_space = observ_space
    self.action_space = action_space

  def __str__(self):
    return "InGraphEnv(%s)" % str(self._batch_env)

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
    raise NotImplementedError

  def reset(self, indices=None):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset.

    Returns:
      Batch tensor of the new observations.
    """
    return tf.cond(
        tf.cast(tf.reduce_sum(indices + 1), tf.bool),
        lambda: self._reset_non_empty(indices),
        lambda: tf.cast(0, self.observ_dtype))

  @staticmethod
  def _get_tf_dtype(space):
    if isinstance(space, gym.spaces.Discrete):
      return tf.int32
    if isinstance(space, gym.spaces.Box):
      return tf.as_dtype(space.dtype)
    raise NotImplementedError()

  @property
  def observ_dtype(self):
    return self._get_tf_dtype(self.observ_space)

  @property
  def observ_shape(self):
    return self.observ_space.shape

  @property
  def action_dtype(self):
    return self._get_tf_dtype(self.action_space)

  @property
  def action_shape(self):
    return self.action_space.shape

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ.read_value()

  def close(self):
    """Send close messages to the external process and join them."""
    self._batch_env.close()
