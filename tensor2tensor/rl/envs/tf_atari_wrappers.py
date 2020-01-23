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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv

import tensorflow.compat.v1 as tf


class WrapperBase(InGraphBatchEnv):
  """Base wrapper class."""

  def __init__(self, batch_env):
    super(WrapperBase, self).__init__(
        batch_env.observ_space, batch_env.action_space)
    self._length = len(batch_env)
    self._batch_env = batch_env

  def initialize(self, sess):
    """Initializations to be run once the tf.Session is available."""
    pass

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ.read_value()

  @property
  def observ_shape(self):
    return self._batch_env.observ_shape

  def __len__(self):
    """Number of combined environments."""
    return self._length

  def _reset_non_empty(self, indices):
    # pylint: disable=protected-access
    new_values = self._batch_env._reset_non_empty(indices)
    # pylint: enable=protected-access
    assign_op = tf.scatter_update(self._observ, indices, new_values)
    with tf.control_dependencies([assign_op]):
      return tf.identity(new_values)

  def _transform_history_observations(self, frames):
    """Applies a wrapper-specific transformation to the history observations.

    Overridden in wrappers that alter observations.

    Args:
      frames: A tensor of history frames to transform.

    Returns:
      a tensor of transformed frames.
    """
    return frames

  @property
  def history_observations(self):
    """Returns observations from the root simulated env's history_buffer.

    Transforms them with a wrapper-specific function if necessary.

    Raises:
      AttributeError: if root env doesn't have a history_buffer (i.e. is not
        simulated).
    """
    return self._transform_history_observations(
        self._batch_env.history_observations
    )


class StackWrapper(WrapperBase):
  """A wrapper which stacks previously seen frames."""

  def __init__(self, batch_env, history=4):
    super(StackWrapper, self).__init__(batch_env)
    self.history = history
    self.old_shape = batch_env.observ_shape
    # TODO(afrozm): Make into tf.get_variable and use_resource=True
    self._observ = tf.Variable(
        tf.zeros((len(self),) + self.observ_shape, self.observ_dtype),
        trainable=False)

  def __str__(self):
    return "StackWrapper(%s)" % str(self._batch_env)

  @property
  def observ_shape(self):
    return (self.history,) + self.old_shape

  def simulate(self, action):
    reward, done = self._batch_env.simulate(action)
    with tf.control_dependencies([reward, done]):
      new_observ = tf.expand_dims(self._batch_env.observ, axis=1)

      # If we shouldn't stack, i.e. self.history == 1, then just assign
      # new_observ to self._observ and return from here.
      if self.history == 1:
        with tf.control_dependencies([self._observ.assign(new_observ)]):
          return tf.identity(reward), tf.identity(done)

      # If we should stack, then do the required work.
      old_observ = tf.gather(
          self._observ.read_value(),
          list(range(1, self.history)),
          axis=1)
      with tf.control_dependencies([new_observ, old_observ]):
        with tf.control_dependencies([self._observ.assign(
            tf.concat([old_observ, new_observ], axis=1))]):
          return tf.identity(reward), tf.identity(done)

  def _reset_non_empty(self, indices):
    # pylint: disable=protected-access
    new_values = self._batch_env._reset_non_empty(indices)
    # pylint: enable=protected-access
    initial_frames = getattr(self._batch_env, "history_observations", None)

    num_dimensions_in_env_observation = len(self.old_shape)

    if initial_frames is None:
      inx = [1, self.history] + ([1] * num_dimensions_in_env_observation)
      initial_frames = tf.tile(tf.expand_dims(new_values, axis=1), inx)
    with tf.control_dependencies([new_values]):
      assign_op = tf.scatter_update(self._observ, indices, initial_frames)
    with tf.control_dependencies([assign_op]):
      return tf.gather(self.observ, indices)

  def _transform_history_observations(self, frames):
    # Should be implemented if ever two StackWrappers are to be used together.
    raise NotImplementedError
