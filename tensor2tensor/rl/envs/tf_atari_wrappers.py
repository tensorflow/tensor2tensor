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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import discretization
from tensor2tensor.models.research import autoencoders
from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv
from tensor2tensor.utils import registry

import tensorflow as tf


class WrapperBase(InGraphBatchEnv):
  """Base wrapper class."""

  def __init__(self, batch_env):
    super(WrapperBase, self).__init__(
        batch_env.observ_space, batch_env.action_space)
    self._length = len(batch_env)
    self._batch_env = batch_env
    self._observ = self._observ = tf.Variable(
        tf.zeros((len(self),) + self.observ_shape, self.observ_dtype),
        trainable=False)

  def initialize(self, sess):
    """Initializations to be run once the tf.Session is available."""
    pass

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ.read_value()

  def __len__(self):
    """Number of combined environments."""
    return self._length

  def simulate(self, action):
    reward, done = self._batch_env.simulate(action)
    with tf.control_dependencies([reward, done]):
      observations = self._batch_env.observ
      observations = self._transform_observations(observations)
      assign_op = self._observ.assign(observations)
      with tf.control_dependencies([assign_op]):
        return tf.identity(reward), tf.identity(done)

  def _reset_non_empty(self, indices):
    # pylint: disable=protected-access
    new_values = self._batch_env._reset_non_empty(indices)
    # pylint: enable=protected-access
    new_values = self._transform_observations(new_values)
    assign_op = tf.scatter_update(self._observ, indices, new_values)
    with tf.control_dependencies([assign_op]):
      return tf.identity(new_values)

  def _transform_observations(self, observations):
    """Applies a wrapper-specific transformation to the batch of observations.

    Overridden in wrappers that alter observations.

    Args:
      observations: A tensor of history frames to transform. Shape:
        [batch_size, ...].

    Returns a tensor of transformed observations.
    """
    return observations

  def _transform_history_observations(self, observations):
    """Applies a wrapper-specific transformation to the history observations.

    Overridden in wrappers that alter the history of observations.

    Args:
      frames: A tensor of history frames to transform. Shape:
        [batch_size, history_size, ...].

    Returns:
      a tensor of transformed frames.
    """
    batch_size, history_size = observations.get_shape().as_list()[:2]
    observations = tf.reshape(
        observations, (-1,) + self._batch_env.observ_shape
    )
    observations = self._transform_observations(observations)
    return tf.reshape(
        observations, (batch_size, history_size) + self.observ_shape
    )

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


class RewardClippingWrapper(WrapperBase):
  """ Reward clipping wrapper.
      The rewards are clipped to -1, 0, 1
      This is a common strategy to ensure learning stability
      of rl algorithms
  """

  def simulate(self, action):
    reward, done = self._batch_env.simulate(action)
    with tf.control_dependencies([reward, done]):
      return tf.sign(reward), tf.identity(done)


class MaxAndSkipWrapper(WrapperBase):
  """ Max and skip wrapper.
      The wrapper works under assumptions that issuing an action
      to an environment with done=True has not effect.
  """

  def __init__(self, batch_env, skip=4):
    super(MaxAndSkipWrapper, self).__init__(batch_env)
    self.skip = skip

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):  # Do we need this?
      initializer = (tf.zeros_like(self._observ),
                     tf.fill((len(self),), 0.0), tf.fill((len(self),), False))

      def not_done_step(a, _):
        reward, done = self._batch_env.simulate(action)
        with tf.control_dependencies([reward, done]):
          # TODO(piotrmilos): possibly ignore envs with done
          r0 = tf.maximum(a[0], self._batch_env.observ)
          r1 = tf.add(a[1], reward)
          r2 = tf.logical_or(a[2], done)

          return (r0, r1, r2)

      simulate_ret = tf.scan(not_done_step, tf.range(self.skip),
                             initializer=initializer, parallel_iterations=1,
                             infer_shape=False)
      simulate_ret = [ret[-1, ...] for ret in simulate_ret]

      with tf.control_dependencies([self._observ.assign(simulate_ret[0])]):
        return tf.identity(simulate_ret[1]), tf.identity(simulate_ret[2])

  def _transform_history_observations(self, frames):
    # Should be implemented if ever MaxAndSkipWrapper and StackWrapper are to
    # be used together.
    raise NotImplementedError


class StackWrapper(WrapperBase):
  """ A wrapper which stacks previously seen frames. """

  def __init__(self, batch_env, history=4):
    self.history = history
    self.old_shape = batch_env.observ_shape
    super(StackWrapper, self).__init__(batch_env)

  @property
  def observ_shape(self):
    return self.old_shape[:-1] + (self.old_shape[-1] * self.history,)

  def simulate(self, action):
    reward, done = self._batch_env.simulate(action)
    with tf.control_dependencies([reward, done]):
      new_observ = self._batch_env.observ + 0
      old_observ = tf.gather(
          self._observ.read_value(),
          list(range(self.old_shape[-1], self.old_shape[-1] * self.history)),
          axis=-1)
      with tf.control_dependencies([new_observ, old_observ]):
        with tf.control_dependencies([self._observ.assign(
            tf.concat([old_observ, new_observ], axis=-1))]):
          return tf.identity(reward), tf.identity(done)

  def _reset_non_empty(self, indices):
    # pylint: disable=protected-access
    new_values = self._batch_env._reset_non_empty(indices)
    # pylint: enable=protected-access
    initial_frames = getattr(self._batch_env, "history_observations", None)
    if initial_frames is not None:
      # Using history buffer frames for initialization, if they are available.
      with tf.control_dependencies([new_values]):
        # Transpose to [batch, height, width, history, channels] and merge
        # history and channels into one dimension.
        initial_frames = tf.transpose(initial_frames, [0, 2, 3, 1, 4])
        initial_frames = tf.reshape(initial_frames,
                                    (len(self),) + self.observ_shape)
    else:
      inx = tf.concat(
          [
              tf.ones(tf.size(tf.shape(new_values)),
                      dtype=tf.int32)[:-1],
              [self.history]
          ],
          axis=0)
      initial_frames = tf.tile(new_values, inx)
    assign_op = tf.scatter_update(self._observ, indices, initial_frames)
    with tf.control_dependencies([assign_op]):
      return tf.gather(self.observ, indices)

  def _transform_history_observations(self, frames):
    # Should be implemented if ever two StackWrappers are to be used together.
    raise NotImplementedError


class AutoencoderWrapper(WrapperBase):
  """ Transforms the observations taking the bottleneck
      state of an autoencoder"""

  def __init__(self, batch_env):
    super(AutoencoderWrapper, self).__init__(batch_env)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      autoencoder_hparams = autoencoders.autoencoder_discrete_pong()
      problem = registry.problem("dummy_autoencoder_problem")
      autoencoder_hparams.problem_hparams = problem.get_hparams(
          autoencoder_hparams)
      autoencoder_hparams.problem = problem
      self.autoencoder_model = autoencoders.AutoencoderOrderedDiscrete(
          autoencoder_hparams, tf.estimator.ModeKeys.EVAL)

  @property
  def observ_shape(self):
    height, width, _ = self._batch_env.observ_shape
    ae_height = int(math.ceil(height / self.autoencoder_factor))
    ae_width = int(math.ceil(width / self.autoencoder_factor))
    ae_channels = 24  # TODO(piotrmilos): make it better
    return (ae_height, ae_width, ae_channels)

  @property
  def autoencoder_factor(self):
    """By how much to divide sizes when using autoencoders."""
    hparams = autoencoders.autoencoder_discrete_pong()
    return 2**hparams.num_hidden_layers

  def _transform_observations(self, observations):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      observations = tf.cast(observations, tf.int32)
      observations = self.autoencoder_model.encode(observations)
    return tf.cast(observations, self.observ_dtype)


class IntToBitWrapper(WrapperBase):
  """Unpacks the observations from integer values to bit values"""

  @property
  def observ_shape(self):
    height, width, channels = self._batch_env.observ_shape
    # We treat each channel as 8-bit integer to be expanded to 8 channels
    return (height, width, channels*8)

  def _transform_observations(self, observations):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      observations = discretization.int_to_bit(observations, 8)
      observations = tf.reshape(observations, (-1,) + self.observ_shape)
      return tf.cast(observations, self.observ_dtype)
