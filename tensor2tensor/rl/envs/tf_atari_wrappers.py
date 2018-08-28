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
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


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
    observs_shape = batch_env.observ.shape
    self._observ = tf.Variable(tf.zeros(observs_shape, self.observ_dtype),
                               trainable=False)

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


class StackAndSkipWrapper(WrapperBase):
  """ Stack and skip wrapper.
      The wrapper works under assumptions that issuing an action
      to an environment with done=True has not effect.
  """

  def __init__(self, batch_env, skip=4):
    super(StackAndSkipWrapper, self).__init__(batch_env)
    self.skip = skip
    self.old_shape = self._batch_env.observ_shape
    self._observ = tf.Variable(
        tf.zeros((len(self),) + self.observ_shape, self.observ_dtype),
        trainable=False)

  @property
  def observ_shape(self):
    return self.old_shape[:-1] + (self.old_shape[-1] * self.skip,)

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):  # Do we need this?
      initializer = (tf.zeros((len(self),) + self.old_shape, dtype=self.observ_dtype),
                     tf.fill((len(self),), 0.0), tf.fill((len(self),), False))

      def not_done_step(a, _):
        reward, done = self._batch_env.simulate(action)
        with tf.control_dependencies([reward, done]):
          r0 = self._batch_env.observ + 0
          r1 = tf.add(a[1], reward)
          r2 = tf.logical_or(a[2], done)
          return (r0, r1, r2)

      simulate_ret = tf.scan(not_done_step, tf.range(self.skip),
                             initializer=initializer, parallel_iterations=1,
                             infer_shape=False)
      observations, rewards, dones = simulate_ret
      split_observations = tf.split(observations, self.skip, axis=0)
      split_observations = [tf.squeeze(o, axis=0) for o in split_observations]
      observation = tf.concat(split_observations, axis=-1)
      with tf.control_dependencies([self._observ.assign(observation)]):
        return tf.identity(rewards[-1, ...]), tf.identity(dones[-1, ...])

  def _reset_non_empty(self, indices):
    # pylint: disable=protected-access
    new_values = self._batch_env._reset_non_empty(indices)
    # pylint: enable=protected-access
    inx = tf.concat(
        [
            tf.ones(tf.size(tf.shape(new_values)),
                dtype=tf.int32)[:-1],
            [self.skip]
        ],
        axis=0)
    assign_op = tf.scatter_update(self._observ, indices, tf.tile(
        new_values, inx))
    with tf.control_dependencies([assign_op]):
      return tf.gather(self.observ, indices)


class StackWrapper(WrapperBase):
  """ A wrapper which stacks previously seen frames. """

  def __init__(self, batch_env, history=4):
    super(StackWrapper, self).__init__(batch_env)
    self.history = history
    self.old_shape = batch_env.observ_shape
    self._observ = tf.Variable(
        tf.zeros((len(self),) + self.observ_shape, self.observ_dtype),
        trainable=False)

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
    inx = tf.concat(
        [
            tf.ones(tf.size(tf.shape(new_values)),
                dtype=tf.int32)[:-1],
            [self.history]
        ],
        axis=0)
    assign_op = tf.scatter_update(self._observ, indices, tf.tile(
        new_values, inx))
    with tf.control_dependencies([assign_op]):
      return tf.gather(self.observ, indices)


class AutoencoderWrapper(WrapperBase):
  """ Transforms the observations taking the bottleneck
      state of an autoencoder"""

  def __init__(self, batch_env):
    super(AutoencoderWrapper, self).__init__(batch_env)
    self._observ = self._observ = tf.Variable(
        tf.zeros((len(self),) + self.observ_shape, self.observ_dtype),
        trainable=False)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      autoencoder_hparams = autoencoders.autoencoder_discrete_pong()
      trainer_lib.add_problem_hparams(autoencoder_hparams,
          "gym_discrete_problem_with_agent_on_wrapped_full_pong_with"
          "_autoencoder")
      self.autoencoder_model = autoencoders.AutoencoderOrderedDiscrete(
          autoencoder_hparams, tf.estimator.ModeKeys.EVAL)
    self.autoencoder_model.set_mode(tf.estimator.ModeKeys.EVAL)

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

  def simulate(self, action):
    reward, done = self._batch_env.simulate(action)
    with tf.control_dependencies([reward, done]):
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        ret = self.autoencoder_model.encode(self._batch_env.observ)
        ret = tf.cast(ret, self.observ_dtype)
        assign_op = self._observ.assign(ret)
        with tf.control_dependencies([assign_op]):
          return tf.identity(reward), tf.identity(done)

  def _reset_non_empty(self, indices):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      new_values = self._batch_env._reset_non_empty(indices)  # pylint: disable=protected-access
      ret = self.autoencoder_model.encode(new_values)
      ret = tf.cast(ret, self.observ_dtype)
      assign_op = tf.scatter_update(self._observ, indices, ret)
      with tf.control_dependencies([assign_op]):
        return tf.gather(self.observ, indices)


class IntToBitWrapper(WrapperBase):
  """Unpacks the observations from integer values to bit values"""

  def __init__(self, batch_env):
    super(IntToBitWrapper, self).__init__(batch_env)
    self._observ = self._observ = tf.Variable(
        tf.zeros((len(self),) + self.observ_shape, self.observ_dtype),
        trainable=False)

  @property
  def observ_shape(self):
    height, width, channels = self._batch_env.observ_shape
    # We treat each channel as 8-bit integer to be expanded to 8 channels
    return (height, width, channels*8)

  def simulate(self, action):
    action = tf.Print(action, [action], message="action=", summarize=200)

    # action = tf.zeros_like(action) #Temporary hacked bugfix
    reward, done = self._batch_env.simulate(action)
    with tf.control_dependencies([reward, done]):
      with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        unpacked = discretization.int_to_bit(self._batch_env.observ, 8)
        unpacked = tf.reshape(unpacked, (-1,)+self.observ_shape)
        unpacked = tf.cast(unpacked, self.observ_dtype)
        assign_op = self._observ.assign(unpacked)
        with tf.control_dependencies([assign_op]):
          return tf.identity(reward), tf.identity(done)

  def _reset_non_empty(self, indices):
    # pylint: disable=protected-access
    new_values = self._batch_env._reset_non_empty(indices)
    new_values_unpacked = discretization.int_to_bit(new_values, 8)
    new_values_unpacked = tf.reshape(new_values_unpacked, (-1,)
                                     +self.observ_shape)
    new_values_unpacked = tf.cast(new_values_unpacked, self.observ_dtype)
    # pylint: enable=protected-access
    assign_op = tf.scatter_update(self._observ, indices, new_values_unpacked)
    with tf.control_dependencies([assign_op]):
      return tf.identity(new_values_unpacked)
