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

# Dependency imports

from tensor2tensor.rl.envs.in_graph_batch_env import InGraphBatchEnv

import tensorflow as tf


class WrapperBase(InGraphBatchEnv):
  """Base wrapper class."""

  def __init__(self, batch_env):
    self._length = len(batch_env)
    self._batch_env = batch_env
    self.action_shape = batch_env.action_shape
    self.action_dtype = batch_env.action_dtype

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  def __len__(self):
    """Number of combined environments."""
    return self._length

  def reset(self, indices=None):
    return self._batch_env.reset(indices)


class TransformWrapper(WrapperBase):
  """Transform wrapper."""

  def __init__(self, batch_env, transform_observation=None,
               transform_reward=tf.identity, transform_done=tf.identity):
    super(TransformWrapper, self).__init__(batch_env)
    if transform_observation is not None:
      _, observ_shape, observ_dtype = transform_observation  # pylint: disable=unpacking-non-sequence
      self._observ = tf.Variable(
          tf.zeros(len(self) + observ_shape, observ_dtype), trainable=False)
    else:
      self._observ = self._batch_env.observ

    self.transform_observation = transform_observation
    self.transform_reward = transform_reward
    self.transform_done = transform_done

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):  # Do we need this?
      reward, done = self._batch_env.simulate(action)
      with tf.control_dependencies([reward]):
        if self.transform_observation:
          observ = self.transform_observation[0](self._batch_env.observ)
          assign_op = self._observ.assign(observ)
        else:
          assign_op = tf.no_op()  # TODO(lukaszkaiser): looks as if it's broken.
        with tf.control_dependencies([assign_op]):
          return self.transform_reward(reward), self.transform_done(done)


class WarpFrameWrapper(TransformWrapper):
  """Wrap frames."""

  def __init__(self, batch_env):
    """Warp frames to 84x84 as done in the Nature paper and later work."""

    dims = [84, 84]
    nature_transform = lambda o: tf.image.rgb_to_grayscale(  # pylint: disable=g-long-lambda
        tf.image.resize_images(o, dims))

    super(WarpFrameWrapper, self).__init__(batch_env, transform_observation=(
        nature_transform, dims, tf.float32))


class ShiftRewardWrapper(TransformWrapper):
  """Shift the reward."""

  def __init__(self, batch_env, add_value):
    shift_reward = lambda r: tf.add(r, add_value)
    super(ShiftRewardWrapper, self).__init__(
        batch_env, transform_reward=shift_reward)


class MaxAndSkipWrapper(WrapperBase):
  """Max and skip wrapper."""

  def __init__(self, batch_env, skip=4):
    super(MaxAndSkipWrapper, self).__init__(batch_env)
    self.skip = skip
    self._observ = None
    observs_shape = batch_env.observ.shape
    observ_dtype = tf.float32
    self._observ = tf.Variable(tf.zeros(observs_shape, observ_dtype),
                               trainable=False)

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):  # Do we need this?
      initializer = (tf.zeros_like(self._observ),
                     tf.fill((len(self),), 0.0), tf.fill((len(self),), False))

      def not_done_step(a, _):
        reward, done = self._batch_env.simulate(action)
        with tf.control_dependencies([reward, done]):
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
  """Stack and skip wrapper."""

  def __init__(self, batch_env, skip=4):
    super(StackAndSkipWrapper, self).__init__(batch_env)
    self.skip = skip
    self._observ = None
    self.old_shape = batch_env.observ.shape.as_list()
    observs_shape = self.old_shape[:-1] + [self.old_shape[-1] * self.skip]
    observ_dtype = tf.float32
    self._observ = tf.Variable(tf.zeros(observs_shape, observ_dtype),
                               trainable=False)

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):  # Do we need this?
      initializer = (tf.zeros(self.old_shape, dtype=tf.float32),
                     tf.fill((len(self),), 0.0), tf.fill((len(self),), False))

      def not_done_step(a, _):
        reward, done = self._batch_env.simulate(action)
        with tf.control_dependencies([reward, done]):
          r0 = self._batch_env.observ
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


class TimeLimitWrapper(WrapperBase):
  """Time limit wrapper."""

  # TODO(lukaszkaiser): Check if TimeLimitWrapper does what it's supposed to do.
  def __init__(self, batch_env, timelimit=100):
    super(TimeLimitWrapper, self).__init__(batch_env)
    self.timelimit = timelimit
    self._time_elapsed = tf.Variable(tf.zeros((len(self),), tf.int32),
                                     trainable=False)

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):
      reward, done = self._batch_env.simulate(action)
      with tf.control_dependencies([reward, done]):
        new_done = tf.logical_or(done, self._time_elapsed > self.timelimit)
        inc = self._time_elapsed.assign_add(tf.ones_like(self._time_elapsed))

        with tf.control_dependencies([inc]):
          return tf.identity(reward), tf.identity(new_done)

  def reset(self, indices=None):
    op_zero = tf.scatter_update(self._time_elapsed, indices,
                                tf.zeros(tf.shape(indices), dtype=tf.int32))
    with tf.control_dependencies([op_zero]):
      return self._batch_env.reset(indices)


class MemoryWrapper(WrapperBase):
  """Memory wrapper."""

  def __init__(self, batch_env):
    super(MemoryWrapper, self).__init__(batch_env)
    MemoryWrapper.singleton = self
    assert self._length == 1, "We support only one environment"
    infinity = 10000000
    self.speculum = tf.FIFOQueue(infinity, dtypes=[
        tf.uint8, tf.float32, tf.int32, tf.bool])
    self._observ = self._batch_env.observ

  def simulate(self, action):
    with tf.name_scope("environment/simulate"):  # Do we need this?
      reward, done = self._batch_env.simulate(action)
      image = tf.cast(self._batch_env.observ[0, ...], tf.uint8)
      with tf.control_dependencies([reward, done]):
        enqueue_op = self.speculum.enqueue(
            [image, reward, action, done])
        with tf.control_dependencies([enqueue_op]):
          return tf.identity(reward), tf.identity(done)
