# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Neural networks for actor-critic algorithms."""

import collections
import functools
import operator

# Dependency imports

import gym
import tensorflow as tf


NetworkOutput = collections.namedtuple(
    'NetworkOutput', 'policy, value, action_postprocessing')


def feed_forward_gaussian_fun(observation_space, action_space, config,
                              observations):
  """Feed-forward gaussian."""
  assert isinstance(observation_space, gym.spaces.box.Box)

  mean_weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_mean_factor)
  logstd_initializer = tf.random_normal_initializer(config.init_logstd, 1e-10)

  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])

  with tf.variable_scope('policy'):
    x = flat_observations
    for size in config.policy_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    mean = tf.contrib.layers.fully_connected(
        x, action_space.shape[0], tf.tanh,
        weights_initializer=mean_weights_initializer)
    logstd = tf.get_variable(
        'logstd', mean.shape[2:], tf.float32, logstd_initializer)
    logstd = tf.tile(
        logstd[None, None],
        [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
  with tf.variable_scope('value'):
    x = flat_observations
    for size in config.value_layers:
      x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
    value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  mean = tf.check_numerics(mean, 'mean')
  logstd = tf.check_numerics(logstd, 'logstd')
  value = tf.check_numerics(value, 'value')

  policy = tf.contrib.distributions.MultivariateNormalDiag(mean,
                                                           tf.exp(logstd))

  return NetworkOutput(policy, value, lambda a: tf.clip_by_value(a, -2., 2))
