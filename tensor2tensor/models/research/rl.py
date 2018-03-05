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

"""Reinforcement learning models and parameters."""

import collections
import functools
import operator

# Dependency imports

import gym
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_hparams
def ppo_base_v1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.learning_rate = 1e-4
  hparams.add_hparam("init_mean_factor", 0.1)
  hparams.add_hparam("init_logstd", 0.1)
  hparams.add_hparam("policy_layers", (100, 100))
  hparams.add_hparam("value_layers", (100, 100))
  hparams.add_hparam("num_agents", 30)
  hparams.add_hparam("clipping_coef", 0.2)
  hparams.add_hparam("gae_gamma", 0.99)
  hparams.add_hparam("gae_lambda", 0.95)
  hparams.add_hparam("entropy_loss_coef", 0.01)
  hparams.add_hparam("value_loss_coef", 1)
  hparams.add_hparam("optimization_epochs", 15)
  hparams.add_hparam("epoch_length", 200)
  hparams.add_hparam("epochs_num", 2000)
  hparams.add_hparam("eval_every_epochs", 10)
  hparams.add_hparam("num_eval_agents", 3)
  hparams.add_hparam("video_during_eval", True)
  hparams.add_hparam("save_models_every_epochs", 30)
  return hparams


@registry.register_hparams
def continuous_action_base():
  hparams = ppo_base_v1()
  hparams.add_hparam("network", feed_forward_gaussian_fun)
  return hparams


@registry.register_hparams
def discrete_action_base():
  hparams = ppo_base_v1()
  hparams.add_hparam("network", feed_forward_categorical_fun)
  return hparams


@registry.register_hparams
def atari_base():
  """Atari base parameters."""
  hparams = discrete_action_base()
  hparams.learning_rate = 16e-5
  hparams.num_agents = 5
  hparams.epoch_length = 200
  hparams.gae_gamma = 0.985
  hparams.gae_lambda = 0.985
  hparams.entropy_loss_coef = 0.002
  hparams.value_loss_coef = 0.025
  hparams.optimization_epochs = 10
  hparams.epochs_num = 10000
  hparams.num_eval_agents = 1
  hparams.network = feed_forward_cnn_small_categorical_fun
  return hparams


NetworkOutput = collections.namedtuple(
    "NetworkOutput", "policy, value, action_postprocessing")


def feed_forward_gaussian_fun(action_space, config, observations):
  """Feed-forward Gaussian."""
  if not isinstance(action_space, gym.spaces.box.Box):
    raise ValueError("Expecting continuous action space.")

  mean_weights_initializer = tf.contrib.layers.variance_scaling_initializer(
      factor=config.init_mean_factor)
  logstd_initializer = tf.random_normal_initializer(config.init_logstd, 1e-10)

  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])

  with tf.variable_scope("network_parameters"):
    with tf.variable_scope("policy"):
      x = flat_observations
      for size in config.policy_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      mean = tf.contrib.layers.fully_connected(
          x, action_space.shape[0], tf.tanh,
          weights_initializer=mean_weights_initializer)
      logstd = tf.get_variable(
          "logstd", mean.shape[2:], tf.float32, logstd_initializer)
      logstd = tf.tile(
          logstd[None, None],
          [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
    with tf.variable_scope("value"):
      x = flat_observations
      for size in config.value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  mean = tf.check_numerics(mean, "mean")
  logstd = tf.check_numerics(logstd, "logstd")
  value = tf.check_numerics(value, "value")

  policy = tf.contrib.distributions.MultivariateNormalDiag(mean,
                                                           tf.exp(logstd))

  return NetworkOutput(policy, value, lambda a: tf.clip_by_value(a, -2., 2))


def feed_forward_categorical_fun(action_space, config, observations):
  """Feed-forward categorical."""
  if not isinstance(action_space, gym.spaces.Discrete):
    raise ValueError("Expecting discrete action space.")
  flat_observations = tf.reshape(observations, [
      tf.shape(observations)[0], tf.shape(observations)[1],
      functools.reduce(operator.mul, observations.shape.as_list()[2:], 1)])
  with tf.variable_scope("network_parameters"):
    with tf.variable_scope("policy"):
      x = flat_observations
      for size in config.policy_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      logits = tf.contrib.layers.fully_connected(x, action_space.n,
                                                 activation_fn=None)
    with tf.variable_scope("value"):
      x = flat_observations
      for size in config.value_layers:
        x = tf.contrib.layers.fully_connected(x, size, tf.nn.relu)
      value = tf.contrib.layers.fully_connected(x, 1, None)[..., 0]
  policy = tf.contrib.distributions.Categorical(logits=logits)
  return NetworkOutput(policy, value, lambda a: a)


def feed_forward_cnn_small_categorical_fun(action_space, config, observations):
  """Small cnn network with categorical output."""
  del config

  obs_shape = observations.shape.as_list()
  x = tf.reshape(observations, [-1] + obs_shape[2:])

  with tf.variable_scope("network_parameters"):
    x = tf.to_float(x) / 255.0
    x = tf.contrib.layers.conv2d(x, 32, [5, 5], [2, 2],
                                 activation_fn=tf.nn.relu, padding="SAME")
    x = tf.contrib.layers.conv2d(x, 32, [5, 5], [2, 2],
                                 activation_fn=tf.nn.relu, padding="SAME")

    flat_x = tf.reshape(
        x, [tf.shape(observations)[0], tf.shape(observations)[1],
            functools.reduce(operator.mul, x.shape.as_list()[1:], 1)])

    x = tf.contrib.layers.fully_connected(flat_x, 128, tf.nn.relu)
    logits = tf.contrib.layers.fully_connected(x, action_space.n,
                                               activation_fn=None)

    value = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)[..., 0]
    policy = tf.contrib.distributions.Categorical(logits=logits)

  return NetworkOutput(policy, value, lambda a: a)
