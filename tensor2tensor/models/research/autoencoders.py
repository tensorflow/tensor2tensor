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

"""Autoencoders."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_layers
from tensor2tensor.models import basic
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class ResidualAutoencoder(basic.BasicAutoencoder):
  """Residual autoencoder."""

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self._hparams
      kernel, strides = self._get_kernel_and_strides()
      residual_kernel = (3, 1) if self.is1d else (3, 3)
      residual_conv = tf.layers.conv2d
      if hparams.residual_use_separable_conv:
        residual_conv = tf.layers.separable_conv2d
      # Down-convolutions.
      for i in xrange(hparams.num_hidden_layers):
        with tf.variable_scope("layer_%d" % i):
          x = tf.nn.dropout(x, 1.0 - hparams.dropout)
          filters = hparams.hidden_size * 2**(i + 1)
          filters = min(filters, hparams.max_hidden_size)
          x = tf.layers.conv2d(
              x, filters, kernel, strides=strides,
              padding="SAME", activation=common_layers.belu, name="strided")
          y = x
          for r in xrange(hparams.num_residual_layers):
            residual_filters = filters
            if r < hparams.num_residual_layers - 1:
              residual_filters = int(
                  filters * hparams.residual_filter_multiplier)
            y = residual_conv(
                y, residual_filters, residual_kernel,
                padding="SAME", activation=common_layers.belu,
                name="residual_%d" % r)
          x += tf.nn.dropout(y, 1.0 - hparams.residual_dropout)
          x = common_layers.layer_norm(x)
      return x

  def decoder(self, x):
    with tf.variable_scope("decoder"):
      hparams = self._hparams
      kernel, strides = self._get_kernel_and_strides()
      residual_kernel = (3, 1) if self.is1d else (3, 3)
      residual_conv = tf.layers.conv2d
      if hparams.residual_use_separable_conv:
        residual_conv = tf.layers.separable_conv2d
      # Up-convolutions.
      for i in xrange(hparams.num_hidden_layers):
        x = tf.nn.dropout(x, 1.0 - hparams.dropout)
        j = hparams.num_hidden_layers - i - 1
        filters = hparams.hidden_size * 2**j
        filters = min(filters, hparams.max_hidden_size)
        with tf.variable_scope("layer_%d" % i):
          j = hparams.num_hidden_layers - i - 1
          filters = hparams.hidden_size * 2**j
          x = tf.layers.conv2d_transpose(
              x, filters, kernel, strides=strides,
              padding="SAME", activation=common_layers.belu, name="strided")
          y = x
          for r in xrange(hparams.num_residual_layers):
            residual_filters = filters
            if r < hparams.num_residual_layers - 1:
              residual_filters = int(
                  filters * hparams.residual_filter_multiplier)
            y = residual_conv(
                y, residual_filters, residual_kernel,
                padding="SAME", activation=common_layers.belu,
                name="residual_%d" % r)
          x += tf.nn.dropout(y, 1.0 - hparams.residual_dropout)
          x = common_layers.layer_norm(x)
      return x


@registry.register_model
class BasicDiscreteAutoencoder(basic.BasicAutoencoder):
  """Discrete autoencoder."""

  def bottleneck(self, x):
    hparams = self._hparams
    x = tf.tanh(tf.layers.dense(x, hparams.bottleneck_size, name="bottleneck"))
    d = x + tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      noise = tf.random_uniform(common_layers.shape_list(x))
      noise = 2.0 * tf.to_float(tf.less(hparams.bottleneck_noise, noise)) - 1.0
      d *= noise
    x = common_layers.mix(d, x, hparams.discretize_warmup_steps,
                          hparams.mode == tf.estimator.ModeKeys.TRAIN)
    return x

  def sample(self):
    hp = self._hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
            hp.bottleneck_size]
    rand = tf.random_uniform(size)
    return 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0


@registry.register_model
class OrderedDiscreteAutoencoder(BasicDiscreteAutoencoder):
  """Ordered discrete autoencoder."""

  def bottleneck(self, x):
    hparams = self._hparams
    x = tf.tanh(tf.layers.dense(x, hparams.bottleneck_size, name="bottleneck"))
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      # In the ordered case, we'll have no noise on top bits, let's make a mask.
      # Start with randomly uniformly choosing numbers [0, number_of_bits) where
      # the number of bits in our case is bottleneck size. We pick separately
      # for every position and batch just to keep it varied.
      no_noise_mask = tf.random_uniform(common_layers.shape_list(x)[:-1])
      no_noise_mask *= hparams.bottleneck_size
      # Now let's make a 1-hot vector that is 1 on the index i from which on
      # we want to be noisy and 0 everywhere else.
      no_noise_mask = tf.one_hot(tf.to_int32(no_noise_mask),
                                 hparams.bottleneck_size)
      # Use tf.cumsum to make the mask (0 before index i, 1 after index i).
      no_noise_mask = tf.cumsum(no_noise_mask, axis=-1)
      # Having the no-noise mask, we can make noise just uniformly at random.
      ordered_noise = tf.random_uniform(tf.shape(x)) * no_noise_mask
      # We want our noise to be 1s at the start and random {-1, 1} bits later.
      ordered_noise = 2.0 * tf.to_float(tf.less(ordered_noise, 0.5))- 1.0
      # Now we flip the bits of x on the noisy positions (ordered and normal).
      noise = tf.random_uniform(common_layers.shape_list(x))
      noise = 2.0 * tf.to_float(tf.less(hparams.bottleneck_noise, noise)) - 1.0
      x *= ordered_noise * noise
    # Discretize as before.
    d = x + tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
    x = common_layers.mix(d, x, hparams.discretize_warmup_steps,
                          hparams.mode == tf.estimator.ModeKeys.TRAIN)
    return x


@registry.register_hparams
def residual_autoencoder():
  """Residual autoencoder model."""
  hparams = basic.basic_autoencoder()
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_constant = 0.001
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.dropout = 0.1
  hparams.add_hparam("max_hidden_size", 2048)
  hparams.add_hparam("num_residual_layers", 2)
  hparams.add_hparam("residual_filter_multiplier", 2.0)
  hparams.add_hparam("residual_dropout", 0.3)
  hparams.add_hparam("residual_use_separable_conv", int(True))
  return hparams


@registry.register_hparams
def basic_discrete_autoencoder():
  """Basic autoencoder model."""
  hparams = basic.basic_autoencoder()
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.bottleneck_size = 2048
  hparams.bottleneck_noise = 0.2
  hparams.bottleneck_warmup_steps = 3000
  hparams.add_hparam("discretize_warmup_steps", 5000)
  return hparams


@registry.register_hparams
def ordered_discrete_autoencoder():
  """Basic autoencoder model."""
  hparams = basic.basic_autoencoder()
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.bottleneck_size = 4096
  hparams.bottleneck_noise = 0.2
  hparams.bottleneck_warmup_steps = 3000
  hparams.add_hparam("discretize_warmup_steps", 5000)
  return hparams
