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
from tensor2tensor.layers import discretization
from tensor2tensor.models import basic
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class AutoencoderAutoregressive(basic.BasicAutoencoder):
  """Autoencoder with an autoregressive part."""

  def body(self, features):
    hparams = self._hparams
    shape = common_layers.shape_list(features["targets"])
    # Run the basic autoencoder part first.
    basic_result, losses = super(AutoencoderAutoregressive, self).body(features)
    # Prepare inputs for autoregressive modes.
    targets_keep_prob = 1.0 - hparams.autoregressive_dropout
    targets_dropout = common_layers.dropout_with_broadcast_dims(
        features["targets"], targets_keep_prob, broadcast_dims=[-1])
    targets1d = tf.reshape(targets_dropout, [shape[0], -1, shape[3]])
    targets_shifted = common_layers.shift_right_3d(targets1d)
    basic1d = tf.reshape(basic_result, [shape[0], -1, shape[3]])
    concat1d = tf.concat([basic1d, targets_shifted], axis=-1)
    # The forget_base hparam sets purely-autoregressive mode, no autoencoder.
    if hparams.autoregressive_forget_base:
      concat1d = tf.reshape(features["targets"], [shape[0], -1, shape[3]])
      concat1d = common_layers.shift_right_3d(concat1d)
    # The autoregressive part depends on the mode.
    if hparams.autoregressive_mode == "none":
      assert not hparams.autoregressive_forget_base
      return basic_result, losses
    if hparams.autoregressive_mode == "conv3":
      res = common_layers.conv1d(concat1d, shape[3], 3, padding="LEFT",
                                 activation=common_layers.belu,
                                 name="autoregressive_conv3")
      return tf.reshape(res, shape), losses
    if hparams.autoregressive_mode == "conv5":
      res = common_layers.conv1d(concat1d, shape[3], 5, padding="LEFT",
                                 activation=common_layers.belu,
                                 name="autoregressive_conv5")
      return tf.reshape(res, shape), losses
    if hparams.autoregressive_mode == "sru":
      res = common_layers.conv1d(concat1d, shape[3], 3, padding="LEFT",
                                 activation=common_layers.belu,
                                 name="autoregressive_sru_conv3")
      res = common_layers.sru(res)
      return tf.reshape(res, shape), losses

    raise ValueError("Unsupported autoregressive mode: %s"
                     % hparams.autoregressive_mode)


@registry.register_model
class AutoencoderResidual(AutoencoderAutoregressive):
  """Residual autoencoder."""

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self._hparams
      kernel, strides = self._get_kernel_and_strides()
      residual_kernel = (hparams.residual_kernel_height,
                         hparams.residual_kernel_width)
      residual_kernel1d = (hparams.residual_kernel_height, 1)
      residual_kernel = residual_kernel1d if self.is1d else residual_kernel
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
      residual_kernel = (hparams.residual_kernel_height,
                         hparams.residual_kernel_width)
      residual_kernel1d = (hparams.residual_kernel_height, 1)
      residual_kernel = residual_kernel1d if self.is1d else residual_kernel
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
class AutoencoderBasicDiscrete(AutoencoderAutoregressive):
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
class AutoencoderResidualDiscrete(AutoencoderResidual):
  """Discrete residual autoencoder."""

  def bottleneck(self, x, bottleneck_size=None):
    if bottleneck_size is not None:
      old_bottleneck_size = self._hparams.bottleneck_size
      self._hparams.bottleneck_size = bottleneck_size
    res = discretization.parametrized_bottleneck(x, self._hparams)
    if bottleneck_size is not None:
      self._hparams.bottleneck_size = old_bottleneck_size
    return res

  def unbottleneck(self, x, res_size):
    return discretization.parametrized_unbottleneck(x, res_size, self._hparams)

  def bottleneck_loss(self, b):
    part = tf.random_uniform(common_layers.shape_list(b))
    selection = tf.to_float(tf.less(part, tf.random_uniform([])))
    part_avg = tf.abs(tf.reduce_sum(b * selection)) / tf.reduce_sum(selection)
    return part_avg

  def sample(self):
    hp = self._hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
            hp.bottleneck_size]
    rand = tf.random_uniform(size)
    res = 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0
    # If you want to set some first bits to a fixed value, do this:
    # fixed = tf.zeros_like(rand) - 1.0
    # res = tf.concat([fixed[:, :, :, :2], res[:, :, :, 2:]], axis=-1)
    return res


@registry.register_model
class AutoencoderOrderedDiscrete(AutoencoderResidualDiscrete):
  """Ordered discrete autoencoder."""

  def bottleneck(self, x):
    hparams = self._hparams
    x = discretization.parametrized_bottleneck(x, hparams)
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
      ordered_noise = 2.0 * tf.to_float(tf.less(ordered_noise, 0.5)) - 1.0
      # Now we flip the bits of x on the noisy positions (ordered and normal).
      x *= ordered_noise
    return x


@registry.register_model
class AutoencoderStacked(AutoencoderResidualDiscrete):
  """A stacked autoencoder."""

  def stack(self, b, size, bottleneck_size, name):
    with tf.variable_scope(name + "_stack"):
      unb = self.unbottleneck(b, size)
      enc = self.encoder(unb)
      return self.bottleneck(enc, bottleneck_size=bottleneck_size)

  def unstack(self, b, size, bottleneck_size, name):
    with tf.variable_scope(name + "_unstack"):
      unb = self.unbottleneck(b, size)
      dec = self.decoder(unb)
      pred = tf.layers.dense(dec, bottleneck_size, name="pred")
      pred_shape = common_layers.shape_list(pred)
      pred1 = tf.reshape(pred, pred_shape[:-1] + [-1, 2])
      x, y = tf.split(pred1, 2, axis=-1)
      x = tf.squeeze(x, axis=[-1])
      y = tf.squeeze(y, axis=[-1])
      gt = 2.0 * tf.to_float(tf.less(x, y)) - 1.0
      gtc = tf.tanh(y - x)
      gt += gtc - tf.stop_gradient(gtc)
      return gt, pred1

  def stack_loss(self, b, b_pred, name):
    with tf.variable_scope(name):
      labels_discrete = tf.to_int32((b + 1.0) * 0.5)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels_discrete, logits=b_pred)
      return tf.reduce_mean(loss)

  def full_stack(self, b, x_size, bottleneck_size, losses, is_training, i):
    stack1_b = self.stack(b, x_size, bottleneck_size, "step%d" % i)
    if i > 1:
      stack1_b = self.full_stack(stack1_b, 2 * x_size, 2 * bottleneck_size,
                                 losses, is_training, i - 1)
    b1, b_pred = self.unstack(stack1_b, x_size, bottleneck_size, "step%d" % i)
    losses["bottleneck%d_loss" % i] = self.bottleneck_loss(stack1_b)
    losses["stack%d_loss" % i] = self.stack_loss(b, b_pred, "step%d" % i)
    b_shape = common_layers.shape_list(b)
    if is_training:
      b1 = tf.cond(tf.less(tf.random_uniform([]), 0.5),
                   lambda: b, lambda: b1)
    return tf.reshape(b1, b_shape)

  def body(self, features):
    hparams = self._hparams
    num_stacks = hparams.num_hidden_layers
    hparams.num_hidden_layers = 1
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      x = features["targets"]
      shape = common_layers.shape_list(x)
      is1d = shape[2] == 1
      self.is1d = is1d
      x, _ = common_layers.pad_to_same_length(
          x, x, final_length_divisible_by=2**num_stacks, axis=1)
      if not is1d:
        x, _ = common_layers.pad_to_same_length(
            x, x, final_length_divisible_by=2**num_stacks, axis=2)
      # Run encoder.
      x = self.encoder(x)
      x_size = common_layers.shape_list(x)[-1]
      # Bottleneck (mix during early training, not too important but stable).
      b = self.bottleneck(x)
      b_loss = self.bottleneck_loss(b)
      losses = {"bottleneck0_loss": b_loss}
      b = self.full_stack(b, 2 * x_size, 2 * hparams.bottleneck_size,
                          losses, is_training, num_stacks - 1)
      b = self.unbottleneck(b, x_size)
      b = common_layers.mix(b, x, hparams.bottleneck_warmup_steps, is_training)
      # With probability bottleneck_max_prob use the bottleneck, otherwise x.
      if hparams.bottleneck_max_prob < 1.0:
        x = tf.where(tf.less(tf.random_uniform([]),
                             hparams.bottleneck_max_prob), b, x)
      else:
        x = b
    else:
      b = self.sample()
      res_size = self._hparams.hidden_size * 2**self._hparams.num_hidden_layers
      res_size = min(res_size, hparams.max_hidden_size)
      x = self.unbottleneck(b, res_size)
    # Run decoder.
    x = self.decoder(x)
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      return x
    # Cut to the right size and mix before returning.
    res = x[:, :shape[1], :shape[2], :]
    res = common_layers.mix(res, features["targets"],
                            hparams.bottleneck_warmup_steps // 2, is_training)
    hparams.num_hidden_layers = num_stacks
    return res, losses


@registry.register_hparams
def autoencoder_autoregressive():
  """Autoregressive autoencoder model."""
  hparams = basic.basic_autoencoder()
  hparams.add_hparam("autoregressive_forget_base", False)
  hparams.add_hparam("autoregressive_mode", "conv3")
  hparams.add_hparam("autoregressive_dropout", 0.4)
  return hparams


@registry.register_hparams
def autoencoder_residual():
  """Residual autoencoder model."""
  hparams = autoencoder_autoregressive()
  hparams.optimizer = "Adam"
  hparams.learning_rate_constant = 0.0001
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.dropout = 0.05
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.max_hidden_size = 1024
  hparams.add_hparam("num_residual_layers", 2)
  hparams.add_hparam("residual_kernel_height", 3)
  hparams.add_hparam("residual_kernel_width", 3)
  hparams.add_hparam("residual_filter_multiplier", 2.0)
  hparams.add_hparam("residual_dropout", 0.2)
  hparams.add_hparam("residual_use_separable_conv", int(True))
  return hparams


@registry.register_hparams
def autoencoder_basic_discrete():
  """Basic autoencoder model."""
  hparams = autoencoder_autoregressive()
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.bottleneck_size = 4096
  hparams.bottleneck_noise = 0.1
  hparams.bottleneck_warmup_steps = 3000
  hparams.add_hparam("discretize_warmup_steps", 5000)
  return hparams


@registry.register_hparams
def autoencoder_residual_discrete():
  """Residual discrete autoencoder model."""
  hparams = autoencoder_residual()
  hparams.bottleneck_size = 4096
  hparams.bottleneck_noise = 0.1
  hparams.bottleneck_warmup_steps = 3000
  hparams.add_hparam("discretize_warmup_steps", 5000)
  hparams.add_hparam("bottleneck_kind", "tanh_discrete")
  hparams.add_hparam("isemhash_noise_dev", 0.5)
  hparams.add_hparam("isemhash_mix_prob", 0.5)
  hparams.add_hparam("isemhash_filter_size_multiplier", 2.0)
  return hparams


@registry.register_hparams
def autoencoder_residual_discrete_big():
  """Residual discrete autoencoder model, big version."""
  hparams = autoencoder_residual_discrete()
  hparams.hidden_size = 128
  hparams.max_hidden_size = 4096
  hparams.bottleneck_noise = 0.1
  hparams.dropout = 0.1
  hparams.residual_dropout = 0.4
  return hparams


@registry.register_hparams
def autoencoder_ordered_discrete():
  """Basic autoencoder model."""
  hparams = autoencoder_residual_discrete()
  return hparams


@registry.register_hparams
def autoencoder_stacked():
  """Stacked autoencoder model."""
  hparams = autoencoder_residual_discrete()
  hparams.bottleneck_size = 128
  return hparams
