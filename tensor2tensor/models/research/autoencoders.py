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

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.maximum(input_, leak * input_, name=name)


def reverse_gradient(x):
  return -x + tf.stop_gradient(2 * x)


@registry.register_model
class AutoencoderBasic(t2t_model.T2TModel):
  """A basic autoencoder, try with image_mnist_rev or image_cifar10_rev."""

  def __init__(self, *args, **kwargs):
    super(AutoencoderBasic, self).__init__(*args, **kwargs)
    self._cur_bottleneck_tensor = None
    self.is1d = None

  def bottleneck(self, x):
    with tf.variable_scope("bottleneck"):
      hparams = self.hparams
      x = tf.layers.dense(x, hparams.bottleneck_bits, name="bottleneck")
      if hparams.mode == tf.estimator.ModeKeys.TRAIN:
        noise = 2.0 * tf.random_uniform(common_layers.shape_list(x)) - 1.0
        return tf.tanh(x) + noise * hparams.bottleneck_noise, 0.0
      return tf.tanh(x), 0.0

  def discriminator(self, x, is_training):
    """Discriminator architecture based on InfoGAN.

    Args:
      x: input images, shape [bs, h, w, channels]
      is_training: boolean, are we in train or eval model.

    Returns:
      out_logit: the output logits (before sigmoid).
    """
    hparams = self.hparams
    with tf.variable_scope(
        "discriminator",
        initializer=tf.random_normal_initializer(stddev=0.02)):
      batch_size, height, width = common_layers.shape_list(x)[:3]
      # Mapping x from [bs, h, w, c] to [bs, 1]
      net = tf.layers.conv2d(x, 64, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv1")
      # [bs, h/2, w/2, 64]
      net = lrelu(net)
      net = tf.layers.conv2d(net, 128, (4, 4), strides=(2, 2),
                             padding="SAME", name="d_conv2")
      # [bs, h/4, w/4, 128]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn2")
      net = lrelu(net)
      size = height * width
      net = tf.reshape(net, [batch_size, size * 8])  # [bs, h * w * 8]
      net = tf.layers.dense(net, 1024, name="d_fc3")  # [bs, 1024]
      if hparams.discriminator_batchnorm:
        net = tf.layers.batch_normalization(net, training=is_training,
                                            momentum=0.999, name="d_bn3")
      net = lrelu(net)
      return net

  def unbottleneck(self, x, res_size, reuse=None):
    with tf.variable_scope("unbottleneck", reuse=reuse):
      x = tf.layers.dense(x, res_size, name="dense")
      return x

  def make_even_size(self, x):
    if not self.is1d:
      return common_layers.make_even_size(x)
    shape1 = x.get_shape().as_list()[1]
    if shape1 is not None and shape1 % 2 == 0:
      return x
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2, axis=1)
    return x

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      # Down-convolutions.
      for i in range(hparams.num_hidden_layers):
        x = self.make_even_size(x)
        x = tf.layers.conv2d(
            x, hparams.hidden_size * 2**(i + 1), kernel, strides=strides,
            padding="SAME", activation=common_layers.belu, name="conv_%d" % i)
        x = common_layers.layer_norm(x)
      return x

  def decoder(self, x):
    with tf.variable_scope("decoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      # Up-convolutions.
      for i in range(hparams.num_hidden_layers):
        j = hparams.num_hidden_layers - i - 1
        x = tf.layers.conv2d_transpose(
            x, hparams.hidden_size * 2**j, kernel, strides=strides,
            padding="SAME", activation=common_layers.belu, name="deconv_%d" % j)
        x = common_layers.layer_norm(x)
      return x

  def body(self, features):
    hparams = self.hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      x = features["targets"]
      shape = common_layers.shape_list(x)
      is1d = shape[2] == 1
      self.is1d = is1d
      # Run encoder.
      x = self.encoder(x)
      # Bottleneck (mix during early training, not too important but stable).
      b, b_loss = self.bottleneck(x)
      self._cur_bottleneck_tensor = b
      b = self.unbottleneck(b, common_layers.shape_list(x)[-1])
      b = common_layers.mix(b, x, hparams.bottleneck_warmup_steps, is_training)
      if hparams.gan_loss_factor != 0.0:
        # Add a purely sampled batch on which we'll compute the GAN loss.
        g = self.unbottleneck(self.sample(), common_layers.shape_list(x)[-1],
                              reuse=True)
        b = tf.concat([g, b], axis=0)
      # With probability bottleneck_max_prob use the bottleneck, otherwise x.
      if hparams.bottleneck_max_prob < -1.0:
        x = tf.where(tf.less(tf.random_uniform([]),
                             hparams.bottleneck_max_prob), b, x)
      else:
        x = b
    else:
      if self._cur_bottleneck_tensor is None:
        b = self.sample()
      else:
        b = self._cur_bottleneck_tensor
      res_size = self.hparams.hidden_size * 2**self.hparams.num_hidden_layers
      res_size = min(res_size, hparams.max_hidden_size)
      x = self.unbottleneck(b, res_size)
    # Run decoder.
    x = self.decoder(x)
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      return x, {"bottleneck_loss": 0.0}
    # Cut to the right size and mix before returning.
    res = x[:, :shape[1], :shape[2], :]
    # Add GAN loss if requested.
    gan_loss = 0.0
    if hparams.gan_loss_factor != 0.0:
      # Split back if we added a purely sampled batch.
      res_gan, res = tf.split(res, 2, axis=0)
      num_channels = self.hparams.problem.num_channels
      res_rgb = common_layers.convert_real_to_rgb(tf.nn.sigmoid(
          tf.layers.dense(res_gan, num_channels, name="gan_rgb")))
      tf.summary.image("gan", tf.cast(res_rgb, tf.uint8), max_outputs=1)
      orig_rgb = tf.to_float(features["targets_raw"])
      def discriminate(x):
        return self.discriminator(x, is_training=is_training)
      gan_loss = common_layers.sliced_gan_loss(
          orig_rgb, reverse_gradient(res_rgb),
          discriminate, self.hparams.num_sliced_vecs)
      gan_loss *= hparams.gan_loss_factor
    # Mix the final result and return.
    res = common_layers.mix(res, features["targets"],
                            hparams.bottleneck_warmup_steps // 2, is_training)
    return res, {"bottleneck_loss": b_loss, "gan_loss": - gan_loss}

  def sample(self, features=None, shape=None):
    del features, shape
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
            hp.bottleneck_bits]
    # Sample in [-1, 1] as the bottleneck is under tanh.
    return 2.0 * tf.random_uniform(size) - 1.0

  def encode(self, x):
    """Auto-encode x and return the bottleneck."""
    features = {"targets": x}
    self(features)  # pylint: disable=not-callable
    res = tf.maximum(0.0, self._cur_bottleneck_tensor)  # Be 0/1 and not -1/1.
    self._cur_bottleneck_tensor = None
    return res

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    """Produce predictions from the model by sampling."""
    del args, kwargs
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Sample and decode.
    # TODO(lukaszkaiser): is this a universal enough way to get channels?
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    if "targets" not in features:
      features["targets"] = tf.zeros(
          [self.hparams.batch_size, 1, 1, num_channels],
          dtype=tf.int32)
    logits, _ = self(features)  # pylint: disable=not-callable
    samples = tf.argmax(logits, axis=-1)

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return samples.
    return samples

  def decode(self, bottleneck):
    """Auto-decode from the bottleneck and return the result."""
    # Get the shape from bottleneck and num channels.
    shape = common_layers.shape_list(bottleneck)
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    dummy_targets = tf.zeros(shape[:-1] + [num_channels])
    # Set the bottleneck to decode.
    if len(shape) > 4:
      bottleneck = tf.squeeze(bottleneck, axis=[1])
    bottleneck = 2 * bottleneck - 1  # Be -1/1 instead of 0/1.
    self._cur_bottleneck_tensor = bottleneck
    # Run decoding.
    res = self.infer({"targets": dummy_targets})
    self._cur_bottleneck_tensor = None
    return res

  def _get_kernel_and_strides(self):
    hparams = self.hparams
    kernel = (hparams.kernel_height, hparams.kernel_width)
    kernel = (hparams.kernel_height, 1) if self.is1d else kernel
    strides = (2, 1) if self.is1d else (2, 2)
    return (kernel, strides)


@registry.register_model
class AutoencoderAutoregressive(AutoencoderBasic):
  """Autoencoder with an autoregressive part."""

  def body(self, features):
    hparams = self.hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    # Run the basic autoencoder part first.
    basic_result, losses = super(AutoencoderAutoregressive, self).body(features)
    if hparams.autoregressive_mode == "none":
      assert not hparams.autoregressive_forget_base
      return basic_result, losses
    shape = common_layers.shape_list(basic_result)
    basic1d = tf.reshape(basic_result, [shape[0], -1, shape[3]])
    # During autoregressive inference, don't resample.
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      if hasattr(hparams, "sampled_basic1d_tensor"):
        basic1d = hparams.sampled_basic1d_tensor
      else:
        hparams.sampled_basic1d_tensor = basic1d
    # Prepare inputs for autoregressive modes.
    if common_layers.shape_list(features["targets"])[1] == 1:
      # This happens on the first step of predicitions.
      assert hparams.mode == tf.estimator.ModeKeys.PREDICT
      features["targets"] = tf.zeros_like(basic_result)
    targets_dropout = common_layers.mix(
        features["targets"], tf.zeros_like(basic_result),
        hparams.bottleneck_warmup_steps, is_training,
        max_prob=1.0 - hparams.autoregressive_dropout, broadcast_last=True)
    # Sometimes it's useful to look at non-autoregressive evals.
    if (hparams.mode == tf.estimator.ModeKeys.EVAL and
        hparams.autoregressive_eval_pure_autoencoder):
      targets_dropout = tf.zeros_like(basic_result)
    # Now combine the basic reconstruction with shifted targets.
    targets1d = tf.reshape(targets_dropout, [shape[0], -1, shape[3]])
    targets_shifted = common_layers.shift_right_3d(targets1d)
    concat1d = tf.concat([basic1d, targets_shifted], axis=-1)
    # The forget_base hparam sets purely-autoregressive mode, no autoencoder.
    if hparams.autoregressive_forget_base:
      concat1d = tf.reshape(features["targets"], [shape[0], -1, shape[3]])
      concat1d = common_layers.shift_right_3d(concat1d)
    # The autoregressive part depends on the mode.
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

  def infer(self, features, *args, **kwargs):
    """Produce predictions from the model by sampling."""
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Sample first.
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    if "targets" not in features:
      features["targets"] = tf.zeros(
          [self.hparams.batch_size, 1, 1, num_channels],
          dtype=tf.int32)
    logits, _ = self(features)  # pylint: disable=not-callable
    samples = common_layers.sample_with_temperature(
        logits, 0.0)
    shape = common_layers.shape_list(samples)

    # Sample again if requested for the autoregressive part.
    extra_samples = self.hparams.autoregressive_decode_steps
    self.hparams.autoregressive_dropout = 0.2
    for i in range(extra_samples):
      if i == extra_samples - 2:
        self.hparams.autoregressive_dropout -= 0.1
        self.hparams.sampling_temp /= 2
      if i == extra_samples - 1:
        self.hparams.autoregressive_dropout -= 0.1
        self.hparams.sampling_temp = 0.0
      features["targets"] = samples
      old_samples1d = tf.reshape(samples, [shape[0], -1, shape[3]])
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        logits, _ = self(features)  # pylint: disable=not-callable
        samples = common_layers.sample_with_temperature(
            logits, self.hparams.sampling_temp)
        samples1d = tf.reshape(samples, [shape[0], -1, shape[3]])
        samples1d = tf.concat([old_samples1d[:, :i, :], samples1d[:, i:, :]],
                              axis=1)
        samples = tf.reshape(samples1d, shape)

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return samples.
    return samples


@registry.register_model
class AutoencoderResidual(AutoencoderAutoregressive):
  """Residual autoencoder."""

  def dropout(self, x):
    if self.hparams.dropout <= 0.0:
      return x
    # For simple dropout just do this:
    # return tf.nn.dropout(x, 1.0 - self.hparams.dropout)
    is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN
    return common_layers.mix(
        tf.zeros_like(x), x,
        self.hparams.bottleneck_warmup_steps, is_training,
        max_prob=self.hparams.dropout, broadcast_last=True)

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      residual_kernel = (hparams.residual_kernel_height,
                         hparams.residual_kernel_width)
      residual_kernel1d = (hparams.residual_kernel_height, 1)
      residual_kernel = residual_kernel1d if self.is1d else residual_kernel
      residual_conv = tf.layers.conv2d
      if hparams.residual_use_separable_conv:
        residual_conv = tf.layers.separable_conv2d
      # Input embedding with a non-zero bias for uniform inputs.
      x = tf.layers.dense(
          x, hparams.hidden_size, name="embed", activation=common_layers.belu,
          bias_initializer=tf.random_normal_initializer(stddev=0.01))
      x = common_attention.add_timing_signal_nd(x)
      # Down-convolutions.
      for i in range(hparams.num_hidden_layers):
        with tf.variable_scope("layer_%d" % i):
          x = self.make_even_size(x)
          x = self.dropout(x)
          filters = hparams.hidden_size * 2**(i + 1)
          filters = min(filters, hparams.max_hidden_size)
          x = tf.layers.conv2d(
              x, filters, kernel, strides=strides,
              padding="SAME", activation=common_layers.belu, name="strided")
          y = x
          for r in range(hparams.num_residual_layers):
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
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      residual_kernel = (hparams.residual_kernel_height,
                         hparams.residual_kernel_width)
      residual_kernel1d = (hparams.residual_kernel_height, 1)
      residual_kernel = residual_kernel1d if self.is1d else residual_kernel
      residual_conv = tf.layers.conv2d
      if hparams.residual_use_separable_conv:
        residual_conv = tf.layers.separable_conv2d
      # Up-convolutions.
      for i in range(hparams.num_hidden_layers):
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
          for r in range(hparams.num_residual_layers):
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
          x = common_attention.add_timing_signal_nd(x)
      return x


@registry.register_model
class AutoencoderBasicDiscrete(AutoencoderAutoregressive):
  """Discrete autoencoder."""

  def bottleneck(self, x):
    hparams = self.hparams
    x = tf.tanh(tf.layers.dense(x, hparams.bottleneck_bits, name="bottleneck"))
    d = x + tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      noise = tf.random_uniform(common_layers.shape_list(x))
      noise = 2.0 * tf.to_float(tf.less(hparams.bottleneck_noise, noise)) - 1.0
      d *= noise
    x = common_layers.mix(d, x, hparams.discretize_warmup_steps,
                          hparams.mode == tf.estimator.ModeKeys.TRAIN)
    return x, 0.0

  def sample(self, features=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
            hp.bottleneck_bits]
    rand = tf.random_uniform(size)
    return 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0


@registry.register_model
class AutoencoderResidualDiscrete(AutoencoderResidual):
  """Discrete residual autoencoder."""

  def variance_loss(self, b):
    part = tf.random_uniform(common_layers.shape_list(b))
    selection = tf.to_float(tf.less(part, tf.random_uniform([])))
    selection_size = tf.reduce_sum(selection)
    part_avg = tf.abs(tf.reduce_sum(b * selection)) / (selection_size + 1)
    return part_avg

  def bottleneck(self, x, bottleneck_bits=None):  # pylint: disable=arguments-differ
    if bottleneck_bits is not None:
      old_bottleneck_bits = self.hparams.bottleneck_bits
      self.hparams.bottleneck_bits = bottleneck_bits
    res, loss = discretization.parametrized_bottleneck(x, self.hparams)
    if bottleneck_bits is not None:
      self.hparams.bottleneck_bits = old_bottleneck_bits
    return res, loss

  def unbottleneck(self, x, res_size, reuse=None):
    with tf.variable_scope("unbottleneck", reuse=reuse):
      return discretization.parametrized_unbottleneck(x, res_size, self.hparams)

  def sample(self, features=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
            hp.bottleneck_bits]
    rand = tf.random_uniform(size)
    res = 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0
    # If you want to set some first bits to a fixed value, do this:
    # fixed = tf.zeros_like(rand) - 1.0
    # nbits = 3
    # res = tf.concat([fixed[:, :, :, :nbits], res[:, :, :, nbits:]], axis=-1)
    return res


@registry.register_model
class AutoencoderOrderedDiscrete(AutoencoderResidualDiscrete):
  """Ordered discrete autoencoder."""

  def bottleneck(self, x):  # pylint: disable=arguments-differ
    hparams = self.hparams
    if hparams.unordered:
      return super(AutoencoderOrderedDiscrete, self).bottleneck(x)
    noise = hparams.bottleneck_noise
    hparams.bottleneck_noise = 0.0  # We'll add noise below.
    x, loss = discretization.parametrized_bottleneck(x, hparams)
    hparams.bottleneck_noise = noise
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      # We want a number p such that p^bottleneck_bits = 1 - noise.
      # So log(p) * bottleneck_bits = log(noise)
      log_p = tf.log(1 - float(noise) / 2) / float(hparams.bottleneck_bits)
      # Probabilities of flipping are p, p^2, p^3, ..., p^bottleneck_bits.
      noise_mask = 1.0 - tf.exp(tf.cumsum(tf.zeros_like(x) + log_p, axis=-1))
      # Having the no-noise mask, we can make noise just uniformly at random.
      ordered_noise = tf.random_uniform(tf.shape(x))
      # We want our noise to be 1s at the start and random {-1, 1} bits later.
      ordered_noise = tf.to_float(tf.less(noise_mask, ordered_noise))
      # Now we flip the bits of x on the noisy positions (ordered and normal).
      x *= 2.0 * ordered_noise - 1
    return x, loss


@registry.register_model
class AutoencoderStacked(AutoencoderResidualDiscrete):
  """A stacked autoencoder."""

  def stack(self, b, size, bottleneck_bits, name):
    with tf.variable_scope(name + "_stack"):
      unb = self.unbottleneck(b, size)
      enc = self.encoder(unb)
      b, _ = self.bottleneck(enc, bottleneck_bits=bottleneck_bits)
      return b

  def unstack(self, b, size, bottleneck_bits, name):
    with tf.variable_scope(name + "_unstack"):
      unb = self.unbottleneck(b, size)
      dec = self.decoder(unb)
      pred = tf.layers.dense(dec, bottleneck_bits, name="pred")
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

  def full_stack(self, b, x_size, bottleneck_bits, losses, is_training, i):
    stack1_b = self.stack(b, x_size, bottleneck_bits, "step%d" % i)
    if i > 1:
      stack1_b = self.full_stack(stack1_b, 2 * x_size, 2 * bottleneck_bits,
                                 losses, is_training, i - 1)
    b1, b_pred = self.unstack(stack1_b, x_size, bottleneck_bits, "step%d" % i)
    losses["stack%d_loss" % i] = self.stack_loss(b, b_pred, "step%d" % i)
    b_shape = common_layers.shape_list(b)
    if is_training:
      b1 = tf.cond(tf.less(tf.random_uniform([]), 0.5),
                   lambda: b, lambda: b1)
    return tf.reshape(b1, b_shape)

  def body(self, features):
    hparams = self.hparams
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
      b, b_loss = self.bottleneck(x)
      losses = {"bottleneck0_loss": b_loss}
      b = self.full_stack(b, 2 * x_size, 2 * hparams.bottleneck_bits,
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
      res_size = self.hparams.hidden_size * 2**self.hparams.num_hidden_layers
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
def autoencoder_basic():
  """Basic autoencoder model."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "Adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.0
  hparams.batch_size = 128
  hparams.hidden_size = 64
  hparams.num_hidden_layers = 5
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.dropout = 0.1
  hparams.add_hparam("max_hidden_size", 1024)
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("bottleneck_noise", 0.1)
  hparams.add_hparam("bottleneck_warmup_steps", 3000)
  hparams.add_hparam("bottleneck_max_prob", 1.0)
  hparams.add_hparam("sample_height", 32)
  hparams.add_hparam("sample_width", 32)
  hparams.add_hparam("discriminator_batchnorm", True)
  hparams.add_hparam("num_sliced_vecs", 4096)
  hparams.add_hparam("gan_loss_factor", 0.0)
  return hparams


@registry.register_hparams
def autoencoder_autoregressive():
  """Autoregressive autoencoder model."""
  hparams = autoencoder_basic()
  hparams.add_hparam("autoregressive_forget_base", False)
  hparams.add_hparam("autoregressive_mode", "none")
  hparams.add_hparam("autoregressive_dropout", 0.4)
  hparams.add_hparam("autoregressive_decode_steps", 0)
  hparams.add_hparam("autoregressive_eval_pure_autoencoder", False)
  return hparams


@registry.register_hparams
def autoencoder_residual():
  """Residual autoencoder model."""
  hparams = autoencoder_autoregressive()
  hparams.optimizer = "Adafactor"
  hparams.clip_grad_norm = 1.0
  hparams.learning_rate_constant = 0.5
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup * rsqrt_decay"
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
  hparams.bottleneck_bits = 4096
  hparams.bottleneck_noise = 0.1
  hparams.bottleneck_warmup_steps = 3000
  hparams.add_hparam("discretize_warmup_steps", 5000)
  return hparams


@registry.register_hparams
def autoencoder_residual_discrete():
  """Residual discrete autoencoder model."""
  hparams = autoencoder_residual()
  hparams.bottleneck_bits = 4096
  hparams.bottleneck_noise = 0.1
  hparams.bottleneck_warmup_steps = 3000
  hparams.add_hparam("discretize_warmup_steps", 5000)
  hparams.add_hparam("bottleneck_kind", "tanh_discrete")
  hparams.add_hparam("isemhash_noise_dev", 0.5)
  hparams.add_hparam("isemhash_mix_prob", 0.5)
  hparams.add_hparam("isemhash_filter_size_multiplier", 2.0)
  hparams.add_hparam("vq_beta", 0.25)
  hparams.add_hparam("vq_decay", 0.999)
  hparams.add_hparam("vq_epsilon", 1e-5)
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
  """Ordered discrete autoencoder model."""
  hparams = autoencoder_residual_discrete()
  hparams.bottleneck_noise = 1.0
  hparams.gan_loss_factor = 0.0
  hparams.dropout = 0.1
  hparams.residual_dropout = 0.3
  hparams.add_hparam("unordered", False)
  return hparams


@registry.register_hparams
def autoencoder_ordered_text():
  """Ordered discrete autoencoder model for text."""
  hparams = autoencoder_ordered_discrete()
  hparams.learning_rate_constant = 2.0
  hparams.learning_rate_warmup_steps = 2000
  hparams.bottleneck_bits = 1024
  hparams.batch_size = 2048
  hparams.autoregressive_mode = "sru"
  hparams.hidden_size = 256
  hparams.max_hidden_size = 4096
  hparams.bottleneck_warmup_steps = 10000
  hparams.discretize_warmup_steps = 15000
  return hparams


@registry.register_hparams
def autoencoder_ordered_discrete_vq():
  """Ordered discrete autoencoder model with VQ bottleneck."""
  hparams = autoencoder_ordered_discrete()
  hparams.bottleneck_kind = "vq"
  hparams.bottleneck_bits = 16
  return hparams


@registry.register_hparams
def autoencoder_discrete_pong():
  """Discrete autoencoder model for compressing pong frames."""
  hparams = autoencoder_ordered_discrete()
  hparams.num_hidden_layers = 2
  hparams.bottleneck_bits = 24
  hparams.dropout = 0.1
  hparams.batch_size = 2
  hparams.bottleneck_noise = 0.2
  hparams.max_hidden_size = 1024
  hparams.unordered = True
  return hparams


@registry.register_hparams
def autoencoder_discrete_cifar():
  """Discrete autoencoder model for compressing cifar."""
  hparams = autoencoder_ordered_discrete()
  hparams.bottleneck_noise = 0.0
  hparams.bottleneck_bits = 90
  hparams.unordered = True
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.num_residual_layers = 4
  hparams.batch_size = 32
  hparams.learning_rate_constant = 1.0
  hparams.dropout = 0.1
  return hparams


@registry.register_ranged_hparams
def autoencoder_discrete_pong_range(rhp):
  """Narrow tuning grid."""
  rhp.set_float("dropout", 0.0, 0.2)
  rhp.set_discrete("max_hidden_size", [1024, 2048])


@registry.register_hparams
def autoencoder_stacked():
  """Stacked autoencoder model."""
  hparams = autoencoder_residual_discrete()
  hparams.bottleneck_bits = 128
  return hparams
