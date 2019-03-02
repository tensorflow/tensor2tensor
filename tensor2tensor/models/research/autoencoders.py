# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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
from tensor2tensor.layers import latent_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def reverse_gradient(x, lr=1.0):
  return -lr * x + tf.stop_gradient((1.0 + lr) * x)


def time_to_channels(embedded_video):
  """Put time dimension on channels in an embedded video."""
  video_shape = common_layers.shape_list(embedded_video)
  if len(video_shape) != 5:
    raise ValueError("Assuming videos given as tensors in the format "
                     "[batch, time, height, width, channels] but got one "
                     "of shape: %s" % str(video_shape))
  transposed = tf.transpose(embedded_video, [0, 2, 3, 1, 4])
  return tf.reshape(transposed, [
      video_shape[0], video_shape[2], video_shape[3],
      video_shape[1] * video_shape[4]
  ])


@registry.register_model
class AutoencoderBasic(t2t_model.T2TModel):
  """A basic autoencoder, try with image_mnist_rev or image_cifar10_rev."""

  def __init__(self, *args, **kwargs):
    super(AutoencoderBasic, self).__init__(*args, **kwargs)
    self._cur_bottleneck_tensor = None
    self.is1d = None
    self._encode_on_predict = False

  @property
  def num_channels(self):
    # TODO(lukaszkaiser): is this a universal enough way to get channels?
    try:
      num_channels = self.hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    return num_channels

  def image_summary(self, name, image_logits, max_outputs=1):
    """Helper for image summaries that are safe on TPU."""
    if len(image_logits.get_shape()) != 5:
      tf.logging.info("Not generating image summary, maybe not an image.")
      return
    return tf.summary.image(
        name,
        common_layers.tpu_safe_image_summary(tf.argmax(image_logits, -1)),
        max_outputs=max_outputs)

  def embed(self, x, name="embedding"):
    """Input embedding with a non-zero bias for uniform inputs."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      x_shape = common_layers.shape_list(x)
      # Merge channels and depth before embedding.
      x = tf.reshape(x, x_shape[:-2] + [x_shape[-2] * x_shape[-1]])
      x = tf.layers.dense(
          x,
          self.hparams.hidden_size,
          name="embed",
          activation=common_layers.belu,
          bias_initializer=tf.random_normal_initializer(stddev=0.01))
      x = common_layers.layer_norm(x, name="ln_embed")
      return common_attention.add_timing_signal_nd(x)

  def bottleneck(self, x):
    with tf.variable_scope("bottleneck"):
      hparams = self.hparams
      x = tf.layers.dense(x, hparams.bottleneck_bits, name="bottleneck")
      if hparams.mode == tf.estimator.ModeKeys.TRAIN:
        noise = 2.0 * tf.random_uniform(common_layers.shape_list(x)) - 1.0
        return tf.tanh(x) + noise * hparams.bottleneck_noise, 0.0
      return tf.tanh(x), 0.0

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
      layers = []
      kernel, strides = self._get_kernel_and_strides()
      # Down-convolutions.
      for i in range(hparams.num_hidden_layers):
        x = self.make_even_size(x)
        layers.append(x)
        x = tf.layers.conv2d(
            x,
            hparams.hidden_size * 2**(i + 1),
            kernel,
            strides=strides,
            padding="SAME",
            activation=common_layers.belu,
            name="conv_%d" % i)
        x = common_layers.layer_norm(x, name="ln_%d" % i)
      return x, layers

  def decoder(self, x, encoder_layers):
    del encoder_layers
    with tf.variable_scope("decoder"):
      hparams = self.hparams
      kernel, strides = self._get_kernel_and_strides()
      # Up-convolutions.
      for i in range(hparams.num_hidden_layers):
        j = hparams.num_hidden_layers - i - 1
        x = tf.layers.conv2d_transpose(
            x,
            hparams.hidden_size * 2**j,
            kernel,
            strides=strides,
            padding="SAME",
            activation=common_layers.belu,
            name="deconv_%d" % j)
        x = common_layers.layer_norm(x, name="ln_%d" % i)
      return x

  def gumbel_sample(self, reconstr_gan):
    hparams = self.hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    vocab_size = self._problem_hparams.vocab_size["targets"]
    if hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    reconstr_gan = tf.nn.log_softmax(reconstr_gan)
    if is_training and hparams.gumbel_temperature > 0.0:
      gumbel_samples = discretization.gumbel_sample(
          common_layers.shape_list(reconstr_gan))
      gumbel_samples *= hparams.gumbel_noise_factor
      reconstr_gan += gumbel_samples
      reconstr_sample = latent_layers.multinomial_sample(
          reconstr_gan, temperature=hparams.gumbel_temperature)
      reconstr_gan = tf.nn.softmax(reconstr_gan / hparams.gumbel_temperature)
    else:
      reconstr_sample = tf.argmax(reconstr_gan, axis=-1)
      reconstr_gan = tf.nn.softmax(reconstr_gan / 0.1)  # Sharpen a bit.
    # Use 1-hot forward, softmax backward.
    reconstr_hot = tf.one_hot(reconstr_sample, vocab_size)
    reconstr_gan += reconstr_hot - tf.stop_gradient(reconstr_gan)
    return reconstr_gan

  def body(self, features):
    hparams = self.hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    vocab_size = self._problem_hparams.vocab_size["targets"]
    if hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    encoder_layers = None
    self.is1d = hparams.sample_width == 1
    if (hparams.mode != tf.estimator.ModeKeys.PREDICT
        or self._encode_on_predict):
      labels = features["targets_raw"]
      labels_shape = common_layers.shape_list(labels)
      # handle videos
      if len(labels.shape) == 5:
        labels = time_to_channels(labels)
      shape = common_layers.shape_list(labels)
      x = tf.one_hot(labels, vocab_size)
      x = self.embed(x)
      target_codes = x
      if shape[2] == 1:
        self.is1d = True
      # Run encoder.
      x, encoder_layers = self.encoder(x)
      # Bottleneck.
      b, b_loss = self.bottleneck(x)
      xb_loss = 0.0
      b_shape = common_layers.shape_list(b)
      self._cur_bottleneck_tensor = b
      res_size = common_layers.shape_list(x)[-1]
      b = self.unbottleneck(b, res_size)
      if not is_training:
        x = b
      else:
        l = 2**hparams.num_hidden_layers
        warm_step = int(hparams.bottleneck_warmup_steps * 0.25 * l)
        nomix_p = common_layers.inverse_lin_decay(warm_step) + 0.01
        if common_layers.should_generate_summaries():
          tf.summary.scalar("nomix_p_bottleneck", nomix_p)
        rand = tf.random_uniform(common_layers.shape_list(x))
        # This is the distance between b and x. Having this as loss helps learn
        # the bottleneck function, but if we back-propagated to x it would be
        # minimized by just setting x=0 and b=0 -- so we don't want too much
        # of the influence of this, and we stop-gradient to not zero-out x.
        x_stop = tf.stop_gradient(x)
        xb_loss = tf.reduce_mean(tf.reduce_sum(
            tf.squared_difference(x_stop, b), axis=-1))
        # To prevent this loss from exploding we clip at 1, but anneal clipping.
        clip_max = 1.0 / common_layers.inverse_exp_decay(
            warm_step, min_value=0.001)
        xb_clip = tf.maximum(tf.stop_gradient(xb_loss), clip_max)
        xb_loss *= clip_max / xb_clip
        x = tf.where(tf.less(rand, nomix_p), b, x)
      if hparams.gan_loss_factor != 0.0:
        # Add a purely sampled batch on which we'll compute the GAN loss.
        g = self.unbottleneck(
            self.sample(shape=b_shape),
            common_layers.shape_list(x)[-1],
            reuse=True)
        x = tf.concat([x, g], axis=0)
    else:
      if self._cur_bottleneck_tensor is None:
        b = self.sample()
      else:
        b = self._cur_bottleneck_tensor
      self._cur_bottleneck_tensor = b
      res_size = self.hparams.hidden_size * 2**self.hparams.num_hidden_layers
      res_size = min(res_size, hparams.max_hidden_size)
      x = self.unbottleneck(b, res_size)
    # Run decoder.
    x = self.decoder(x, encoder_layers)

    # Cut to the right size and mix before returning.
    res = x
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      res = x[:, :shape[1], :shape[2], :]

    # Final dense layer.
    res = tf.layers.dense(
        res, self.num_channels * hparams.hidden_size, name="res_dense")

    output_shape = common_layers.shape_list(res)[:-1] + [
        self.num_channels, self.hparams.hidden_size
    ]
    res = tf.reshape(res, output_shape)

    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      if hparams.use_vq_loss:
        (reconstr, _, _, _, _) = discretization.vq_loss(res, labels, vocab_size)
      else:
        reconstr = tf.layers.dense(res, vocab_size, name="autoencoder_final")
      return reconstr, {"bottleneck_loss": 0.0}

    if hparams.gan_loss_factor != 0.0:
      res, res_gan = tf.split(res, 2, axis=0)

    # Losses.
    losses = {
        "bottleneck_extra": b_loss,
        "bottleneck_l2": hparams.bottleneck_l2_factor * xb_loss
    }

    if hparams.use_vq_loss:
      vq_temperature = hparams.vq_temperature / common_layers.inverse_exp_decay(
          hparams.gan_codes_warmup_steps * 1.2,
          min_value=hparams.vq_temperature * 2)
      if hparams.mode != tf.estimator.ModeKeys.TRAIN:
        vq_temperature = None
      with tf.variable_scope("vq_loss"):
        (reconstr, _, target_codes, code_loss,
         targets_loss) = discretization.vq_loss(
             res, labels, vocab_size, temperature=vq_temperature)
      losses["code_loss"] = code_loss * hparams.code_loss_factor
      losses["training"] = targets_loss
    else:
      reconstr = tf.layers.dense(res, vocab_size, name="autoencoder_final")
      targets_loss = tf.losses.sparse_softmax_cross_entropy(
          logits=tf.reshape(reconstr, labels_shape + [vocab_size]),
          labels=tf.reshape(labels, labels_shape))
      losses["training"] = targets_loss

    # GAN losses.
    if hparams.gan_loss_factor != 0.0:
      update_means_factor = common_layers.inverse_exp_decay(
          hparams.gan_codes_warmup_steps, min_value=0.0001)
      if hparams.use_vq_loss:
        with tf.variable_scope("vq_loss", reuse=True):
          update_means = tf.less(tf.random_uniform([]), update_means_factor)
          reconstr_gan, gan_codes, _, code_loss_gan, _ = discretization.vq_loss(
              res_gan,
              labels,
              vocab_size,
              do_update=update_means,
              temperature=vq_temperature)
          reconstr_gan_nonoise = reconstr_gan
          code_loss_gan *= hparams.code_loss_factor * update_means_factor
          losses["code_loss_gan"] = code_loss_gan
      else:
        reconstr_gan = tf.layers.dense(
            res_gan, vocab_size, name="autoencoder_final", reuse=True)
        reconstr_gan_nonoise = reconstr_gan
        reconstr_gan = self.gumbel_sample(reconstr_gan)
        # Embed to codes.
        gan_codes = self.embed(reconstr_gan)

    # Add GAN loss if requested.
    gan_loss = 0.0
    if hparams.gan_loss_factor != 0.0:
      self.image_summary("gan", reconstr_gan_nonoise)

      def discriminate(x):
        """Run a dioscriminator depending on the hparams."""
        if hparams.discriminator == "default":
          return common_layers.deep_discriminator(
              x, hparams.discriminator_batchnorm, is_training)
        elif hparams.discriminator == "patched":
          return common_layers.patch_discriminator(x)
        elif hparams.discriminator == "single":
          return common_layers.single_discriminator(
              x,
              hparams.discriminator_size,
              hparams.discriminator_kernel_size,
              hparams.discriminator_strides,
              pure_mean=hparams.discriminator_pure_mean)
        elif hparams.discriminator == "double":
          return common_layers.double_discriminator(
              x,
              hparams.discriminator_size,
              hparams.discriminator_kernel_size,
              hparams.discriminator_strides,
              pure_mean=hparams.discriminator_pure_mean)
        else:
          raise Exception("Unknown discriminator %s" % hparams.discriminator)

      tc_shape = common_layers.shape_list(target_codes)
      if len(tc_shape) > 4:
        target_codes = tf.reshape(target_codes,
                                  tc_shape[:-2] + [tc_shape[-1] * tc_shape[-2]])
        gan_codes = tf.reshape(gan_codes,
                               tc_shape[:-2] + [tc_shape[-1] * tc_shape[-2]])
      gan_lr = common_layers.inverse_exp_decay(
          hparams.gan_codes_warmup_steps * 1.5)
      rev_grad_gan_codes = reverse_gradient(gan_codes, lr=gan_lr)
      gan_loss = common_layers.sliced_gan_loss(
          target_codes,
          rev_grad_gan_codes,
          discriminate,
          self.hparams.num_sliced_vecs,
          do_tanh=hparams.sliced_do_tanh)
      gan_loss *= hparams.gan_loss_factor * update_means_factor
      losses["gan_loss"] = -gan_loss

    self.image_summary("ae", reconstr)

    logits = tf.reshape(reconstr, labels_shape + [vocab_size])
    return logits, losses

  def sample(self, features=None, shape=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [
        hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
        hp.bottleneck_bits
    ]
    size = size if shape is None else shape
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
    num_channels = self.num_channels
    if "targets" not in features:
      features["targets"] = tf.zeros(
          [self.hparams.batch_size, 1, 1, num_channels], dtype=tf.int32)
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
    # Run the basic autoencoder part first.
    basic_result, losses = super(AutoencoderAutoregressive, self).body(features)
    if hparams.autoregressive_mode == "none":
      assert not hparams.autoregressive_forget_base
      return basic_result, losses
    if "training" in losses:
      plain_training_loss = losses.pop("training")
      losses["plain"] = plain_training_loss
    res_shape = common_layers.shape_list(basic_result)
    vocab_size = self._problem_hparams.vocab_size["targets"]
    if hasattr(self._hparams, "vocab_divisor"):
      vocab_size += (-vocab_size) % self._hparams.vocab_divisor
    targets = tf.one_hot(features["targets_raw"], vocab_size)
    # Prepare inputs for autoregressive modes.
    if common_layers.shape_list(features["targets"])[1] == 1:
      # This happens on the first step of predicitions.
      assert hparams.mode == tf.estimator.ModeKeys.PREDICT
      targets = tf.zeros_like(basic_result)
    targets = self.embed(targets)
    if hparams.autoregressive_gumbel_sample:
      basic_hot = self.gumbel_sample(basic_result)
    else:
      basic_hot = basic_result
    basic_result = self.embed(basic_hot)
    shape = common_layers.shape_list(basic_result)
    basic1d = tf.reshape(basic_result, [shape[0], -1, shape[-1]])
    targets = tf.reshape(targets, common_layers.shape_list(basic_result))
    # During autoregressive inference, don't resample.
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      if hasattr(hparams, "sampled_basic1d_tensor"):
        basic1d = hparams.sampled_basic1d_tensor
      else:
        hparams.sampled_basic1d_tensor = basic1d
    # Sometimes it's useful to look at non-autoregressive evals.
    targets_dropout = targets
    if (hparams.mode == tf.estimator.ModeKeys.EVAL and
        hparams.autoregressive_eval_pure_autoencoder):
      targets_dropout = tf.zeros_like(basic_result)
    # Now combine the basic reconstruction with shifted targets.
    targets1d = tf.reshape(targets_dropout, [shape[0], -1, shape[-1]])
    targets_shifted = common_layers.shift_right_3d(targets1d)
    concat1d = tf.concat([basic1d, targets_shifted], axis=-1)
    # The forget_base hparam sets purely-autoregressive mode, no autoencoder.
    if hparams.autoregressive_forget_base:
      concat1d = tf.reshape(targets, [shape[0], -1, shape[-1]])
      concat1d = common_layers.shift_right_3d(concat1d)
    # The autoregressive part depends on the mode.
    if hparams.autoregressive_mode == "conv3":
      res = common_layers.conv1d(
          concat1d,
          hparams.hidden_size,
          3,
          padding="LEFT",
          activation=common_layers.belu,
          name="autoregressive_conv3")
      res = tf.layers.dense(res, vocab_size, name="autoregressive_final")
      return tf.reshape(res, res_shape), losses
    if hparams.autoregressive_mode == "conv5":
      res = common_layers.conv1d(
          concat1d,
          hparams.hidden_size,
          5,
          padding="LEFT",
          activation=common_layers.belu,
          name="autoregressive_conv5")
      res = tf.layers.dense(res, vocab_size, name="autoregressive_final")
      return tf.reshape(res, res_shape), losses
    if hparams.autoregressive_mode == "sru":
      res = common_layers.conv1d(
          concat1d,
          hparams.hidden_size,
          3,
          padding="LEFT",
          activation=common_layers.belu,
          name="autoregressive_sru_conv3")
      res = common_layers.sru(res)
      res = tf.layers.dense(res, vocab_size, name="autoregressive_final")
      return tf.reshape(res, res_shape), losses

    raise ValueError(
        "Unsupported autoregressive mode: %s" % hparams.autoregressive_mode)

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
          [self.hparams.batch_size, 1, 1, num_channels], dtype=tf.int32)
    logits, _ = self(features)  # pylint: disable=not-callable
    samples = common_layers.sample_with_temperature(logits, 0.0)
    shape = common_layers.shape_list(samples)

    # Sample again if requested for the autoregressive part.
    extra_samples = self.hparams.autoregressive_decode_steps
    for i in range(extra_samples):
      if i == extra_samples - 2:
        self.hparams.sampling_temp /= 2
      if i == extra_samples - 1:
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
    is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN
    hparams = self.hparams
    if hparams.dropout <= 0.0 or not is_training:
      return x
    warm_step = hparams.bottleneck_warmup_steps * 2**hparams.num_hidden_layers
    dropout = common_layers.inverse_lin_decay(warm_step // 2) * hparams.dropout
    return common_layers.dropout_with_broadcast_dims(
        x, 1.0 - dropout, broadcast_dims=[-1])

  def encoder(self, x):
    with tf.variable_scope("encoder"):
      hparams = self.hparams
      layers = []
      kernel, strides = self._get_kernel_and_strides()
      residual_kernel = (hparams.residual_kernel_height,
                         hparams.residual_kernel_width)
      residual_kernel1d = (hparams.residual_kernel_height, 1)
      residual_kernel = residual_kernel1d if self.is1d else residual_kernel
      residual_conv = tf.layers.conv2d
      if hparams.residual_use_separable_conv:
        residual_conv = tf.layers.separable_conv2d
      # Down-convolutions.
      for i in range(hparams.num_hidden_layers):
        with tf.variable_scope("layer_%d" % i):
          x = self.make_even_size(x)
          layers.append(x)
          x = self.dropout(x)
          filters = hparams.hidden_size * 2**(i + 1)
          filters = min(filters, hparams.max_hidden_size)
          x = common_attention.add_timing_signal_nd(x)
          x = tf.layers.conv2d(
              x,
              filters,
              kernel,
              strides=strides,
              padding="SAME",
              activation=common_layers.belu,
              name="strided")
          y = x
          y = tf.nn.dropout(y, 1.0 - hparams.residual_dropout)
          for r in range(hparams.num_residual_layers):
            residual_filters = filters
            if r < hparams.num_residual_layers - 1:
              residual_filters = int(
                  filters * hparams.residual_filter_multiplier)
            y = residual_conv(
                y,
                residual_filters,
                residual_kernel,
                padding="SAME",
                activation=common_layers.belu,
                name="residual_%d" % r)
          x += y
          x = common_layers.layer_norm(x, name="ln")
      return x, layers

  def decoder(self, x, encoder_layers=None):
    with tf.variable_scope("decoder"):
      hparams = self.hparams
      is_training = self.hparams.mode == tf.estimator.ModeKeys.TRAIN
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
        if is_training:
          nomix_p = common_layers.inverse_lin_decay(
              int(hparams.bottleneck_warmup_steps * 0.25 * 2**j)) + 0.01
          if common_layers.should_generate_summaries():
            tf.summary.scalar("nomix_p_%d" % j, nomix_p)
        filters = hparams.hidden_size * 2**j
        filters = min(filters, hparams.max_hidden_size)
        with tf.variable_scope("layer_%d" % i):
          j = hparams.num_hidden_layers - i - 1
          x = tf.layers.conv2d_transpose(
              x,
              filters,
              kernel,
              strides=strides,
              padding="SAME",
              activation=common_layers.belu,
              name="strided")
          y = x
          for r in range(hparams.num_residual_layers):
            residual_filters = filters
            if r < hparams.num_residual_layers - 1:
              residual_filters = int(
                  filters * hparams.residual_filter_multiplier)
            y = residual_conv(
                y,
                residual_filters,
                residual_kernel,
                padding="SAME",
                activation=common_layers.belu,
                name="residual_%d" % r)
          x += tf.nn.dropout(y, 1.0 - hparams.residual_dropout)
          x = common_layers.layer_norm(x, name="ln")
          x = common_attention.add_timing_signal_nd(x)
          if encoder_layers is not None:
            enc_x = encoder_layers[j]
            enc_shape = common_layers.shape_list(enc_x)
            x_mix = x[:enc_shape[0], :enc_shape[1], :enc_shape[2], :]
            if is_training:  # Mix at the beginning of training.
              rand = tf.random_uniform(common_layers.shape_list(x_mix))
              x_mix = tf.where(tf.less(rand, nomix_p), x_mix, enc_x)
            if hparams.gan_loss_factor != 0:
              x_gan = x[enc_shape[0]:, :enc_shape[1], :enc_shape[2], :]
              x = tf.concat([x_mix, x_gan], axis=0)
            else:
              x = x_mix
      return x


@registry.register_model
class AutoencoderResidualVAE(AutoencoderResidual):
  """Residual VAE autoencoder."""

  def bottleneck(self, x):
    hparams = self.hparams
    z_size = hparams.bottleneck_bits
    x_shape = common_layers.shape_list(x)
    with tf.variable_scope("vae"):
      mu = tf.layers.dense(x, z_size, name="mu")
      if hparams.mode != tf.estimator.ModeKeys.TRAIN:
        return mu, 0.0  # No sampling or kl loss on eval.
      log_sigma = tf.layers.dense(x, z_size, name="log_sigma")
      epsilon = tf.random_normal(x_shape[:-1] + [z_size])
      z = mu + tf.exp(log_sigma / 2) * epsilon
      kl = 0.5 * tf.reduce_mean(
          tf.expm1(log_sigma) + tf.square(mu) - log_sigma, axis=-1)
      free_bits = z_size // 4
      kl_loss = tf.reduce_mean(tf.maximum(kl - free_bits, 0.0))
    return z, kl_loss * hparams.kl_beta

  def sample(self, features=None, shape=None):
    del features
    hparams = self.hparams
    div_x = 2**hparams.num_hidden_layers
    div_y = 1 if self.is1d else 2**hparams.num_hidden_layers
    size = [
        hparams.batch_size, hparams.sample_height // div_x,
        hparams.sample_width // div_y, hparams.bottleneck_bits
    ]
    size = size if shape is None else shape
    return tf.random_normal(size)


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

  def sample(self, features=None, shape=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [
        hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
        hp.bottleneck_bits
    ]
    size = size if shape is None else shape
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

  def sample(self, features=None, shape=None):
    del features
    hp = self.hparams
    div_x = 2**hp.num_hidden_layers
    div_y = 1 if self.is1d else 2**hp.num_hidden_layers
    size = [
        hp.batch_size, hp.sample_height // div_x, hp.sample_width // div_y,
        hp.bottleneck_bits
    ]
    size = size if shape is None else shape
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
      log_p = tf.log1p(-float(noise) / 2) / float(hparams.bottleneck_bits)
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
class AutoencoderDualDiscrete(AutoencoderResidualDiscrete):
  """Dual discrete autoencoder."""

  def body(self, features):
    if self.hparams.mode != tf.estimator.ModeKeys.EVAL:
      t, i = features["targets_raw"], features["inputs_raw"]
      t, i = common_layers.pad_to_same_length(t, i)
      features["targets_raw"] = tf.concat([t, i], axis=0)
    return super(AutoencoderDualDiscrete, self).body(features)

  def embed(self, x, name="embedding"):
    if self.hparams.mode == tf.estimator.ModeKeys.EVAL:
      return super(AutoencoderDualDiscrete, self).embed(x, name=name + "_t")
    xt, xi = tf.split(x, 2, axis=0)
    xte = super(AutoencoderDualDiscrete, self).embed(xt, name=name + "_t")
    xie = super(AutoencoderDualDiscrete, self).embed(xi, name=name + "_i")
    return tf.concat([xte, xie], axis=0)

  def bottleneck(self, x):
    hparams = self.hparams
    b, _ = super(AutoencoderDualDiscrete, self).bottleneck(x)
    if hparams.mode == tf.estimator.ModeKeys.EVAL:
      return b, 0.0
    bt, bi = tf.split(b, 2, axis=0)
    if self.hparams.mode != tf.estimator.ModeKeys.TRAIN:
      return tf.concat([bi, bi], axis=0), 0.0
    # Share the first hparams.bottleneck_shared_bits.
    shared = (bt + bi) / 2  # -1 if both -1, 1 if both were 1, 0 if disagree.
    rand = tf.random_uniform(common_layers.shape_list(bt))
    br = tf.where(rand < 0.5, bt, bi)  # Break ties at random.
    bs = tf.where(shared == 0, br, shared)
    bs = tf.concat([bs, bs], axis=0)
    n = hparams.bottleneck_shared_bits
    step = tf.train.get_global_step()
    zero = tf.constant(0, dtype=tf.int64)
    if step is None:
      step = zero
    step = tf.maximum(zero, step - hparams.bottleneck_shared_bits_start_warmup)
    f = common_layers.inverse_lin_decay(
        hparams.bottleneck_shared_bits_stop_warmup, min_value=0.1, step=step)
    n = tf.where(step > 1, n * f, n)
    n = tf.cast(n, tf.int64)
    b_shape = common_layers.shape_list(b)
    b = tf.concat([bs[..., :n], b[..., n:]], axis=-1)
    b = tf.reshape(b, b_shape)
    return b, 0.0

  def unbottleneck(self, b, res_size, reuse=None):
    x = super(AutoencoderDualDiscrete, self).unbottleneck(
        b, res_size, reuse=reuse)
    if self.hparams.mode == tf.estimator.ModeKeys.EVAL:
      return tf.layers.dense(x, res_size, name="dual_unbottleneck_t")
    xt, xi = tf.split(x, 2, axis=0)
    xt = tf.layers.dense(xt, res_size, name="dual_unbottleneck_t")
    xi = tf.layers.dense(xt, res_size, name="dual_unbottleneck_i")
    return tf.concat([xt, xi], axis=0)

  def infer(self, features, *args, **kwargs):  # pylint: disable=arguments-differ
    """Produce predictions from the model."""
    del args, kwargs
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Set targets to input size firts.
    features["targets"] = tf.zeros_like(features["inputs"])
    self._encode_on_predict = True
    logits, _ = self(features)  # pylint: disable=not-callable
    if self.hparams.gan_loss_factor != 0:
      logits, _ = tf.split(logits, 2, axis=0)  # Remove GAN.
    logits, _ = tf.split(logits, 2, axis=0)  # Targets and inputs from encoding.
    # Uncomment the line below to get reconstructed inputs instead of targets.
    # (and comment out the line above at the same time).
    # _, logits = tf.split(logits, 2, axis=0)
    samples = tf.argmax(logits, axis=-1)

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return samples.
    return samples


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
      condition = tf.less(tf.random_uniform([]), 0.5)
      condition = tf.reshape(condition, [1] * len(b.shape))
      condition = tf.tile(condition, b.shape)
      b1 = tf.where(condition, b, b1)
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
      b = self.full_stack(b, 2 * x_size, 2 * hparams.bottleneck_bits, losses,
                          is_training, num_stacks - 1)
      b = self.unbottleneck(b, x_size)
      b = common_layers.mix(b, x, hparams.bottleneck_warmup_steps, is_training)
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
  hparams.optimizer = "adam"
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
  hparams.dropout = 0.05
  hparams.add_hparam("max_hidden_size", 1024)
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("bottleneck_shared_bits", 0)
  hparams.add_hparam("bottleneck_shared_bits_start_warmup", 0)
  hparams.add_hparam("bottleneck_shared_bits_stop_warmup", 0)
  hparams.add_hparam("bottleneck_noise", 0.1)
  hparams.add_hparam("bottleneck_warmup_steps", 2000)
  hparams.add_hparam("sample_height", 32)
  hparams.add_hparam("sample_width", 32)
  hparams.add_hparam("discriminator_batchnorm", True)
  hparams.add_hparam("num_sliced_vecs", 20000)
  hparams.add_hparam("sliced_do_tanh", int(True))
  hparams.add_hparam("discriminator_size", 256)
  hparams.add_hparam("discriminator_kernel_size", 6)
  hparams.add_hparam("discriminator_strides", 4)
  hparams.add_hparam("discriminator_pure_mean", int(False))
  hparams.add_hparam("code_loss_factor", 1.0)
  hparams.add_hparam("gan_codes_warmup_steps", 16000)
  hparams.add_hparam("gan_loss_factor", 0.0)
  hparams.add_hparam("bottleneck_l2_factor", 0.05)
  hparams.add_hparam("gumbel_temperature", 0.5)
  hparams.add_hparam("gumbel_noise_factor", 0.5)
  hparams.add_hparam("vq_temperature", 0.001)
  hparams.add_hparam("use_vq_loss", int(False))
  hparams.add_hparam("discriminator", "double")
  return hparams


@registry.register_hparams
def autoencoder_autoregressive():
  """Autoregressive autoencoder model."""
  hparams = autoencoder_basic()
  hparams.add_hparam("autoregressive_forget_base", False)
  hparams.add_hparam("autoregressive_mode", "none")
  hparams.add_hparam("autoregressive_decode_steps", 0)
  hparams.add_hparam("autoregressive_eval_pure_autoencoder", False)
  hparams.add_hparam("autoregressive_gumbel_sample", False)
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
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.max_hidden_size = 1024
  hparams.add_hparam("num_residual_layers", 2)
  hparams.add_hparam("residual_kernel_height", 3)
  hparams.add_hparam("residual_kernel_width", 3)
  hparams.add_hparam("residual_filter_multiplier", 2.0)
  hparams.add_hparam("residual_dropout", 0.2)
  hparams.add_hparam("residual_use_separable_conv", int(True))
  hparams.add_hparam("kl_beta", 1.0)
  return hparams


@registry.register_hparams
def autoencoder_residual_text():
  """Residual autoencoder model for text."""
  hparams = autoencoder_residual()
  hparams.bottleneck_bits = 32
  hparams.batch_size = 1024
  hparams.hidden_size = 64
  hparams.max_hidden_size = 512
  hparams.bottleneck_noise = 0.0
  hparams.bottom = {
      "inputs": modalities.identity_bottom,
      "targets": modalities.identity_bottom,
  }
  hparams.top = {
      "targets": modalities.identity_top,
  }
  hparams.autoregressive_mode = "none"
  hparams.sample_width = 1
  return hparams


@registry.register_hparams
def autoencoder_basic_discrete():
  """Basic autoencoder model."""
  hparams = autoencoder_autoregressive()
  hparams.num_hidden_layers = 5
  hparams.hidden_size = 64
  hparams.bottleneck_bits = 1024
  hparams.bottleneck_noise = 0.1
  hparams.add_hparam("discretize_warmup_steps", 16000)
  return hparams


@registry.register_hparams
def autoencoder_residual_discrete():
  """Residual discrete autoencoder model."""
  hparams = autoencoder_residual()
  hparams.bottleneck_bits = 1024
  hparams.bottleneck_noise = 0.05
  hparams.add_hparam("discretize_warmup_steps", 16000)
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
  hparams.residual_dropout = 0.4
  return hparams


@registry.register_hparams
def autoencoder_ordered_discrete():
  """Ordered discrete autoencoder model."""
  hparams = autoencoder_residual_discrete()
  hparams.bottleneck_noise = 0.05  # Use 0.8 for ordered.
  hparams.gan_loss_factor = 0.05
  hparams.add_hparam("unordered", True)
  return hparams


@registry.register_hparams
def autoencoder_ordered_discrete_image64():
  """Ordered discrete autoencoder model."""
  hparams = autoencoder_ordered_discrete()
  hparams.batch_size = 32
  hparams.num_hidden_layers = 6
  hparams.bottleneck_warmup_steps *= 2
  hparams.gan_codes_warmup_steps *= 2

  return hparams


@registry.register_hparams
def autoencoder_ordered_discrete_patched():
  """Ordered discrete autoencoder model."""
  hparams = autoencoder_ordered_discrete()
  hparams.discriminator = "patched"
  return hparams


@registry.register_hparams
def autoencoder_ordered_discrete_single():
  """Ordered discrete autoencoder model."""
  hparams = autoencoder_ordered_discrete()
  hparams.discriminator = "single"
  return hparams


@registry.register_hparams
def autoencoder_ordered_discrete_hs256():
  """Ordered discrete autoencoder model."""
  hparams = autoencoder_ordered_discrete()
  hparams.hidden_size = 256
  return hparams


@registry.register_hparams
def autoencoder_ordered_text():
  """Ordered discrete autoencoder model for text."""
  hparams = autoencoder_ordered_discrete()
  hparams.bottleneck_bits = 1024
  hparams.bottleneck_shared_bits = 1024-64
  hparams.bottleneck_shared_bits_start_warmup = 75000
  hparams.bottleneck_shared_bits_stop_warmup = 275000
  hparams.num_hidden_layers = 7
  hparams.batch_size = 1024
  hparams.autoregressive_mode = "conv5"
  hparams.max_hidden_size = 1024
  hparams.bottom = {
      "inputs": modalities.identity_bottom,
      "targets": modalities.identity_bottom,
  }
  hparams.top = {
      "targets": modalities.identity_top,
  }
  hparams.sample_height = 128
  hparams.sample_width = 1
  return hparams


@registry.register_hparams
def autoencoder_ordered_text_small():
  """Ordered discrete autoencoder model for text, small version."""
  hparams = autoencoder_ordered_text()
  hparams.bottleneck_bits = 32
  hparams.num_hidden_layers = 3
  hparams.hidden_size = 64
  hparams.max_hidden_size = 512
  hparams.bottleneck_noise = 0.0
  hparams.autoregressive_mode = "conv5"
  hparams.sample_height = 4
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
  hparams.num_hidden_layers = 3
  hparams.bottleneck_bits = 24
  hparams.batch_size = 2
  hparams.gan_loss_factor = 0.01
  hparams.bottleneck_l2_factor = 0.001
  hparams.add_hparam("video_modality_loss_cutoff", 0.02)
  return hparams


@registry.register_hparams
def autoencoder_discrete_tiny():
  """Discrete autoencoder model for compressing pong frames for testing."""
  hparams = autoencoder_ordered_discrete()
  hparams.num_hidden_layers = 2
  hparams.bottleneck_bits = 24
  hparams.batch_size = 2
  hparams.gan_loss_factor = 0.
  hparams.bottleneck_l2_factor = 0.001
  hparams.add_hparam("video_modality_loss_cutoff", 0.02)
  hparams.num_residual_layers = 1
  hparams.hidden_size = 32
  hparams.max_hidden_size = 64
  return hparams


@registry.register_hparams
def autoencoder_discrete_cifar():
  """Discrete autoencoder model for compressing cifar."""
  hparams = autoencoder_ordered_discrete()
  hparams.bottleneck_noise = 0.0
  hparams.bottleneck_bits = 90
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.num_residual_layers = 4
  hparams.batch_size = 32
  hparams.learning_rate_constant = 1.0
  return hparams


@registry.register_ranged_hparams
def autoencoder_range(rhp):
  """Tuning grid of the main autoencoder params."""
  rhp.set_float("dropout", 0.01, 0.3)
  rhp.set_float("gan_loss_factor", 0.01, 0.1)
  rhp.set_float("bottleneck_l2_factor", 0.001, 0.1, scale=rhp.LOG_SCALE)
  rhp.set_discrete("bottleneck_warmup_steps", [200, 2000])
  rhp.set_float("gumbel_temperature", 0, 1)
  rhp.set_float("gumbel_noise_factor", 0, 0.5)


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
