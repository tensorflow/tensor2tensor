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

"""Modalities define the bottom and top of the model (not the body)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_audio
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.layers import discretization
from tensor2tensor.utils import modality

import tensorflow as tf
import tensorflow_probability as tfp


class SymbolModality(modality.Modality):
  """Modality for sets of discrete symbols.

  Input:
    Embedding.

  Output:
    Linear transformation + softmax.
  """

  @property
  def name(self):
    return "symbol_modality_%d_%d" % (self._vocab_size,
                                      self._model_hparams.hidden_size)

  @property
  def top_is_pointwise(self):
    return True

  @property
  def targets_weights_fn(self):
    weights_fn = common_layers.weights_nonzero

    hp = self._model_hparams
    if hp and hp.prepend_mode != "none":
      assert (hp.prepend_mode == "prepend_inputs_masked_attention" or
              hp.prepend_mode == "prepend_inputs_full_attention")

      if (
          # In masked attention mode, during training, the network try to
          # autoregressively predicting the inputs portion, while the
          # evaluation is only done on the output
          hp.prepend_mode != "prepend_inputs_masked_attention" or
          hp.mode != tf.estimator.ModeKeys.TRAIN):
        weights_fn = common_layers.weights_prepend_inputs_to_targets

    return weights_fn

  def _get_weights(self, hidden_dim=None):
    """Create or get concatenated embedding or softmax variable.

    Args:
      hidden_dim: dim of the variable. Defaults to _model_hparams' hidden_size

    Returns:
       a list of self._num_shards Tensors.
    """
    if hidden_dim is None:
      hidden_dim = self._model_hparams.hidden_size
    num_shards = self._model_hparams.symbol_modality_num_shards
    shards = []
    for i in range(num_shards):
      shard_size = (self._vocab_size // num_shards) + (
          1 if i < self._vocab_size % num_shards else 0)
      var_name = "weights_%d" % i
      shards.append(
          tf.get_variable(
              var_name, [shard_size, hidden_dim],
              initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5)))
    if num_shards == 1:
      ret = shards[0]
    else:
      ret = tf.concat(shards, 0)
    # Convert ret to tensor.
    if not tf.executing_eagerly():
      ret = common_layers.convert_gradient_to_tensor(ret)
    return ret

  def bottom_simple(self, x, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
      # Ensure the inputs are 3-D
      if len(x.get_shape()) == 4:
        x = tf.squeeze(x, axis=3)
      while len(x.get_shape()) < 3:
        x = tf.expand_dims(x, axis=-1)

      var = self._get_weights()
      x = common_layers.dropout_no_scaling(
          x, 1.0 - self._model_hparams.symbol_dropout)
      ret = common_layers.gather(var, x)
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._model_hparams.hidden_size**0.5
      ret *= tf.expand_dims(
          common_layers.cast_like(tf.not_equal(x, 0), ret), -1)
      return ret

  def bottom(self, x):
    if (self._model_hparams.shared_embedding_and_softmax_weights or
        self._model_hparams.get("shared_embedding")):
      return self.bottom_simple(x, "shared", reuse=None)
    return self.bottom_simple(x, "input_emb", reuse=None)

  def targets_bottom(self, x):
    if (self._model_hparams.shared_embedding_and_softmax_weights or
        self._model_hparams.get("shared_embedding")):
      try:
        return self.bottom_simple(x, "shared", reuse=True)
      except ValueError:
        # perhaps there were no inputs, and this is a new variable.
        return self.bottom_simple(x, "shared", reuse=None)
    else:
      return self.bottom_simple(x, "target_emb", reuse=None)

  def top(self, body_output, _):
    """Generate logits.

    Args:
      body_output: A Tensor with shape
        [batch, p0, p1, self._model_hparams.hidden_size].

    Returns:
      logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
    """
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)

    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = tf.AUTO_REUSE
    else:
      scope_name = "softmax"
      reuse = False
    with tf.variable_scope(scope_name, reuse=reuse):
      body_output_shape = common_layers.shape_list(body_output)
      var = self._get_weights(body_output_shape[-1])
      if (self._model_hparams.factored_logits and
          self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
        # insert channels dimension
        body_output = tf.expand_dims(body_output, 3)
        return common_layers.FactoredTensor(body_output, var)
      else:
        body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
        logits = tf.matmul(body_output, var, transpose_b=True)
        return tf.reshape(logits,
                          body_output_shape[:-1] + [1, self._vocab_size])


class SymbolModalityWeightsAll(SymbolModality):
  """SymbolModality for features that do not have 0-padding."""

  @property
  def targets_weights_fn(self):
    return common_layers.weights_all


class SymbolModalityOneHot(SymbolModality):
  """Simple SymbolModality with one hot as embeddings."""

  def bottom(self, x):
    return tf.one_hot(x, self._vocab_size)

  def targets_bottom(self, x):
    return tf.one_hot(x, self._vocab_size)

  def top(self, body_output, _):
    return body_output

  def loss(self, top_out, targets):
    labels = tf.one_hot(targets, self._vocab_size)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=top_out, labels=labels)
    return tf.reduce_mean(loss), tf.constant(1.0)


class CTCSymbolModality(SymbolModality):
  """SymbolModality that uses CTC loss."""

  def loss(self, top_out, targets):
    """Compute the CTC loss."""
    logits = top_out
    with tf.name_scope("ctc_loss", values=[logits, targets]):
      # For CTC we assume targets are 1d, [batch, length, 1, 1] here.
      targets_shape = targets.get_shape().as_list()
      assert len(targets_shape) == 4
      assert targets_shape[2] == 1
      assert targets_shape[3] == 1
      targets = tf.squeeze(targets, axis=[2, 3])
      logits = tf.squeeze(logits, axis=[2, 3])
      targets_mask = 1 - tf.to_int32(tf.equal(targets, 0))
      targets_lengths = tf.reduce_sum(targets_mask, axis=1)
      sparse_targets = tf.keras.backend.ctc_label_dense_to_sparse(
          targets, targets_lengths)
      xent = tf.nn.ctc_loss(
          sparse_targets,
          logits,
          targets_lengths,
          time_major=False,
          preprocess_collapse_repeated=False,
          ctc_merge_repeated=False)
      weights = self.targets_weights_fn(targets)  # pylint: disable=not-callable
      return tf.reduce_sum(xent), tf.reduce_sum(weights)


class ImageModality(modality.Modality):
  """Modality for images."""
  PIXEL_EMBEDDING_SIZE = 64

  def bottom(self, x):
    with tf.variable_scope(self.name):
      if not tf.executing_eagerly():
        tf.summary.image(
            "inputs", common_layers.tpu_safe_image_summary(x), max_outputs=2)
      return tf.to_float(x)

  def targets_bottom(self, x):
    inputs = x
    with tf.variable_scope(self.name):
      if not tf.executing_eagerly():
        tf.summary.image(
            "targets_bottom",
            common_layers.tpu_safe_image_summary(inputs),
            max_outputs=1)
      inputs_shape = common_layers.shape_list(inputs)
      if len(inputs_shape) != 4:
        raise ValueError("Assuming images given as int tensors in the format "
                         "[batch, height, width, channels] (256 values).")
      # We embed each of 256=self.top_dimensionality possible pixel values.
      embedding_var = tf.get_variable(
          "pixel_embedding",
          [self.top_dimensionality, self.PIXEL_EMBEDDING_SIZE])
      hot_inputs = tf.one_hot(tf.to_int32(inputs), self.top_dimensionality)
      hot_inputs = tf.reshape(hot_inputs, [-1, self.top_dimensionality])
      embedded = tf.matmul(hot_inputs, embedding_var)
      # Let's now merge all channels that were embedded into a single vector.
      merged_size = self.PIXEL_EMBEDDING_SIZE * inputs_shape[3]
      embedded = tf.reshape(embedded, inputs_shape[:3] + [merged_size])
      merged = tf.layers.dense(
          embedded,
          self._model_hparams.hidden_size,
          name="merge_pixel_embedded_channels")
      return merged

  def top(self, body_output, _):
    # TODO(lukaszkaiser): is this a universal enough way to get channels?
    num_channels = self._model_hparams.problem.num_channels
    with tf.variable_scope("rgb_softmax"):
      body_output_shape = common_layers.shape_list(body_output)
      reshape_shape = body_output_shape[:3]
      reshape_shape.extend([num_channels, self.top_dimensionality])
      res = tf.layers.dense(body_output, self.top_dimensionality * num_channels)
      res = tf.reshape(res, reshape_shape)
      if not tf.get_variable_scope().reuse:
        res_argmax = tf.argmax(res, axis=-1)
        tf.summary.image(
            "result",
            common_layers.tpu_safe_image_summary(res_argmax),
            max_outputs=1)
      return res

  def loss(self, top_out, targets):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    cutoff = getattr(self._model_hparams, "video_modality_loss_cutoff", 0.0)
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        cutoff=cutoff,
        weights_fn=self.targets_weights_fn)


class ImageChannelCompressModality(modality.Modality):
  """Modality for images using channel compression for generation."""

  @property
  def num_channels(self):
    return 3

  def bottom_compress(self, inputs, name="bottom"):
    """Compresses channel-wise input pixels into whole pixel representions.

    Perform conversion of RGB pixel values to a real number in the range -1 to
    1. This combines pixel channels to form a representation of shape
    [img_len, img_len].

    Args:
      inputs: Tensor representing RGB pixel intensities as integers, of shape
        [batch, img_len, img_len, channels].
      name: string, scope.

    Returns:
      body_input: Tensor of shape
        [batch, img_len, img_len, self._model_hparams.hidden_size].
    """
    with tf.variable_scope(name):
      inputs = tf.to_float(inputs)
      hp = self._model_hparams
      if hp.mode != tf.estimator.ModeKeys.PREDICT:
        tf.summary.image(
            "inputs",
            common_layers.tpu_safe_image_summary(inputs),
            max_outputs=2)
      inputs = common_layers.convert_rgb_to_symmetric_real(inputs)

      # Reshape inputs to apply convolutions across [img_len, img_len*channels].
      inputs_shape = common_layers.shape_list(inputs)
      inputs = tf.reshape(
          inputs, [-1, inputs_shape[1], inputs_shape[2] * inputs_shape[3], 1])

      # Compress RGB intensities for each pixel using a convolution.
      outputs = tf.layers.conv2d(
          inputs,
          self._model_hparams.hidden_size,
          kernel_size=(1, self.num_channels),
          padding="VALID",
          strides=(1, self.num_channels),
          activation=tf.nn.relu,
          name="conv_input")
      return outputs

  def bottom(self, x):
    return self.bottom_compress(x, "input_bottom")

  def targets_bottom(self, x):
    return self.bottom_compress(x, "output_bottom")

  def top(self, body_output, _):
    """Transforms body output to return logits.

    Args:
      body_output: Tensor of shape [batch, img_len, img_len, depth].

    Returns:
      Tensor of shape [batch, img_len, img_len, channels, top_dimensionality].
    """
    with tf.variable_scope(self.name):
      hidden_size = self._model_hparams.hidden_size
      img_len = self._model_hparams.img_len
      channels = self.num_channels  # RGB
      batch = common_layers.shape_list(body_output)[0]
      x = tf.layers.conv2d(
          body_output,
          hidden_size * channels,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          activation=tf.nn.relu,
          name="decompress_conv")
      x = tf.reshape(x, [batch, img_len, img_len * channels, hidden_size])
      x = common_layers.layer_preprocess(x, self._model_hparams)
      x = tf.layers.dense(x,
                          self.top_dimensionality,
                          use_bias=True,
                          activation=None,
                          name="output_conv")
      x = tf.reshape(
          x, [batch, img_len, img_len, channels, self.top_dimensionality])
      return x


class ImageChannelBottomIdentityModality(ImageChannelCompressModality):

  def top(self, body_output, _):
    return body_output


class ImageChannelEmbeddingsBottom(modality.Modality):
  """Modality for images using channel compression for generation."""

  def get_channel_embeddings(self,
                             io_depth,
                             targets,
                             hidden_size,
                             name="channel"):
    """Get separate embedding for each of the channels."""
    targets_split = tf.split(targets, io_depth, axis=3)
    rgb_embedding_var = tf.get_variable("rgb_target_emb_%s" % name,
                                        [256 * io_depth, hidden_size])
    rgb_embedding_var = tf.identity(rgb_embedding_var)
    rgb_embedding_var *= float(hidden_size)**0.5
    channel_target_embs = []
    for i in range(io_depth):
      # Adding the channel offsets to get the right embedding since the
      # embedding tensor has shape 256 * io_depth, hidden_size
      target_ids = tf.squeeze(targets_split[i], axis=3) + i * 256
      target_embs = common_layers.gather(rgb_embedding_var, target_ids)
      channel_target_embs.append(target_embs)

    return tf.concat(channel_target_embs, axis=-1)

  def targets_bottom(self, x):
    inputs = x
    io_depth = self._model_hparams.num_channels
    tshape = common_layers.shape_list(inputs)
    hidden_size = self._model_hparams.hidden_size
    target_embeddings = self.get_channel_embeddings(io_depth, inputs,
                                                    hidden_size, "input_bottom")
    return tf.reshape(target_embeddings,
                      [tshape[0], tshape[1], tshape[2] * io_depth, hidden_size])

  def top(self, body_output, _):
    with tf.variable_scope(self.name):
      img_len = self._model_hparams.img_len
      channels = self._model_hparams.num_channels
      x = tf.layers.dense(
          body_output, 256, use_bias=True, activation=None, name="output_conv")
      x = tf.reshape(x,
                     [-1, img_len, img_len, channels, self.top_dimensionality])
      return x


class AudioModality(modality.Modality):
  """Performs strided conv compressions for audio data."""

  def bottom(self, x):
    """Transform input from data space to model space.

    Args:
      x: A Tensor with shape [batch, ...]

    Returns:
      body_input: A Tensor with shape [batch, ?, ?,
        self._model_hparams.hidden_size].
    """
    inputs = x
    with tf.variable_scope(self.name):
      # TODO(aidangomez): Will need to sort out a better audio pipeline
      def xnet_resblock(x, filters, res_relu, name):
        """Xception block."""
        with tf.variable_scope(name):
          # Typically audio samples are >100k samples in length and have a width
          # of 2 or 4. Mono audio has a single channel while stereo has 2.
          y = common_layers.separable_conv_block(
              x,
              filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
              first_relu=True,
              padding="SAME",
              force2d=True,
              name="sep_conv_block")
          y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 2))
          return y + common_layers.conv_block(
              x,
              filters, [((1, 1), (1, 1))],
              padding="SAME",
              strides=(2, 2),
              first_relu=res_relu,
              force2d=True,
              name="res_conv0")

      x = tf.to_float(inputs) / 255.
      x.set_shape([None, None, None, 1])
      for i in range(self._model_hparams.audio_compression):
        x = xnet_resblock(x, 2**(i + 1), True, "compress_block_%d" % i)
      return xnet_resblock(x,
                           self._model_hparams.hidden_size,
                           False,
                           "compress_block_final")


class AudioSpectralModality(modality.Modality):
  """Performs strided conv compressions for audio spectral data."""

  def bottom(self, x):
    """Transform input from data space to model space.

    Args:
      x: A Tensor with shape [batch, ...]

    Returns:
      body_input: A Tensor with shape [batch, ?, ?,
        self._model_hparams.hidden_size].
    """
    inputs = x
    with tf.variable_scope(self.name):
      # TODO(aidangomez): Will need to sort out a better audio pipeline
      def xnet_resblock(x, filters, res_relu, name):
        """Xception-like block."""
        with tf.variable_scope(name):
          # We only stride along the length dimension to preserve the spectral
          # bins (which are tiny in dimensionality relative to length)
          y = common_layers.separable_conv_block(
              x,
              filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
              first_relu=True,
              padding="SAME",
              force2d=True,
              name="sep_conv_block")
          y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 1))
          return y + common_layers.conv_block(
              x,
              filters, [((1, 1), (1, 1))],
              padding="SAME",
              strides=(2, 1),
              first_relu=res_relu,
              force2d=True,
              name="res_conv0")

      # Bitcast back from int32
      x = tf.bitcast(inputs, tf.float32)
      x.set_shape([None, None, None, 1])
      for i in range(self._model_hparams.audio_compression):
        x = xnet_resblock(x, 2**(i + 1), True, "compress_block_%d" % i)
      return xnet_resblock(x,
                           self._model_hparams.hidden_size,
                           False,
                           "compress_block_final")


class SpeechRecognitionModality(modality.Modality):
  """Common ASR filterbank processing."""

  def bottom(self, x):
    """Use batchnorm instead of CMVN and shorten the stft with strided convs.

    Args:
      x: float32 tensor with shape [batch_size, len, 1, freqs * channels]

    Returns:
      float32 tensor with shape [batch_size, shorter_len, 1, hidden_size]
    """
    inputs = x
    p = self._model_hparams

    num_mel_bins = p.audio_num_mel_bins
    num_channels = 3 if p.audio_add_delta_deltas else 1

    with tf.variable_scope(self.name):
      if p.audio_preproc_in_bottom:
        # Compute filterbanks
        with tf.variable_scope("fbanks"):
          waveforms = tf.squeeze(inputs, [2, 3])
          mel_fbanks = common_audio.compute_mel_filterbank_features(
              waveforms,
              sample_rate=p.audio_sample_rate,
              dither=p.audio_dither,
              preemphasis=p.audio_preemphasis,
              frame_length=p.audio_frame_length,
              frame_step=p.audio_frame_step,
              lower_edge_hertz=p.audio_lower_edge_hertz,
              upper_edge_hertz=p.audio_upper_edge_hertz,
              num_mel_bins=p.audio_num_mel_bins,
              apply_mask=True)
          if p.audio_add_delta_deltas:
            mel_fbanks = common_audio.add_delta_deltas(mel_fbanks)
          x = tf.reshape(mel_fbanks,
                         common_layers.shape_list(mel_fbanks)[:2] +
                         [num_mel_bins, num_channels])

          nonpadding_mask = 1. - common_attention.embedding_to_padding(x)
          num_of_nonpadding_elements = tf.reduce_sum(
              nonpadding_mask) * num_mel_bins * num_channels

          # This replaces CMVN estimation on data
          var_epsilon = 1e-09
          mean = tf.reduce_sum(
              x, axis=[1], keepdims=True) / num_of_nonpadding_elements
          variance = (num_of_nonpadding_elements * mean**2. -
                      2. * mean * tf.reduce_sum(x, axis=[1], keepdims=True) +
                      tf.reduce_sum(x**2, axis=[1], keepdims=True)
                     ) / num_of_nonpadding_elements
          x = (x - mean) * tf.rsqrt(variance + var_epsilon) * tf.expand_dims(
              nonpadding_mask, -1)
      else:
        x = inputs

      # The convention is that the models are flattened along the spatial,
      # dimensions, thus the speech preprocessor treats frequencies and
      # channels as image colors (last axis)
      x.set_shape([None, None, num_mel_bins, num_channels])

      # TODO(chorowski): how to specify bottom's hparams and avoid hardcoding?
      x = tf.pad(x, [[0, 0], [0, 8], [0, 0], [0, 0]])
      for _ in range(2):
        x = tf.layers.conv2d(
            x, 128, (3, 3), (2, 2), use_bias=False)
        x = common_layers.layer_norm(x)
        x = tf.nn.relu(x)

      xshape = common_layers.shape_list(x)
      # apply a conv that will remove all frequencies and at the same time
      # project the output into desired hidden_size
      x = tf.pad(x, [[0, 0], [0, 2], [0, 0], [0, 0]])
      x = tf.layers.conv2d(x, p.hidden_size, (3, xshape[2]), use_bias=False)

      assert common_layers.shape_list(x)[2] == 1
      x = common_layers.layer_norm(x)
      x = tf.nn.relu(x)
    return x


class VideoModality(modality.Modality):
  """Modality for videos, i.e., time-sequences of frames."""

  def bottom(self, x):
    common_video.gif_summary("inputs", x, max_outputs=1)
    x = common_layers.standardize_images(x)
    return x

  def targets_bottom(self, x):
    common_video.gif_summary("targets", x, max_outputs=1)
    x = common_layers.standardize_images(x)
    return x

  def top(self, body_output, targets):
    num_channels = self._model_hparams.problem.num_channels
    shape = common_layers.shape_list(body_output)
    reshape_shape = shape[:-1] + [num_channels, self.top_dimensionality]
    res = tf.reshape(body_output, reshape_shape)
    # Calculate argmax so as to have a summary with the produced images.
    x = tf.argmax(tf.reshape(res, [-1, self.top_dimensionality]), axis=-1)
    x = tf.reshape(x, shape[:-1] + [num_channels])
    common_video.gif_summary("results", x, max_outputs=1)
    return res

  def loss(self, top_out, targets):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:])
    targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
    cutoff = getattr(self._model_hparams, "video_modality_loss_cutoff", 0.01)
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        cutoff=cutoff,
        weights_fn=self.targets_weights_fn)


class VideoModalityBitwise(VideoModality):
  """Video Modality where bottom embeds pixels bitwise."""
  PIXEL_EMBEDDING_SIZE = 64

  def bottom(self, x):
    inputs = x
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      common_layers.summarize_video(inputs, "bottom")
      # Embed bitwise.
      assert self.top_dimensionality == 256
      embedded = discretization.int_to_bit_embed(inputs, 8,
                                                 self.PIXEL_EMBEDDING_SIZE)
      # Project.
      return tf.layers.dense(
          embedded,
          self._model_hparams.hidden_size,
          name="merge_pixel_embedded_frames")

  def targets_bottom(self, x):  # pylint: disable=arguments-differ
    inputs = x
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      common_layers.summarize_video(inputs, "targets_bottom")
      # Embed bitwise.
      assert self.top_dimensionality == 256
      embedded = discretization.int_to_bit_embed(inputs, 8,
                                                 self.PIXEL_EMBEDDING_SIZE)
      # Transpose and project.
      transposed = common_layers.time_to_channels(embedded)
      return tf.layers.dense(
          transposed,
          self._model_hparams.hidden_size,
          name="merge_pixel_embedded_frames")


class VideoModalityPixelNoise(VideoModality):
  """Video modality that introduces pixel noise on input during training."""

  def bottom(self, x):
    inputs = x
    if self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN:
      background = tfp.distributions.percentile(inputs, 50., axis=[0, 1, 2, 3])
      input_shape = common_layers.shape_list(inputs)
      input_size = tf.reduce_prod(input_shape[:-1])
      input_mask = tf.multinomial(
          tf.log([[self.input_noise, 1.-self.input_noise]]), input_size)
      input_mask = tf.reshape(tf.cast(input_mask, tf.int32),
                              input_shape[:-1]+[1])
      inputs = inputs * input_mask + background * (1 - input_mask)
    return super(VideoModalityPixelNoise, self).bottom(inputs)

  @property
  def input_noise(self):
    return getattr(self._model_hparams, "video_modality_input_noise", 0.25)


class VideoModalityL1(VideoModality):
  """Video modality that predicts a scalar per channel with an L1 loss."""

  def top(self, body_output, _):
    num_channels = self._model_hparams.problem.num_channels
    num_frames = self._model_hparams.video_num_target_frames
    with tf.variable_scope("rgb"):
      body_output_shape = common_layers.shape_list(body_output)
      res = tf.layers.dense(body_output, num_channels * num_frames, name="cast")
      res = tf.reshape(res, body_output_shape[:3] + [num_channels, num_frames])
      res = tf.transpose(res, [0, 4, 1, 2, 3])  # Move frames next to batch.
      if not tf.get_variable_scope().reuse:
        res_argmax = res[:, -1, :, :, :]
        tf.summary.image(
            "result",
            common_layers.tpu_safe_image_summary(res_argmax),
            max_outputs=1)
      return tf.expand_dims(res, axis=-1)  # Add an axis like in perplexity.

  @property
  def cutoff(self):
    return getattr(self._model_hparams, "video_modality_loss_cutoff", 0.2)

  def internal_loss(self, logits, targets):
    return tf.nn.relu(tf.abs(logits - targets) - self.cutoff)

  def loss(self, top_out, targets):
    """Compute loss numerator and denominator for one shard of output."""
    logits = top_out
    logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:-1])
    targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
    weights = self.targets_weights_fn(targets)
    # Shift targets by 0.5 so later just casting to int gives the prediction.
    # So for int targets, say 0 and 7, we actually train to predict 0.5 and 7.5.
    # Later (in merics or infer) this is cast to int anyway. Also, we have no
    # loss beyond self.cutoff = 0.2 as these are already correct predictions.
    targets = tf.to_float(targets) + 0.5
    loss = self.internal_loss(logits, targets)
    return tf.reduce_sum(loss * weights), tf.reduce_sum(weights)


class VideoModalityL2(VideoModalityL1):
  """Modality for videos with L2 loss."""

  def internal_loss(self, logits, targets):
    return tf.nn.relu(
        tf.squared_difference(logits, targets) - self.cutoff * self.cutoff)


class VideoModalityL2Raw(VideoModalityL2):
  """Modality with L2 loss and raw input (sequences of frames)."""

  def convert_rgb_to_real(self, prediction, targets):
    """Convert prediction and target from rgb to real."""
    prediction = tf.squeeze(prediction, axis=-1)
    prediction = common_layers.convert_rgb_to_real(prediction)
    targets = common_layers.convert_rgb_to_real(targets)
    return prediction, targets

  def bottom(self, x):
    common_video.gif_summary("inputs", x)
    return common_layers.convert_rgb_to_real(x)

  def targets_bottom(self, x):  # pylint: disable=arguments-differ
    common_video.gif_summary("targets_bottom", x)
    return common_layers.convert_rgb_to_real(x)

  def top(self, body_output, _):
    frames = body_output
    if isinstance(body_output, list):
      frames = tf.stack(body_output, axis=1)
    rgb_frames = common_layers.convert_real_to_rgb(frames)
    common_video.gif_summary("body_output", rgb_frames)
    return tf.expand_dims(rgb_frames, axis=-1)

  def loss(self, top_out, targets):
    prediction, groundtruth = self.convert_rgb_to_real(top_out, targets)
    loss = tf.losses.mean_squared_error(prediction, groundtruth)
    return loss, tf.constant(1.0)


class VideoModalityL1Raw(VideoModalityL2Raw):
  """Modality with L1 loss and raw input (sequences of frames)."""

  def loss(self, top_out, targets):
    prediction, groundtruth = self.convert_rgb_to_real(top_out, targets)
    loss = tf.losses.absolute_difference(prediction, groundtruth)
    return loss, tf.constant(1.0)


class ClassLabelModality(modality.Modality):
  """Used for label data."""

  @property
  def name(self):
    return "class_label_modality_%d_%d" % (self._vocab_size,
                                           self._model_hparams.hidden_size)

  def bottom(self, x):
    with tf.variable_scope(self.name):
      multiplier = 1.0
      if self._model_hparams.multiply_embedding_mode == "sqrt_depth":
        multiplier = self._model_hparams.hidden_size**0.5
      return common_layers.embedding(x,
                                     self._vocab_size,
                                     self._model_hparams.hidden_size,
                                     multiplier=multiplier)

  def targets_bottom(self, x):
    with tf.variable_scope(self.name):
      return tf.zeros([common_layers.shape_list(x)[0],
                       1,
                       1,
                       self._model_hparams.hidden_size])

  def top(self, body_output, _):
    """Transform inputs from model space to target space.

    Average over inner dims and a linear layer to logits.

    Args:
      body_output: A Tensor with shape [batch, ?, ?, body_output_size].

    Returns:
      a Tensors, each with shape [batch_size, 1, 1, 1, vocab_size]
    """
    with tf.variable_scope(self.name):
      x = body_output
      x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
      res = tf.layers.dense(x, self._vocab_size)
      return tf.expand_dims(res, 3)


class VideoModalityIdentity(VideoModality):
  """Video Modality where top and bottom is an identity function."""

  def bottom(self, x):
    common_video.gif_summary("inputs", x, max_outputs=1)
    return x

  def targets_bottom(self, x):
    common_video.gif_summary("targets", x, max_outputs=1)
    return x

  def top(self, body_output, targets):
    return body_output

  def loss(self, top_out, targets):
    """Compute loss numerator and denominator for one shard of output."""
    # TODO(nikip): Try L2 loss
    logits = top_out
    logits = tf.reshape(logits, [-1] + common_layers.shape_list(logits)[2:])
    targets = tf.reshape(targets, [-1] + common_layers.shape_list(targets)[2:])
    cutoff = getattr(self._model_hparams, "video_modality_loss_cutoff", 0.01)
    return common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        cutoff=cutoff,
        weights_fn=self.targets_weights_fn)


class MultiLabelModality(ClassLabelModality):
  """Used for multi label task."""

  @property
  def targets_weights_fn(self):
    """Target weight function for multi label, defaults to nonzero labels."""
    return common_layers.weights_nonzero

  def loss(self, top_out, targets):
    """Average loss over the labels."""
    logits = top_out
    num_labels = tf.shape(targets)[1]
    logits = tf.tile(logits, [1, num_labels, 1, 1, 1])

    xent, weights = common_layers.padded_cross_entropy(
        logits,
        targets,
        self._model_hparams.label_smoothing,
        weights_fn=self.targets_weights_fn,
        reduce_sum=False,
    )
    xent = tf.squeeze(xent, [2, 3])
    weights = tf.squeeze(weights, [2, 3])
    # average loss over all labels
    loss = tf.reduce_sum(xent, axis=1)
    weights = tf.reduce_sum(weights, axis=1)
    loss /= (weights + 1e-8)
    weights = tf.to_float(tf.greater(weights, 0.))

    return tf.reduce_sum(loss*weights), tf.reduce_sum(weights)


class OneHotClassLabelModality(ClassLabelModality):
  """Used for one-hot encoded class labels."""

  def loss(self, top_out, targets):
    """Apply softmax cross-entropy between outputs and targets.

    Args:
      top_out: logits Tensor with shape [batch, ?, ?, num_classes]
      targets: one-hot encoding Tensor with shape [batch, ?, ?, num_classes]
    Returns:
      loss_scale (cross-entropy), loss_denom
    """
    loss_scale = tf.losses.softmax_cross_entropy(
        onehot_labels=targets, logits=top_out)
    weights = self.targets_weights_fn(targets)
    loss_denom = tf.reduce_sum(weights)
    return loss_scale, loss_denom


class IdentityModality(modality.Modality):
  """Does nothing."""

  def bottom(self, x):
    return tf.to_float(x)

  def top(self, body_output, _):
    return body_output


class GenericL2LossModality(IdentityModality):
  """Generic modality with L2 as Loss."""

  def targets_bottom(self, x):
    return tf.to_float(x)

  def loss(self, body_output, targets):
    loss = tf.squared_difference(body_output, tf.to_float(targets))
    return tf.reduce_mean(loss), tf.constant(1.0)


class RealModality(modality.Modality):
  """Base class for real (i.e. float) vectors.

  * Bottom is a linear projection layer to hparams.hidden_size.
  * Top is a linear projection layer to vocab_size.
  """

  @property
  def top_is_pointwise(self):
    return True

  def bottom(self, x):
    with tf.variable_scope("real"):
      return tf.layers.dense(
          tf.to_float(x), self._model_hparams.hidden_size, name="bottom")

  def top(self, body_output, _):
    with tf.variable_scope("real"):
      return tf.layers.dense(body_output, self._vocab_size, name="top")

  def loss(self, top_out, targets):
    raise NotImplementedError()


class RealL2LossModality(RealModality):
  """Modality for real (i.e. float) vectors with L2 (Gaussian) loss."""

  def loss(self, top_out, targets):
    predictions = top_out
    if (len(common_layers.shape_list(top_out)) != len(
        common_layers.shape_list(targets))):
      predictions = tf.squeeze(top_out, axis=[-1])
    with tf.name_scope("l2"):
      weights = self.targets_weights_fn(targets)
      l2 = tf.pow(predictions - targets, 2)
      return tf.reduce_sum(l2 * weights), tf.reduce_sum(weights)


class RealLogPoissonLossModality(RealModality):
  """Modality for real (i.e. float) vectors with log Poisson regression loss."""

  def loss(self, top_out, targets):
    predictions = top_out
    if (len(common_layers.shape_list(top_out)) != len(
        common_layers.shape_list(targets))):
      predictions = tf.squeeze(top_out, axis=[-1])
    with tf.name_scope("log_possion"):
      weights = self.targets_weights_fn(targets)
      lp_loss = tf.nn.log_poisson_loss(targets, predictions)
      return tf.reduce_sum(lp_loss * weights), tf.reduce_sum(weights)


class IdentitySymbolModality(SymbolModality):
  """Symbol modality with identity top and bottom transformations.

  Uses the weights_fn from SymbolModality so that loss/metrics ignore padding.
  """

  def bottom(self, x):
    return tf.to_float(x)

  def top(self, body_output, _):
    return body_output

  def targets_bottom(self, x):
    """SymbolModality overrides targets_bottom, so need to override here too."""
    return self.bottom(x)

  @property
  def top_is_pointwise(self):
    # pointwise mode manipulates body output, not logits, so it fails here.
    return False


class SigmoidClassLabelModality(ClassLabelModality):
  """Sigmoid cross-entropy for independent class labels."""

  @property
  def name(self):
    return "sigmoid_class_symbol_modality_%d_%d" % (
        self._vocab_size, self._model_hparams.hidden_size)

  def loss(self, top_out, targets):
    # Expect inputs of size [batch-size, timesteps, 1, num-classes], where the
    # last dimension of num-classes represents logits for binary labels
    loss_scale = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=targets, logits=top_out)
    # Weigh all classes equally
    weights = self.targets_weights_fn(targets)
    loss_denom = tf.reduce_sum(weights)
    return loss_scale, loss_denom


class SigmoidMaxPoolingClassLabelModality(ClassLabelModality):
  """Sigmoid cross-entropy applied on max-pooling over timesteps."""

  @property
  def name(self):
    return "sigmoid_max_pooling_class_symbol_modality_%d_%d" % (
        self._vocab_size, self._model_hparams.hidden_size)

  def top(self, body_output, _):
    """Transform inputs from model space to target space.

    Average over inner dims and a linear layer to logits.

    Args:
      body_output: A Tensor with shape [batch, timesteps, 1, body_output_size].

    Returns:
      a Tensors, each with shape [batch_size, 1, 1, vocab_size]
    """
    with tf.variable_scope(self.name):
      x = body_output
      x = tf.reduce_max(x, axis=1, keepdims=True)
      return tf.layers.dense(x, self._vocab_size)

  def loss(self, top_out, targets):
    # Expect inputs of size [batch-size, 1, 1, num-classes], where the
    # last dimension of num-classes represents logits for binary labels
    loss_scale = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=targets, logits=top_out)
    # Weigh all classes equally
    weights = self.targets_weights_fn(targets)
    loss_denom = tf.reduce_sum(weights)
    return loss_scale, loss_denom


class SoftmaxMaxPoolingClassLabelModality(OneHotClassLabelModality):
  """Softmax cross-entropy applied on max-pooling over timesteps."""

  @property
  def name(self):
    return "softmax_max_pooling_onehot_class_label_modality_%d_%d" % (
        self._vocab_size, self._model_hparams.hidden_size)

  def top(self, body_output, _):
    with tf.variable_scope(self.name):
      x = body_output
      x = tf.reduce_max(x, axis=1, keepdims=True)
      return tf.layers.dense(x, self._vocab_size)


class SoftmaxAveragePoolingClassLabelModality(OneHotClassLabelModality):
  """Softmax cross-entropy applied on average-pooling over timesteps."""

  @property
  def name(self):
    return "softmax_average_pooling_onehot_class_label_modality_%d_%d" % (
        self._vocab_size, self._model_hparams.hidden_size)

  def top(self, body_output, _):
    with tf.variable_scope(self.name):
      x = body_output
      x = tf.reduce_mean(x, axis=1, keepdims=True)
      return tf.layers.dense(x, self._vocab_size)


class SoftmaxLastTimestepClassLabelModality(OneHotClassLabelModality):
  """Softmax cross-entropy applied on last-timestep encoding."""

  @property
  def name(self):
    return "softmax_last_timestep_onehot_class_label_modality_%d_%d" % (
        self._vocab_size, self._model_hparams.hidden_size)

  def top(self, body_output, _):
    with tf.variable_scope(self.name):
      x = body_output
      x = tf.expand_dims(x[:, -1], 1)  # Pick the last timestep
      return tf.layers.dense(x, self._vocab_size)


class ModalityType(object):
  """Types of modalities."""

  SYMBOL = "SymbolModality"
  SYMBOL_WEIGHTS_ALL = "SymbolModalityWeightsAll"
  SYMBOL_ONE_HOT = "SymbolModalityOneHot"
  CTC_SYMBOL = "CTCSymbolModality"
  IMAGE = "ImageModality"
  IMAGE_CHANNEL_COMPRESS = "ImageChannelCompressModality"
  IMAGE_CHANNEL_BOTTOM_IDENTITY = "ImageChannelBottomIdentityModality"
  IMAGE_CHANNEL_EMBEDDINGS_BOTTOM = "ImageChannelEmbeddingsBottom"
  AUDIO = "AudioModality"
  AUDIO_SPECTRAL = "AudioSpectralModality"
  SPEECH_RECOGNITION = "SpeechRecognitionModality"
  VIDEO = "VideoModality"
  VIDEO_BITWISE = "VideoModalityBitwise"
  VIDEO_PIXEL_NOISE = "VideoModalityPixelNoise"
  VIDEO_L1 = "VideoModalityL1"
  VIDEO_L2 = "VideoModalityL2"
  VIDEO_L2_RAW = "VideoModalityL2Raw"
  VIDEO_L1_RAW = "VideoModalityL1Raw"
  CLASS_LABEL = "ClassLabelModality"
  VIDEO_IDENTITY = "VideoModalityIdentity"
  MULTI_LABEL = "MultiLabelModality"
  ONE_HOT_CLASS_LABEL = "OneHotClassLabelModality"
  IDENTITY = "IdentityModality"
  GENERIC_L2_LOSS = "GenericL2LossModality"
  REAL = "RealModality"
  REAL_L2_LOSS = "RealL2LossModality"
  REAL_LOG_POISSON_LOSS = "RealLogPoissonLossModality"
  IDENTITY_SYMBOL = "IdentitySymbolModality"
  SIGMOID_CLASS_LABEL = "SigmoidClassLabelModality"
  SIGMOID_MAX_POOLING_CLASS_LABEL = "SigmoidMaxPoolingClassLabelModality"
  SOFTMAX_MAX_POOLING_CLASS_LABEL = "SoftmaxMaxPoolingClassLabelModality"
  SOFTMAX_AVERAGE_POOLING_CLASS_LABEL = "SoftmaxAveragePoolingClassLabelModality"
  SOFTMAX_LAST_TIMESTEP_CLASS_LABEL = "SoftmaxLastTimestepClassLabelModality"

  @staticmethod
  def get_choices():
    return [
        ModalityType.SYMBOL,
        ModalityType.SYMBOL_WEIGHTS_ALL,
        ModalityType.SYMBOL_ONE_HOT,
        ModalityType.CTC_SYMBOL,
        ModalityType.IMAGE,
        ModalityType.IMAGE_CHANNEL_COMPRESS,
        ModalityType.IMAGE_CHANNEL_BOTTOM_IDENTITY,
        ModalityType.IMAGE_CHANNEL_EMBEDDINGS_BOTTOM,
        ModalityType.AUDIO,
        ModalityType.AUDIO_SPECTRAL,
        ModalityType.SPEECH_RECOGNITION,
        ModalityType.VIDEO,
        ModalityType.VIDEO_BITWISE,
        ModalityType.VIDEO_PIXEL_NOISE,
        ModalityType.VIDEO_L1,
        ModalityType.VIDEO_L2,
        ModalityType.VIDEO_L2_RAW,
        ModalityType.VIDEO_L1_RAW,
        ModalityType.CLASS_LABEL,
        ModalityType.VIDEO_IDENTITY,
        ModalityType.MULTI_LABEL,
        ModalityType.ONE_HOT_CLASS_LABEL,
        ModalityType.IDENTITY,
        ModalityType.GENERIC_L2_LOSS,
        ModalityType.REAL,
        ModalityType.REAL_L2_LOSS,
        ModalityType.REAL_LOG_POISSON_LOSS,
        ModalityType.IDENTITY_SYMBOL,
        ModalityType.SIGMOID_CLASS_LABEL,
        ModalityType.SIGMOID_MAX_POOLING_CLASS_LABEL,
        ModalityType.SOFTMAX_MAX_POOLING_CLASS_LABEL,
        ModalityType.SOFTMAX_AVERAGE_POOLING_CLASS_LABEL,
        ModalityType.SOFTMAX_LAST_TIMESTEP_CLASS_LABEL,
    ]
