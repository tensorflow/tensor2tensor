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

"""Utils for latent variable models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range  # pylint: disable=redefined-builtin
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import transformer_layers
from tensor2tensor.utils import beam_search

import tensorflow as tf
import tensorflow_probability as tfp

DO_SUMMARIES = True


def compress_self_attention_layer(x, hparams, name=None):
  """Attend function."""
  with tf.variable_scope(name, default_name="compress_self_attention"):
    x, xshape, _ = cia.maybe_reshape_4d_to_3d(x)
    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams),
        None,
        None,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.num_heads,
        hparams.attention_dropout)
    res = common_layers.layer_postprocess(x, y, hparams)
    return tf.reshape(res, xshape)


def compute_nats_and_bits_per_dim(data_dim,
                                  latent_dim,
                                  average_reconstruction,
                                  average_prior):
  """Computes negative ELBO, which is an upper bound on the negative likelihood.

  Args:
    data_dim: int-like indicating data dimensionality.
    latent_dim: int-like indicating latent dimensionality.
    average_reconstruction: Scalar Tensor indicating the reconstruction cost
      averaged over all data dimensions and any data batches.
    average_prior: Scalar Tensor indicating the negative log-prior probability
      averaged over all latent dimensions and any data batches.

  Returns:
    Tuple of scalar Tensors, representing the nats and bits per data dimension
    (e.g., subpixels) respectively.
  """
  with tf.name_scope(None, default_name="compute_nats_per_dim"):
    data_dim = tf.cast(data_dim, average_reconstruction.dtype)
    latent_dim = tf.cast(latent_dim, average_prior.dtype)
    negative_log_likelihood = data_dim * average_reconstruction
    negative_log_prior = latent_dim * average_prior
    negative_elbo = negative_log_likelihood + negative_log_prior
    nats_per_dim = tf.divide(negative_elbo, data_dim, name="nats_per_dim")
    bits_per_dim = tf.divide(nats_per_dim, tf.log(2.), name="bits_per_dim")
    return nats_per_dim, bits_per_dim


def multinomial_sample(x, vocab_size=None, sampling_method="random",
                       temperature=1.0):
  """Multinomial sampling from a n-dimensional tensor.

  Args:
    x: Tensor of shape [..., vocab_size]. Parameterizes logits of multinomial.
    vocab_size: Number of classes in multinomial distribution.
    sampling_method: String, "random" or otherwise deterministic.
    temperature: Positive float.

  Returns:
    Tensor of shape [...].
  """
  vocab_size = vocab_size or common_layers.shape_list(x)[-1]
  if sampling_method == "random" and temperature > 0.0:
    samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]) / temperature, 1)
  else:
    samples = tf.argmax(x, axis=-1)
  reshaped_samples = tf.reshape(samples, common_layers.shape_list(x)[:-1])
  return reshaped_samples


def ae_latent_softmax(latents_pred, latents_discrete_hot, vocab_size, hparams):
  """Latent prediction and loss.

  Args:
    latents_pred: Tensor of shape [..., depth].
    latents_discrete_hot: Tensor of shape [..., vocab_size].
    vocab_size: an int representing the vocab size.
    hparams: HParams.

  Returns:
    sample: Tensor of shape [...], a sample from a multinomial distribution.
    loss: Tensor of shape [...], the softmax cross-entropy.
  """
  with tf.variable_scope("latent_logits"):
    latents_logits = tf.layers.dense(latents_pred, vocab_size,
                                     name="logits_dense")
    if hparams.logit_normalization:
      latents_logits *= tf.rsqrt(1e-8 +
                                 tf.reduce_mean(tf.square(latents_logits)))
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=latents_discrete_hot, logits=latents_logits)

    # TODO(trandustin): tease this out from ae_latent_softmax.
    # we use just the loss portion to anchor prior / encoder on text.
    sample = multinomial_sample(latents_logits,
                                vocab_size,
                                hparams.sampling_method,
                                hparams.sampling_temp)
    return sample, loss


def ae_latent_sample_beam(latents_dense_in, inputs, ed, embed, hparams):
  """Samples from the latent space in the autoencoder.

  Args:
    latents_dense_in: Tensor of shape [batch, length_q, ...]. Only the shape of
      its first two dimensions are used. length_q is the latent length, which is
      height * width * hparams.num_latents / (2**hparams.num_compress_steps).
    inputs: Tensor of shape [batch, length_kv, hparams.hidden_size]. Encodings
      to attend to in decoder.
    ed: Tensor which broadcasts with shape [batch, hparams.num_heads, length_q,
      length_kv]. Encoder-decoder attention bias.
    embed: Callable which embeds discrete latent hot-vectors and a hidden size
      and returns dense vectors.
    hparams: HParams.

  Returns:
    Tensor of shape [batch, length].
  """

  def symbols_to_logits_fn(ids):
    """Go from ids to logits."""
    ids = tf.expand_dims(ids, axis=2)  # Ids start with added all-zeros.
    latents_discrete = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0]])

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
      latents_dense = embed(
          tf.one_hot(latents_discrete, depth=2**hparams.bottleneck_bits),
          hparams.hidden_size)
      latents_pred = transformer_latent_decoder(
          latents_dense, inputs, ed, hparams, name="latent_prediction")
      logits = tf.layers.dense(
          latents_pred, 2**hparams.bottleneck_bits, name="logits_dense")
      current_output_position = common_layers.shape_list(ids)[1] - 1
      logits = logits[:, current_output_position, :]
    return logits

  initial_ids = tf.zeros([tf.shape(latents_dense_in)[0]], dtype=tf.int32)
  length = tf.shape(latents_dense_in)[1]
  ids, _, _ = beam_search.beam_search(
      symbols_to_logits_fn,
      initial_ids,
      1,
      length,
      2**hparams.bottleneck_bits,
      alpha=0.0,
      eos_id=-1,
      stop_early=False)

  res = tf.expand_dims(ids[:, 0, :], axis=2)  # Pick first beam.
  return res[:, 1:]  # Remove the added all-zeros from ids.


def residual_block_layer(inputs, hparams):
  """Residual block over inputs.

  Runs a residual block consisting of
    conv: kernel_size x kernel_size
    conv: 1x1
    dropout, add and normalize according to hparams.layer_postprocess_sequence.

  Args:
    inputs: Tensor of shape [batch, height, width, hparams.hidden_size].
    hparams: HParams.

  Returns:
    Tensor of shape [batch, height, width, hparams.hidden_size].
  """
  kernel = (hparams.res_kernel_size, hparams.res_kernel_size)
  x = inputs
  for i in range(hparams.num_res_layers):
    with tf.variable_scope("res_conv_%d" % i):
      # kernel_size x kernel_size conv block
      y = common_layers.conv_block(
          common_layers.layer_norm(x, hparams.hidden_size, name="lnorm"),
          hparams.hidden_size, [((1, 1), kernel)],
          strides=(1, 1),
          padding="SAME",
          name="residual_conv")
      # 1x1 conv block
      y = common_layers.conv_block(
          y,
          hparams.hidden_size, [((1, 1), (1, 1))],
          strides=(1, 1),
          padding="SAME",
          name="residual_dense")
      x = common_layers.layer_postprocess(x, y, hparams)
  return x


def compress_encoder(inputs,
                     hparams,
                     strides=(2, 2),
                     kernel_size=(3, 3),
                     name=None):
  """Encoder that compresses 2-D inputs by 2**num_compress_steps.

  Args:
    inputs: Tensor of shape [batch, height, width, channels].
    hparams: HParams.
    strides: Tuple, strides for conv block.
    kernel_size: Tuple, kernel window size for conv block.
    name: string, variable scope.

  Returns:
    Tensor of shape [batch, latent_length, hparams.hidden_size], where
      latent_length is
      hparams.num_latents * (height*width) / 2**(hparams.num_compress_steps).
  """
  with tf.variable_scope(name, default_name="compress"):
    x = inputs
    for i in range(hparams.num_compress_steps // 2):
      with tf.variable_scope("compress_conv_%d" % i):
        y = common_layers.conv_block(
            common_layers.layer_norm(
                x, hparams.hidden_size, name="lnorm"),
            hparams.hidden_size,
            dilation_rates_and_kernel_sizes=[((1, 1), kernel_size)],
            strides=strides,
            padding="SAME",
            name="compress_conv_%d" % i)
        y = tf.nn.dropout(y, 1.0 - hparams.dropout)
        if hparams.do_compress_attend:
          y = compress_self_attention_layer(
              x, hparams, name="compress_selfatt_%d" % i)
          y += x
        x = y

    x = residual_block_layer(x, hparams)

    # If using multiple copies of latents, blow up the hidden size and then
    # reshape to increase by num_latents.
    shape_x = common_layers.shape_list(x)
    x = tf.layers.dense(x,
                        hparams.num_latents * hparams.hidden_size,
                        name=name + "_dense")
    return tf.reshape(x, [shape_x[0],
                          shape_x[1] * shape_x[2] * hparams.num_latents,
                          hparams.hidden_size])


def compress_encoder_2d(x, hparams, name=None):
  """Encoder that compresses 2-D inputs by 2**num_compress_steps.

  Args:
    x: Tensor of shape [batch, height, width, channels].
    hparams: HParams.
    name: string, variable scope.

  Returns:
    Tensor of shape [batch, latent_length, hparams.hidden_size], where
      latent_length is
      hparams.num_latents * (height*width) / 2**(hparams.num_compress_steps).
  """
  return compress_encoder(
      x,
      hparams,
      strides=(2, 2),
      kernel_size=(hparams.kernel_size, hparams.kernel_size),
      name=name)


def compress_encoder_1d(x, hparams, name=None):
  """Encoder that compresses 1-D inputs by 2**num_compress_steps.

  Args:
    x: Tensor of shape [batch, length, channels].
    hparams: HParams.
    name: string, variable scope.

  Returns:
    Tensor of shape [batch, latent_length, hparams.hidden_size], where
      latent_length is
      hparams.num_latents * length / 2**hparams.num_compress_steps.
  """
  x = tf.expand_dims(x, axis=2)
  return compress_encoder(x,
                          hparams,
                          strides=(2, 1),
                          kernel_size=(hparams.kernel_size, 1),
                          name=name)


def decompress_decoder(inputs,
                       hparams,
                       strides=(2, 2),
                       kernel=(3, 3),
                       name=None):
  """Decoder that decompresses 2-D inputs by 2**num_compress_steps.

  Args:
    inputs: Tensor of shape [batch, compress_height, compress_width, channels].
    hparams: HParams.
    strides: Tuple, strides for conv block.
    kernel: Tuple, kernel window size for conv block.
    name: string, variable scope.

  Returns:
    Tensor of shape [batch, height, width, hparams.hidden_size].
  """
  with tf.variable_scope(name, default_name="decompress"):
    x = inputs
    x = tf.layers.dense(x, hparams.hidden_size, name=name + "_dense")
    x = residual_block_layer(x, hparams)
    for i in range(hparams.num_compress_steps // 2):
      j = hparams.num_compress_steps // 2 - i - 1
      with tf.variable_scope(name + "_%d" % j):
        if hparams.do_decompress_attend:
          y = compress_self_attention_layer(
              x, hparams, name="decompress_selfatt")
          x += y
        y = tf.layers.conv2d_transpose(
            x,
            hparams.hidden_size,
            kernel,
            strides=strides,
            padding="SAME",
            activation=tf.nn.relu if i > 0 else None,
            name="decompress_conv")
        x = y
    return x


def decompress_decoder_2d(x, hparams, name=None):
  """Decoder that decompresses 2-D inputs by 2**num_compress_steps.

  Args:
    x: Tensor of shape [batch, compress_height, compress_width, channels].
    hparams: HParams.
    name: string, variable scope.

  Returns:
    Tensor of shape [batch, height, width, hparams.hidden_size].
  """
  return decompress_decoder(x, hparams,
                            strides=(2, 2),
                            kernel=(hparams.kernel_size, hparams.kernel_size),
                            name=name)


def decompress_decoder_1d(x, hparams, name=None):
  """Decoder that decompresses 1-D inputs by 2**num_compress_steps.

  Args:
    x: Tensor of shape [batch, compress_length, channels].
    hparams: HParams.
    name: string, variable scope.

  Returns:
    Tensor of shape [batch, length, hparams.hidden_size].
  """
  x = tf.expand_dims(x, axis=2)
  output = decompress_decoder(x, hparams,
                              strides=(2, 1),
                              kernel=(hparams.kernel_size, 1),
                              name=name)
  return tf.squeeze(output, axis=2)


def transformer_text_encoder(inputs,
                             target_space,
                             hparams,
                             name=None):
  """Transformer text encoder over inputs with unmasked full attention.

  Args:
    inputs: Tensor of shape [batch, length, 1, hparams.hidden_size].
    target_space: int. Used for encoding inputs under a target space id.
    hparams: HParams.
    name: string, variable scope.

  Returns:
    encoder_output: Tensor of shape [batch, length, hparams.hidden_size].
    ed: Tensor of shape [batch, 1, 1, length]. Encoder-decoder attention bias
      for any padded tokens.
  """
  with tf.variable_scope(name, default_name="transformer_text_encoder"):
    inputs = common_layers.flatten4d3d(inputs)
    [
        encoder_input,
        encoder_self_attention_bias,
        ed,
    ] = transformer_layers.transformer_prepare_encoder(
        inputs, target_space=target_space, hparams=hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.dropout)
    encoder_output = transformer_layers.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams)
    return encoder_output, ed


def transformer_image_decoder(targets,
                              encoder_output,
                              ed_attention_bias,
                              hparams,
                              name=None):
  """Transformer image decoder over targets with local attention.

  Args:
    targets: Tensor of shape [batch, ...], and whose size is batch * height *
      width * hparams.num_channels * hparams.hidden_size.
    encoder_output: Tensor of shape [batch, length_kv, hparams.hidden_size].
    ed_attention_bias: Tensor which broadcasts with shape [batch,
      hparams.num_heads, length_q, length_kv]. Encoder-decoder attention bias.
    hparams: HParams.
    name: string, variable scope.

  Returns:
    Tensor of shape [batch, height, width * hparams.num_channels,
    hparams.hidden_size].
  """
  with tf.variable_scope(name, default_name="transformer_dec"):
    batch_size = common_layers.shape_list(targets)[0]
    targets = tf.reshape(targets, [batch_size,
                                   hparams.img_len,
                                   hparams.img_len,
                                   hparams.num_channels * hparams.hidden_size])
    decoder_input, _, _ = cia.prepare_decoder(targets, hparams)
    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        encoder_output,
        hparams.num_decoder_layers or hparams.num_hidden_layers,
        hparams,
        attention_type=hparams.dec_attention_type,
        encoder_decoder_attention_bias=ed_attention_bias,
        name="decoder")
    decoder_output = tf.reshape(decoder_output,
                                [batch_size,
                                 hparams.img_len,
                                 hparams.img_len * hparams.num_channels,
                                 hparams.hidden_size])
    return decoder_output


def transformer_latent_decoder(x,
                               encoder_output,
                               ed_attention_bias,
                               hparams,
                               name=None):
  """Transformer decoder over latents using latent_attention_type.

  Args:
    x: Tensor of shape [batch, length_q, hparams.hidden_size]. length_q is the
      latent length, which is
      height * width * hparams.num_latents / (2**hparams.num_compress_steps).
    encoder_output: Tensor of shape [batch, length_kv, hparams.hidden_size].
    ed_attention_bias: Tensor which broadcasts with shape [batch,
      hparams.num_heads, length_q, length_kv]. Encoder-decoder attention bias.
    hparams: HParams.
    name: string, variable scope.

  Returns:
    Tensor of shape [batch, length_q, hparams.hidden_size].
  """
  with tf.variable_scope(name, default_name="transformer_latent_dec"):
    batch_size = common_layers.shape_list(x)[0]
    compressed_img_len = (hparams.img_len //
                          2**(hparams.num_compress_steps // 2))
    x = tf.reshape(x, [batch_size,
                       compressed_img_len,
                       compressed_img_len * hparams.num_latents,
                       hparams.hidden_size])
    decoder_input, _, _ = cia.prepare_decoder(x, hparams)
    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        encoder_output,
        hparams.num_latent_layers or hparams.num_hidden_layers,
        hparams,
        attention_type=hparams.latent_attention_type,
        encoder_decoder_attention_bias=ed_attention_bias,
        name="decoder")
    decoder_output = tf.reshape(decoder_output,
                                [batch_size,
                                 compressed_img_len**2 * hparams.num_latents,
                                 hparams.hidden_size])
    return decoder_output


def bottleneck_layer(inputs,
                     hparams,
                     name="discrete_bottleneck"):
  """Computes latents given inputs (typically, compressed targets)."""
  [
      latents_dense,
      latents_discrete,
      extra_loss,
      embed_fn,
      _,
  ] = hparams.bottleneck(inputs=inputs,
                         filter_size=hparams.compress_filter_size,
                         name=name,
                         mode=hparams.mode)
  if DO_SUMMARIES:
    tf.summary.histogram("discrete_latents",
                         tf.reshape(latents_discrete, [-1]))
  return latents_dense, latents_discrete, extra_loss, embed_fn


def latent_prediction_model(inputs,
                            ed_attention_bias,
                            latents_discrete,
                            latents_dense,
                            hparams,
                            vocab_size=None,
                            name=None):
  """Transformer-based latent prediction model.

  It is an autoregressive decoder over latents_discrete given inputs.

  Args:
    inputs: Tensor of shape [batch, length_kv, hparams.hidden_size]. Inputs to
      attend to for the decoder on latents.
    ed_attention_bias: Tensor which broadcasts with shape [batch,
      hparams.num_heads, length_q, length_kv]. Encoder-decoder attention bias.
    latents_discrete: Tensor of shape [batch, length_q, vocab_size].
      One-hot latents to compute log-probability of given inputs.
    latents_dense: Tensor of shape [batch, length_q, hparams.hidden_size].
      length_q is the latent length, which is
      height * width * hparams.num_latents / (2**hparams.num_compress_steps).
    hparams: HParams.
    vocab_size: int or None. If None, it is 2**hparams.bottleneck_bits.
    name: string, variable scope.

  Returns:
    latents_pred: Tensor of shape [batch, length_q, hparams.hidden_size].
    latents_pred_loss: Tensor of shape [batch, length_q].
  """
  with tf.variable_scope(name, default_name="latent_prediction"):
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      latents_pred = transformer_latent_decoder(tf.stop_gradient(latents_dense),
                                                inputs,
                                                ed_attention_bias,
                                                hparams,
                                                name)
      if vocab_size is None:
        vocab_size = 2**hparams.bottleneck_bits
      if not hparams.soft_em:
        # TODO(trandustin): latents_discrete is not one-hot from
        # discrete_bottleneck unless hparams.soft_em is True. Refactor.
        latents_discrete = tf.one_hot(latents_discrete, depth=vocab_size)
      _, latent_pred_loss = ae_latent_softmax(
          latents_pred, tf.stop_gradient(latents_discrete), vocab_size, hparams)
  return latents_pred, latent_pred_loss


def transformer_autoencoder(inputs,
                            targets,
                            target_space,
                            hparams,
                            cache=None,
                            predict_mask=1.0):
  """Auto-encoder using a Transformer decoder and a prior over latent sequences.

  Args:
    inputs: Tensor of shape [batch, length, 1, hparams.hidden_size] or None.
    targets: Tensor of shape [batch, ..., channels]. Ellipses may be 1 or 2
      dimensions denoting sequence length.
    target_space: int. Used for encoding inputs under a target space id.
    hparams: HParams.
    cache: Tensor of shape [batch, length] or None.
    predict_mask: Tensor masking whether to use gold targets or predictions.

  Returns:
    decoder_output: Tensor of shape [batch, ..., hparams.hidden_size] presenting
      pre-logit activations. After a transformation (`top` in `T2TModel`), it is
      used with targets to compute the "training" (reconstruction) loss.
    losses: dict of str to Tensors. There are three loss terms: "extra",
      "extra_loss", and "latent_pred". The first is hard-coded to 0. The latter
      two are Tensors of shape [batch].
    cache: Tensor of shape [batch, length], either the same as cache, or newly
      computed if the cache input is None.
  """
  original_targets_shape = common_layers.shape_list(targets)
  batch_size = original_targets_shape[0]
  if len(original_targets_shape) == 4:
    compress_fn = compress_encoder_2d
    decompress_fn = decompress_decoder_2d
  else:
    compress_fn = compress_encoder_1d
    decompress_fn = decompress_decoder_1d

  ed_attention_bias = None
  if inputs is not None:
    inputs, ed_attention_bias = transformer_text_encoder(
        inputs, target_space, hparams, name="input_encoder")

  losses = {"extra": 0.,
            "extra_loss": 0.,
            "latent_pred": 0.}
  if hparams.mode != tf.estimator.ModeKeys.PREDICT:
    targets_compressed = compress_fn(targets, hparams, name="compress")

    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      scale = common_layers.inverse_exp_decay(hparams.startup_steps)
    else:
      scale = 1.0
    scale = tf.to_float(tf.less(tf.random_uniform([batch_size]), scale))

    latents_dense, latents_discrete, extra_loss, _ = bottleneck_layer(
        targets_compressed, hparams)
    extra_loss = scale * tf.reduce_mean(extra_loss)

    _, latents_pred_loss = latent_prediction_model(
        inputs, ed_attention_bias, latents_discrete, latents_dense, hparams,
        name="latent_pred")
    latent_time = tf.less(hparams.mask_startup_steps,
                          tf.to_int32(tf.train.get_global_step()))
    latents_pred_loss = scale * tf.reduce_mean(latents_pred_loss)
    latents_pred_loss *= tf.to_float(latent_time)

    # Apply dropout noise for each data point and time step.
    latents_dense_shape = common_layers.shape_list(latents_dense)
    latents_dense = tf.nn.dropout(
        latents_dense,
        keep_prob=1 - hparams.latent_dropout,
        noise_shape=[latents_dense_shape[0], latents_dense_shape[1], 1])

    # TODO(trandustin): Can we combine extra and extra_loss?
    losses = {"extra": 0.,
              "extra_loss": extra_loss,
              "latent_pred": latents_pred_loss}
  else:
    # Set the latent length, which is num_latents times the number of latent
    # pixels. The number of latent pixels is determined by a compression factor
    # on the number of image pixels.
    latent_len = ((hparams.img_len * hparams.img_len * hparams.num_latents) /
                  (2**hparams.num_compress_steps))
    _, _, _, embed_fn = bottleneck_layer(targets_compressed, hparams)
    latents_dense = tf.zeros([batch_size, latent_len, 1, hparams.hidden_size])
    if cache is None:
      cache = ae_latent_sample_beam(latents_dense,
                                    inputs,
                                    ed_attention_bias,
                                    embed_fn,
                                    hparams)
    cache_one_hot = tf.one_hot(cache, depth=2**hparams.bottleneck_bits)
    latents_dense = embed_fn(cache_one_hot, hparams.hidden_size)

  if len(original_targets_shape) == 4:
    compressed_img_len = (hparams.img_len //
                          2**(hparams.num_compress_steps // 2))
    latents_dense = tf.reshape(latents_dense,
                               [batch_size,
                                compressed_img_len,
                                compressed_img_len,
                                hparams.num_latents * hparams.hidden_size])

  latents_dense = decompress_fn(latents_dense, hparams, name="decompress")
  latents_dense = tf.reshape(
      latents_dense,
      [-1, hparams.img_len, hparams.img_len, hparams.hidden_size])

  if hparams.use_gold_targets:
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      masking = predict_mask
    else:
      masking = common_layers.inverse_exp_decay(hparams.mask_startup_steps)
    targets, _, _ = cia.maybe_reshape_4d_to_3d(targets)
    mask = tf.less(masking,
                   tf.random_uniform(common_layers.shape_list(targets)[:-1]))
    mask = tf.expand_dims(tf.to_float(mask), 2)
    latents_dense = mask * targets + (1.0 - mask) * latents_dense

  latents_dense = tf.reshape(latents_dense, original_targets_shape)
  if hparams.decode_autoregressive:
    decoder_output = transformer_image_decoder(
        latents_dense, inputs, ed_attention_bias, hparams, name="decoder")
  else:
    decoder_output = latents_dense
  return decoder_output, losses, cache


def iaf_flow(one_hot_assignments,
             scale_weights,
             scale_bias,
             num_codes,
             summary=True,
             name=None):
  """Performs a single IAF flow using scale and normalization transformations.

  Args:
    one_hot_assignments: Assignments Tensor with shape [num_samples, batch_size,
      latent_size, num_codes].
    scale_weights: Tensor corresponding to lower triangular matrix used to
      autoregressively generate scale matrix from assignments. To ensure the
      lower-triangular matrix has length of latent_size, scale_weights should
      be a rank-one tensor with size latent_size * (latent_size + 1) / 2.
    scale_bias: Bias tensor to be added to scale tensor, with shape
      [latent_size, num_codes]. If scale weights are zero, initialize scale_bias
      to be log(exp(1.) / 2. - 1) so initial transformation is identity.
    num_codes: Number of codes in codebook.
    summary: Whether to save summaries.
    name: String used for name scope.

  Returns:
    flow_output: Transformed one-hot assignments.
    inverse_log_det_jacobian: Inverse log deteriminant of Jacobian corresponding
      to transformation.
  """
  with tf.name_scope(name, default_name="iaf"):
    # Pad the one_hot_assignments by zeroing out the first latent dimension and
    # shifting the rest down by one (and removing the last dimension).
    padded_assignments = tf.pad(
        one_hot_assignments, [[0, 0], [0, 0], [1, 0], [0, 0]])[:, :, :-1, :]
    scale_bijector = tfp.distributions.bijectors.Affine(
        scale_tril=tfp.distributions.fill_triangular(scale_weights))
    scale = scale_bijector.forward(
        tf.transpose(padded_assignments, [0, 1, 3, 2]))
    # Transpose the bijector output since it performs a batch matmul.
    scale = tf.transpose(scale, [0, 1, 3, 2])
    scale = tf.nn.softplus(scale)
    scale = scale + tf.nn.softplus(scale_bias[tf.newaxis, tf.newaxis, ...])
    # Don't need last dimension since the transformation keeps it constant.
    scale = scale[..., :-1]

    z = one_hot_assignments[..., :-1]
    unnormalized_probs = tf.concat([z * scale,
                                    one_hot_assignments[..., -1, tf.newaxis]],
                                   axis=-1)
    normalizer = tf.reduce_sum(unnormalized_probs, axis=-1)
    flow_output = unnormalized_probs / (normalizer[..., tf.newaxis])
    inverse_log_det_jacobian = (-tf.reduce_sum(tf.log(scale), axis=-1)
                                + num_codes * tf.log(normalizer))
    if summary:
      tf.summary.histogram("iaf/scale", tf.reshape(scale, [-1]))
      tf.summary.histogram("iaf/inverse_log_det_jacobian",
                           tf.reshape(inverse_log_det_jacobian, [-1]))
    return flow_output, inverse_log_det_jacobian
