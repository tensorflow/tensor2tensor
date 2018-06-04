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
"""Utils for latent variable models."""

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer

import tensorflow as tf

DO_SUMMARIES = True


class Latent(object):
  DISCRETE = "discrete"
  DENSE = "dense"

  @staticmethod
  def get_choices():
    return [
        Latent.DISCRETE,
        Latent.DENSE,
    ]


def add_learned_positional_embeddings(x, hparams):
  pos = tf.get_variable("pos",
                        [1, hparams.img_len*hparams.img_len,
                         1, hparams.hidden_size])
  pos = pos[:, :common_layers.shape_list(x)[1], :, :]
  x = tf.expand_dims(x, axis=2)
  x += pos
  return x


def attend(x, source, hparams, name):
  """Attend function."""
  with tf.variable_scope(name):
    # x = tf.squeeze(x, axis=2)
    x, xshape, _ = cia.maybe_reshape_4d_to_3d(x)
    if len(source.get_shape()) > 3:
      source = tf.squeeze(source, axis=2)
    source = common_attention.add_timing_signal_1d(source)
    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams),
        source,
        None,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.num_heads,
        hparams.attention_dropout)
    res = common_layers.layer_postprocess(x, y, hparams)
    return tf.reshape(res, xshape)


def multinomial_sample(x, vocab_size, temperature):
  """Multinomial sampling from a n-dimensional tensor."""
  if temperature > 0:
    samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]) / temperature, 1)
  else:
    samples = tf.argmax(x, axis=-1)
  reshaped_samples = tf.reshape(samples, common_layers.shape_list(x)[:-1])
  return reshaped_samples


def ae_latent_softmax(latents_pred, latents_discrete, hparams):
  """Latent prediction and loss."""
  vocab_size = 2 ** hparams.z_size
  if hparams.num_decode_blocks < 2:
    with tf.variable_scope("extra_logits"):
      latents_logits = tf.layers.dense(latents_pred, vocab_size,
                                       name="extra_logits")
      if hparams.logit_normalization:
        latents_logits *= tf.rsqrt(1e-8 +
                                   tf.reduce_mean(tf.square(latents_logits)))

      loss = None
      if latents_discrete is not None:
        if hparams.soft_em:
          # latents_discrete is actually one-hot of multinomial samples
          assert hparams.num_decode_blocks == 1
          loss = tf.nn.softmax_cross_entropy_with_logits(
              labels=latents_discrete, logits=latents_logits)
        else:
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=latents_discrete, logits=latents_logits)
      sample = multinomial_sample(latents_logits, vocab_size,
                                  hparams.sampling_temp)
      return sample, loss

  # Multi-block case.
  block_vocab_size = 2**(hparams.z_size // hparams.num_decode_blocks)
  latents_logits = [
      tf.layers.dense(
          latents_pred, block_vocab_size, name="extra_logits_%d" % i)
      for i in range(hparams.num_decode_blocks)
  ]
  loss = None
  if latents_discrete is not None:
    losses = []
    for i in range(hparams.num_decode_blocks):
      d = tf.floormod(tf.floordiv(latents_discrete,
                                  block_vocab_size**i), block_vocab_size)
      losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=d, logits=latents_logits[i]))
    loss = sum(losses)
  samples = [multinomial_sample(l, block_vocab_size, hparams.sampling_temp)
             for l in latents_logits]
  sample = sum([s * block_vocab_size**i for i, s in enumerate(samples)])
  return sample, loss


def residual_block_layer(inputs, hparams):
  """Residual block over inputs.

  Runs a residual block consisting of
    conv: kernel_size x kernel_size
    conv: 1x1
    dropout, add and normalize according to hparams.layer_postprocess_sequence.

  Args:
    inputs: Tensor of shape [batch_size, height, width, hidden_dim].
    hparams: Dict, hyperparameters.

  Returns:
    x: Tensor of shape [batch_size, height, width, hidden_dim]
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


def compress_encoder(inputs, hparams,
                     strides=(2, 2),
                     kernel=(3, 3),
                     name="compress"):
  """Encoder that compresses inputs to length/2**num_compress_steps.

  Args:
    inputs: Tensor of shape [batch, height, width, hidden_dim].
    hparams: Dict, hyperparameters.
    strides: Tuple, strides for conv block.
    kernel: Tuple, kernel window size for conv block.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, height*width/2**(compress_steps), hidden_dim].
  """
  with tf.variable_scope(name):
    x = inputs
    # Compress conv layers with strides and kernels as passed to the function.
    for i in range(hparams.num_compress_steps // 2):
      with tf.variable_scope("compress_conv_%d" % i):
        y = common_layers.conv_block(
            common_layers.layer_norm(
                x, hparams.hidden_size, name="lnorm"),
            hparams.hidden_size, [((1, 1), kernel)],
            strides=strides,
            padding="SAME",
            name="compress_conv_%d" % i)
        y = tf.nn.dropout(y, 1.0 - hparams.dropout)
        x = y

    # Residual blocks.
    x = residual_block_layer(x, hparams)

    # If using multiple copies of latents, blow up the hidden size and then
    # reshape to increase by num_latents.
    shape_x = common_layers.shape_list(x)
    x = tf.layers.dense(x, hparams.num_latents*hparams.hidden_size,
                        name=name + "_dense")
    new_shape = [shape_x[0], shape_x[1] * shape_x[2]*hparams.num_latents,
                 hparams.hidden_size]
    return tf.reshape(x, new_shape)


def compress_encoder_2d(x, hparams, name):
  """Encoder that compresses inputs to height*width/2**num_compress_steps.

  Args:
    x: Tensor of shape [batch, height, width, hidden_dim].
    hparams: Dict, hyperparameters.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, height*width/2**(compress_steps), hidden_dim].
  """
  return compress_encoder(x, hparams,
                          strides=(2, 2),
                          kernel=(hparams.kernel_size, hparams.kernel_size),
                          name=name)


def compress_encoder_1d(x, hparams, name):
  """Encoder that compresses inputs to length/2**num_compress_steps.

  Args:
    x: Tensor of shape [batch, length, hidden_dim].
    hparams: Dict, hyperparameters.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, length/2**(compress_steps), hidden_dim].
  """
  x = tf.expand_dims(x, axis=2)
  return compress_encoder(x, hparams,
                          strides=(2, 1),
                          kernel=(hparams.kernel_size, 1),
                          name=name)


def decompress_decoder(inputs, hparams,
                       strides=(2, 2),
                       kernel=(3, 3),
                       name="decompress"):
  """Encoder that compresses inputs to length/2**num_compress_steps.

  Args:
    inputs: Tensor of shape [batch, compress_height, compress_width, hidden_dim]
    hparams: Dict, hyperparameters.
    strides: Tuple, strides for conv block.
    kernel: Tuple, kernel window size for conv block.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, height, width, hidden_dim].
  """
  with tf.variable_scope(name):
    x = inputs
    # Reshape?
    x = tf.layers.dense(x, hparams.hidden_size, name=name + "_dense")
    # Residual blocks.
    x = residual_block_layer(x, hparams)

    # Decompress conv layers with strides and kernels as passed to the function.
    for i in range(hparams.num_compress_steps // 2):
      j = hparams.num_compress_steps // 2 - i - 1
      with tf.variable_scope(name + "_%d" % j):
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


def decompress_decoder_2d(x, hparams, name):
  """Dencoder that decompresses x to length height*width.

  Args:
    x: Tensor of shape [batch, compress_height, compress_width, hidden_dim].
    hparams: Dict, hyperparameters.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, height, width, hidden_dim].
  """
  return decompress_decoder(x, hparams,
                            strides=(2, 2),
                            kernel=(hparams.kernel_size, hparams.kernel_size),
                            name=name)


def decompress_decoder_1d(x, hparams, name):
  """Dencoder that decompresses x to original target length.

  Args:
    x: Tensor of shape [batch, compress_length, hidden_dim].
    hparams: Dict, hyperparameters.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, length, hidden_dim].
  """
  x = tf.expand_dims(x, axis=2)
  output = decompress_decoder(x, hparams,
                              strides=(2, 1),
                              kernel=(hparams.kernel_size, 1),
                              name=name)
  return tf.squeeze(output, axis=2)


def transformer_text_encoder(inputs, space_id,
                             hparams, name="transformer_text_enc"):
  """Transformer text encoder."""
  with tf.variable_scope(name):
    x = common_layers.flatten4d3d(inputs)
    (encoder_input, encoder_self_attention_bias,
     ed) = transformer.transformer_prepare_encoder(x, space_id, hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.dropout)
    return transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams), ed


def transformer_image_decoder(encoder_output,
                              ed_attention_bias,
                              targets,
                              hparams,
                              name="transformer_dec"):
  """Original Transformer decoder."""
  with tf.variable_scope(name):
    batch_size = common_layers.shape_list(targets)[0]
    # Reshape targets as b, 32, 32, 3*hidden size].
    targets = tf.reshape(targets, [
        batch_size, hparams.img_len, hparams.img_len,
        hparams.num_channels*hparams.hidden_size])

    # Prepare decoder inputs and bias. This also shifts targets and adds 2D
    # position embeddings to target.
    decoder_input, _, _ = cia.prepare_decoder(targets, hparams)
    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        encoder_output,
        hparams.num_decoder_layers or hparams.num_hidden_layers,
        hparams,
        attention_type=hparams.dec_attention_type,
        encoder_decoder_attention_bias=ed_attention_bias,
        name="decoder")
    decoder_output_shape = common_layers.shape_list(decoder_output)
    decoder_output = tf.reshape(decoder_output, [
        decoder_output_shape[0],
        hparams.img_len, hparams.img_len*hparams.num_channels,
        hparams.hidden_size])
    return decoder_output


def transformer_latent_decoder(encoder_output,
                               ed_attention_bias,
                               targets,
                               hparams,
                               name="transformer_latent_dec"):
  """Original Transformer decoder."""
  with tf.variable_scope(name):
    batch_size = common_layers.shape_list(targets)[0]
    compress_ratio = 2**(hparams.num_compress_steps // 2)
    # Reshape targets as b, 32, 32, 3*hidden size].
    targets = tf.reshape(targets, [
        batch_size, hparams.img_len / compress_ratio,
        (hparams.img_len*hparams.num_latents) / compress_ratio,
        hparams.hidden_size
    ])

    # Prepare decoder inputs and bias.
    decoder_input, _, _ = cia.prepare_decoder(targets, hparams)
    # hparams.num_channels = 3
    decoder_output = cia.transformer_decoder_layers(
        decoder_input,
        encoder_output,
        hparams.num_latent_layers or hparams.num_hidden_layers,
        hparams,
        attention_type=hparams.latent_attention_type,
        encoder_decoder_attention_bias=ed_attention_bias,
        name="decoder")
    decoder_output_shape = common_layers.shape_list(decoder_output)
    decoder_output = tf.reshape(decoder_output, [
        decoder_output_shape[0],
        (hparams.img_len * hparams.img_len *
         hparams.num_latents) / (2**hparams.num_compress_steps),
        hparams.hidden_size
    ])
    return decoder_output


def bottleneck_layer(targets_c,
                     hparams,
                     name="bottlneck_d"):
  """Compute latents from compressed targets."""
  # TODO(nikip): Condense hparams by removing options we don't use.
  latents_dense, latents_discrete, extra_loss, embed_func = (
      hparams.bottleneck(
          x=targets_c,
          filter_size=hparams.compress_filter_size,
          name=name,
          mode=hparams.mode))
  if DO_SUMMARIES:
    tf.summary.histogram("b0", tf.reshape(latents_discrete, [-1]))
  return latents_dense, latents_discrete, extra_loss, embed_func


def latent_prediction_model(
    inputs, ed_attention_bias,
    latents_discrete, embed,
    hparams, name="latent_pred"):
  """Transformer based latent prediction model."""
  with tf.variable_scope(name):
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      latents_pred = transformer_latent_decoder(
          inputs, ed_attention_bias,
          tf.stop_gradient(embed(latents_discrete)), hparams, name + "_extra")
      _, latent_pred_loss = ae_latent_softmax(
          latents_pred, tf.stop_gradient(latents_discrete), hparams)
  return latents_pred, latent_pred_loss


def transformer_autoencoder(inputs,
                            targets,
                            target_space,
                            hparams,
                            cache=None,
                            predict_mask=1.0):
  """AE Transformer, main step used for training."""
  # Define losses
  losses = {"extra": tf.constant(0.0), "latent_pred": tf.constant(0.0)}

  # Reshape image targets as 4d tensor.
  original_targets_shape = common_layers.shape_list(targets)
  if len(original_targets_shape) == 4:
    compress_fn = compress_encoder_2d
    decompress_fn = decompress_decoder_2d
  else:
    compress_fn = compress_encoder_1d
    decompress_fn = decompress_decoder_1d

  # Encoder decoder attention bias.
  ed_attention_bias = None

  # Input Encoder if present.
  if inputs is not None:
    inputs = common_layers.flatten4d3d(inputs)
    inputs, ed_attention_bias = transformer_text_encoder(
        inputs, target_space, hparams, "input_enc")

  # Encode targets to compute targets compressed.
  targets_c = compress_fn(targets, hparams, "compress")
  targets, _, _ = cia.maybe_reshape_4d_to_3d(targets)

  # Following code creates an exponentially decaying variable based on which
  # we rescale the los values.
  batch_size = common_layers.shape_list(targets_c)[0]
  pc = common_layers.inverse_exp_decay(hparams.startup_steps)
  pc = pc if hparams.mode == tf.estimator.ModeKeys.TRAIN else 1.0
  cond = tf.less(tf.random_uniform([batch_size]), pc)

  # TODO(lukaszkaiser): return extra losses batchwise, multiply before mean.
  # Call bottleneck layer to get the latents.
  # Returns embedded latents, discrete latents, loss and the embedding function.
  latents_dense, latents_discrete, extra_loss, embed = (
      bottleneck_layer(targets_c, hparams))
  extra_loss = tf.reduce_mean(extra_loss) * tf.to_float(cond)

  # Call the autoregressive latent prediction model.
  _, latents_pred_loss = latent_prediction_model(
      targets_c, ed_attention_bias, latents_discrete,
      embed, hparams, name="latent_pred")
  latents_pred_loss = tf.reduce_mean(latents_pred_loss) * tf.to_float(cond)

  # Assign latent loss
  losses["latent_pred"] = latents_pred_loss
  losses["extra_loss"] = extra_loss

  latents_decoder = latents_dense
  if len(original_targets_shape) == 4:
    cmp_img_len = hparams.img_len / (2**(hparams.num_compress_steps // 2))
    latents_decoder = tf.reshape(
        latents_decoder,
        [batch_size, cmp_img_len, cmp_img_len,
         hparams.num_latents*hparams.hidden_size])

  # Decompress either using 1D or 2D upconvs.
  latents_decoder = decompress_fn(latents_decoder, hparams, name="decompress")
  # if we're operating in 2d space on images, then we're assuming that the
  # last dimension will not be a multiple of channels
  latents_decoder = tf.reshape(
      latents_decoder,
      shape=[-1, hparams.img_len, hparams.img_len, hparams.hidden_size])

  if hparams.use_gold_targets:
    latents_decoder, _, _ = cia.maybe_reshape_4d_to_3d(latents_decoder)
    masking = common_layers.inverse_exp_decay(hparams.mask_startup_steps)
    if hparams.mode == tf.estimator.ModeKeys.PREDICT:
      masking = predict_mask
    mask = tf.less(masking, tf.random_uniform(
        common_layers.shape_list(targets)[:-1]))
    mask = tf.expand_dims(tf.to_float(mask), 2)
    targets = mask * targets + (1.0 - mask) * latents_decoder
  else:
    targets = latents_decoder
  # reshape back to 4d here
  targets = tf.reshape(targets, original_targets_shape)
  if hparams.decode_autoregressive:
    # Transformer decoder, that goes from inputs->targets
    res = transformer_image_decoder(inputs, ed_attention_bias,
                                    targets, hparams, "decoder")
  else:
    res = targets

  # We'll start training the extra model of latents after mask_startup_steps.
  latent_time = tf.less(hparams.mask_startup_steps,
                        tf.to_int32(tf.train.get_global_step()))
  losses["latent_pred"] *= tf.to_float(latent_time)
  return res, losses, cache
