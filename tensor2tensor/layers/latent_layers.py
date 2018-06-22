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

import functools

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search

import tensorflow as tf

DO_SUMMARIES = True


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


def multinomial_sample(x, vocab_size, sampling_method, temperature):
  """Multinomial sampling from a n-dimensional tensor."""
  if sampling_method == "random":
    samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]) / temperature, 1)
  else:
    samples = tf.argmax(x, axis=-1)
  reshaped_samples = tf.reshape(samples, common_layers.shape_list(x)[:-1])
  return reshaped_samples


def ae_latent_softmax(latents_pred, latents_discrete_hot, hparams):
  """Latent prediction and loss."""
  vocab_size = 2**hparams.bottleneck_bits
  with tf.variable_scope("latent_logits"):
    latents_logits = tf.layers.dense(latents_pred, vocab_size,
                                     name="logits_dense")
    if hparams.logit_normalization:
      latents_logits *= tf.rsqrt(1e-8 +
                                 tf.reduce_mean(tf.square(latents_logits)))
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=latents_discrete_hot, logits=latents_logits)
    sample = multinomial_sample(latents_logits, vocab_size,
                                hparams.sampling_method,
                                hparams.sampling_temp)
    return sample, loss


def ae_latent_sample_beam(latents_dense_in, inputs, ed, embed, hparams):
  """Sample from the latent space in the autoencoder."""

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
  ids, _ = beam_search.beam_search(
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
  """Decoder that decompresses x to length height*width.

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


def transformer_text_encoder(
    x, space_id, hparams, name="transformer_text_encoder"):
  """Transformer text encoder over inputs with unmasked full attention.

  Args:
    x: Tensor of shape [batch, length, hidden_dim].
    space_id: int, id.
    hparams: Dict, hyperparameters.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, length, hidden_dim].
    ed: Tensor, bias for padded tokens in the input, shape [batch, length]
  """
  with tf.variable_scope(name):
    x = common_layers.flatten4d3d(x)
    (encoder_input, encoder_self_attention_bias,
     ed) = transformer.transformer_prepare_encoder(x, space_id, hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.dropout)
    return transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams), ed


def transformer_image_decoder(x,
                              encoder_output,
                              ed_attention_bias,
                              hparams,
                              name="transformer_dec"):
  """Transformer image decoder over inputs with local attention.

  Args:
    x: Tensor of shape [batch, height, width, hidden_dim].
    encoder_output: Tensor, encoder output of shape [batch, length, hidden_dim].
    ed_attention_bias: Tensor, bias for x.
    hparams: Dict, hyperparameters.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, height, width, hidden_dim].
  """
  with tf.variable_scope(name):
    batch_size = common_layers.shape_list(x)[0]
    # Reshape targets as b, 32, 32, 3*hidden size].
    targets = tf.reshape(x, [
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


def transformer_latent_decoder(x,
                               encoder_output,
                               ed_attention_bias,
                               hparams,
                               name="transformer_latent_dec"):
  """Transformer decoder over latents using latent_attention_type.

  Args:
    x: Tensor of shape [batch, height, width, hidden_dim].
    encoder_output: Tensor, encoder output of shape [batch, length, hidden_dim].
    ed_attention_bias: Tensor, bias for x.
    hparams: Dict, hyperparameters.
    name: string, variable scope.

  Returns:
    x: Tensor of shape [batch, height, width, hidden_dim].
  """
  with tf.variable_scope(name):
    batch_size = common_layers.shape_list(x)[0]
    compress_ratio = 2**(hparams.num_compress_steps // 2)
    # Reshape targets as b, 32, 32, 3*hidden size].
    x = tf.reshape(x, [
        batch_size, hparams.img_len / compress_ratio,
        (hparams.img_len*hparams.num_latents) / compress_ratio,
        hparams.hidden_size
    ])

    # Prepare decoder inputs and bias.
    decoder_input, _, _ = cia.prepare_decoder(x, hparams)
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


def bottleneck_layer(targets_c, hparams):
  """Compute latents from compressed targets."""
  latents_discrete_hot, extra_loss = discretization.parametrized_bottleneck(
      targets_c, hparams)
  latents_dense = discretization.parametrized_unbottleneck(
      latents_discrete_hot, hparams.hidden_size, hparams)
  latents_dense = targets_c + tf.stop_gradient(latents_dense - targets_c)
  latents_discrete = tf.argmax(latents_discrete_hot, axis=-1)

  if DO_SUMMARIES:
    tf.summary.histogram("b0", tf.reshape(latents_discrete, [-1]))
  return latents_dense, latents_discrete_hot, extra_loss


def latent_prediction_model(inputs,
                            ed_attention_bias,
                            latents_discrete,
                            latents_dense,
                            hparams,
                            name="latent_prediction"):
  """Transformer based latent prediction model."""
  with tf.variable_scope(name):
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      latents_pred = transformer_latent_decoder(
          tf.stop_gradient(latents_dense), inputs, ed_attention_bias,
          hparams, name)
      _, latent_pred_loss = ae_latent_softmax(
          latents_pred, tf.stop_gradient(latents_discrete), hparams)
  return latents_pred, latent_pred_loss


def transformer_autoencoder(inputs,
                            targets,
                            target_space,
                            hparams,
                            cache=None,
                            predict_mask=1.0):
  """Auto Encoder using transformer decoder and prior over latents."""
  # Define losses
  losses = {"extra": tf.constant(0.0), "latent_pred": tf.constant(0.0)}

  # Reshape image targets as 4d tensor.
  original_targets_shape = common_layers.shape_list(targets)
  batch_size = original_targets_shape[0]
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
  # we rescale the loss values.
  pc = common_layers.inverse_exp_decay(hparams.startup_steps)
  pc = pc if hparams.mode == tf.estimator.ModeKeys.TRAIN else 1.0
  cond = tf.less(tf.random_uniform([batch_size]), pc)

  # Call bottleneck layer, that takes encoder output and outputs the latents.
  # Returns embedded latents, discrete latent codes, loss.
  if hparams.mode != tf.estimator.ModeKeys.PREDICT:
    latents_dense, latents_discrete, extra_loss = (
        bottleneck_layer(targets_c, hparams))
    extra_loss = tf.reduce_mean(extra_loss) * tf.to_float(cond)

    # Call the autoregressive latent prediction model.
    _, latents_pred_loss = latent_prediction_model(
        inputs,
        ed_attention_bias,
        latents_discrete,
        latents_dense,
        hparams,
        name="latent_pred")
    latents_pred_loss = tf.reduce_mean(latents_pred_loss) * tf.to_float(cond)
    # Latent dropout.
    latents_shape = common_layers.shape_list(latents_dense)
    latents_dense = tf.nn.dropout(
        latents_dense, 1 - hparams.latent_dropout,
        noise_shape=[latents_shape[0], latents_shape[1], 1])
    # Assign latent loss.
    losses["latent_pred"] = latents_pred_loss
    losses["extra_loss"] = extra_loss
  else:
    latent_len = (
        hparams.img_len * hparams.img_len * hparams.num_latents) / 2**(
            hparams.num_compress_steps)
    embed = functools.partial(
        discretization.parametrized_unbottleneck, hparams=hparams)
    latents_dense = tf.zeros([batch_size, latent_len, 1, hparams.hidden_size])
    if cache is None:
      cache = ae_latent_sample_beam(latents_dense, inputs, ed_attention_bias,
                                    embed, hparams)
    latents_dense = embed(
        tf.one_hot(cache, depth=2**hparams.bottleneck_bits),
        hparams.hidden_size)

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
  output = tf.reshape(
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
    output = mask * targets + (1.0 - mask) * output

  # reshape back to 4d here
  output = tf.reshape(output, original_targets_shape)
  if hparams.decode_autoregressive:
    # Transformer decoder, that goes from inputs->targets
    decoder_output = transformer_image_decoder(
        output, inputs, ed_attention_bias, hparams, "decoder")
  else:
    decoder_output = output

  # We'll start training the extra model of latents after mask_startup_steps.
  latent_time = tf.less(hparams.mask_startup_steps,
                        tf.to_int32(tf.train.get_global_step()))
  losses["latent_pred"] *= tf.to_float(latent_time)
  return decoder_output, losses, cache
