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

"""AE Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model


import tensorflow as tf
from tensorflow.python.training import moving_averages


_DO_SUMMARIES = True


def residual_conv(x, repeat, k, hparams, name, reuse=None):
  """A stack of convolution blocks with residual connections."""
  with tf.variable_scope(name, reuse=reuse):
    dilations_and_kernels = [((1, 1), k) for _ in xrange(3)]
    for i in xrange(repeat):
      with tf.variable_scope("repeat_%d" % i):
        y = common_layers.conv_block(
            common_layers.layer_norm(x, hparams.hidden_size, name="lnorm"),
            hparams.hidden_size,
            dilations_and_kernels,
            padding="SAME",
            name="residual_conv")
        y = tf.nn.dropout(y, 1.0 - hparams.dropout)
        x += y
    return x


def attend(x, source, hparams, name):
  with tf.variable_scope(name):
    x = tf.squeeze(x, axis=2)
    if len(source.get_shape()) > 3:
      source = tf.squeeze(source, axis=2)
    source = common_attention.add_timing_signal_1d(source)
    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams), source, None,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.num_heads,
        hparams.attention_dropout)
    res = common_layers.layer_postprocess(x, y, hparams)
    return tf.expand_dims(res, axis=2)


def decompress_step(source, hparams, first_relu, is_2d, name):
  """Decompression function."""
  with tf.variable_scope(name):
    shape = common_layers.shape_list(source)
    multiplier = 4 if is_2d else 2
    kernel = (1, 1) if is_2d else (1, 1)
    thicker = common_layers.conv_block(
        source, hparams.hidden_size * multiplier, [((1, 1), kernel)],
        first_relu=first_relu, name="decompress_conv")
    if is_2d:
      return tf.depth_to_space(thicker, 2)
    return tf.reshape(thicker, [shape[0], shape[1] * 2, 1, hparams.hidden_size])


def top_k_softmax(x, k):
  """Calculate softmax(x), select top-k and rescale to sum to 1."""
  x = tf.nn.softmax(x)
  top_x, _ = tf.nn.top_k(x, k=k+1)
  min_top = tf.reduce_min(top_x, axis=-1, keep_dims=True)
  x = tf.nn.relu((x - min_top) + 1e-12)
  x /= tf.reduce_sum(x, axis=-1, keep_dims=True)
  return x, tf.reduce_max(top_x, axis=-1)


def top_k_experts(x, k, hparams):
  x_shape = common_layers.shape_list(x)
  x_flat = tf.reshape(x, [-1, common_layers.shape_list(x)[-1]])
  is_training = hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
  gates, load = expert_utils.noisy_top_k_gating(
      x_flat, hparams.v_size, is_training, k)
  gates_shape = [x_shape[0], x_shape[1], x_shape[2], hparams.v_size]
  gates = tf.reshape(gates, gates_shape)
  load_loss = expert_utils.cv_squared(load)
  return gates, load_loss


def gumbel_sample(shape):
  """Sample from the Gumbel distribution, protect from overflows."""
  uniform_samples = tf.random_uniform(shape, minval=0.00001, maxval=0.99998)
  return -tf.log(-tf.log(uniform_samples))


def dae(x, hparams, name):
  with tf.variable_scope(name):
    m = tf.layers.dense(x, hparams.v_size, name="mask")
    if hparams.softmax_k > 0:
      m, kl = top_k_softmax(m, hparams.softmax_k)
      return m, m, 1.0 - tf.reduce_mean(kl)
    logsm = tf.nn.log_softmax(m)
    # Gumbel-softmax sample.
    gumbel_samples = gumbel_sample(common_layers.shape_list(m))
    steps = hparams.kl_warmup_steps
    gumbel_samples *= common_layers.inverse_exp_decay(steps // 5) * 0.5
    temperature = 1.2 - common_layers.inverse_lin_decay(steps)
    # 10% of the time keep reasonably high temperature to keep learning.
    temperature = tf.cond(tf.less(tf.random_uniform([]), 0.9),
                          lambda: temperature,
                          lambda: tf.random_uniform([], minval=0.5, maxval=1.0))
    s = tf.nn.softmax((logsm + gumbel_samples) / temperature)
    m = tf.nn.softmax(m)
    kl = - tf.reduce_max(logsm, axis=-1)
    if _DO_SUMMARIES:
      tf.summary.histogram("max-log", tf.reshape(kl, [-1]))
    # Calculate the argmax and construct hot vectors.
    maxvec = tf.reshape(tf.argmax(m, axis=-1), [-1])
    maxvhot = tf.stop_gradient(tf.one_hot(maxvec, hparams.v_size))
    # Add losses that prevent too few being used.
    distrib = tf.reshape(logsm, [-1, hparams.v_size]) * maxvhot
    d_mean = tf.reduce_mean(distrib, axis=[0], keep_dims=True)
    d_variance = tf.reduce_mean(tf.square(distrib - d_mean), axis=[0])
    d_dev = - tf.reduce_mean(d_variance)
    ret = s
    if hparams.mode != tf.contrib.learn.ModeKeys.TRAIN:
      ret = tf.reshape(maxvhot, common_layers.shape_list(s))  # Just hot @eval.
    return m, ret, d_dev * 5.0 + tf.reduce_mean(kl) * 0.002


def vae(x, z_size, name):
  with tf.variable_scope(name):
    mu = tf.layers.dense(x, z_size, name="mu")
    log_sigma = tf.layers.dense(x, z_size, name="log_sigma")
    shape = common_layers.shape_list(x)
    epsilon = tf.random_normal([shape[0], shape[1], 1, z_size])
    z = mu + tf.exp(log_sigma / 2) * epsilon
    kl = 0.5 * tf.reduce_mean(
        tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma, axis=-1)
    free_bits = z_size // 4
    kl_loss = tf.reduce_mean(tf.maximum(kl - free_bits, 0.0))
    return z, kl_loss, mu, log_sigma


def project_hidden(x, hparams):
  """Project encoder hidden state into block_dim using projection tensors.

  Args:
    x: Encoder hidden state of shape [-1, hidden_size]
    hparams: Hparams

  Returns:
    Projected states of shape [-1, num_blocks, block_dim].
  """
  x = tf.reshape(x, shape=[1, -1, hparams.hidden_size])
  x_tiled = tf.reshape(
      tf.tile(x, multiples=[hparams.num_blocks, 1, 1]),
      shape=[hparams.num_blocks, -1, hparams.hidden_size])
  x_projected = tf.matmul(x_tiled, hparams.projection_tensors)
  x_projected = tf.transpose(x_projected, perm=[1, 0, 2])
  return x_projected


def slice_hidden(x, hparams):
  """Slice encoder hidden state into block_dim.

  Args:
    x: Encoder hidden state of shape [-1, hidden_size]
    hparams: Hparams

  Returns:
    Sliced states of shape [-1, num_blocks, block_dim].
  """
  assert hparams.num_blocks * hparams.block_dim == hparams.hidden_size
  x_sliced = tf.reshape(x, shape=[-1, hparams.num_blocks, hparams.block_dim])
  return x_sliced


def nearest(x, means, hparams):
  """Find the nearest means to elements in x."""
  x_reshaped = hparams.reshape_fn(x, hparams)
  x_norm_sq = tf.reduce_sum(tf.square(x_reshaped), axis=-1, keep_dims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keep_dims=True)
  scalar_prod = tf.matmul(
      tf.transpose(x_reshaped, perm=[1, 0, 2]),
      tf.transpose(means, perm=[0, 2, 1]))
  scalar_prod = tf.transpose(scalar_prod, perm=[1, 0, 2])
  dist = x_norm_sq + tf.transpose(
      means_norm_sq, perm=[2, 0, 1]) - 2 * scalar_prod
  if hparams.random_top_k > 1:
    _, top_k_idx = tf.nn.top_k(-dist, k=hparams.random_top_k)
    nearest_idx = tf.gather(
        top_k_idx,
        tf.random_uniform(
            [1], minval=0, maxval=hparams.random_top_k - 1, dtype=tf.int32),
        axis=-1)
  else:
    nearest_idx = tf.argmax(-dist, axis=-1)
  nearest_hot = tf.one_hot(nearest_idx, hparams.block_v_size)
  shape = common_layers.shape_list(x)
  shape[-1] = hparams.num_blocks
  shape.append(hparams.block_v_size)
  nearest_hot = tf.reshape(nearest_hot, shape=shape)
  return tf.stop_gradient(nearest_hot)


def kmeans(x, means, hparams):
  """Compute the nearest neighbors and the loss for training the embeddings."""
  x_means_hot = nearest(x, means, hparams)
  x_means_hot_flat = tf.reshape(x_means_hot,
                                [-1, hparams.num_blocks, hparams.block_v_size])
  x_means = tf.matmul(tf.transpose(x_means_hot_flat, perm=[1, 0, 2]), means)
  x_means = tf.transpose(x_means, [1, 0, 2])
  x_reshaped = hparams.reshape_fn(x, hparams)
  q_loss = tf.reduce_mean(tf.square((tf.stop_gradient(x_reshaped) - x_means)))
  e_loss = tf.reduce_mean(tf.square(x_reshaped - tf.stop_gradient(x_means)))
  return x_means_hot, x_means, q_loss, e_loss


def bit_to_int(x_bit, nbits, base=2):
  """Turn x_bit representing numbers bitwise (lower-endian) to int tensor."""
  x_l = tf.stop_gradient(tf.reshape(x_bit, [-1, nbits]))
  x_labels = []
  for i in range(nbits):
    x_labels.append(x_l[:, i] * tf.to_int32(base)**tf.to_int32(i))
  res = sum(x_labels)
  return tf.to_int32(tf.reshape(res, common_layers.shape_list(x_bit)[:-1]))


def int_to_bit(x_int, nbits, base=2):
  """Turn x_int representing numbers into a bitwise (lower-endian) tensor."""
  x_l = tf.expand_dims(x_int, axis=-1)
  x_labels = []
  for i in range(nbits):
    x_labels.append(
        tf.floormod(
            tf.floordiv(tf.to_int32(x_l),
                        tf.to_int32(base)**i), tf.to_int32(base)))
  res = tf.concat(x_labels, axis=-1)
  return tf.to_float(res)


def bottleneck(x,
               hparams,
               filter_size,
               name,
               means=None,
               ema_count=None,
               ema_means=None):
  """Bottleneck."""
  if hparams.bottleneck_kind == "vq-vae":
    assert means is not None
    if hparams.ema:
      assert ema_count is not None
      assert ema_means is not None

  def embed(x):
    """Embedding function; must be compatible with the code later."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      if hparams.bottleneck_kind == "semhash":
        c = int_to_bit(x, z_size)
        h1a = tf.layers.dense(c, filter_size, name="vch1a")
        h1b = tf.layers.dense(1.0 - c, filter_size, name="vch1b")
        h1 = h1a + h1b
      elif hparams.bottleneck_kind == "gumbel-softmax":
        hot = tf.one_hot(x, hparams.v_size)
        h1 = tf.layers.dense(hot, hparams.hidden_size, name="dae_dense")
      elif hparams.bottleneck_kind == "vq-vae":
        shape_x = common_layers.shape_list(x)
        x_flat = tf.reshape(x, [-1, 1])
        c = int_to_bit(x_flat, nbits=hparams.z_size, base=2)
        shape = common_layers.shape_list(c)
        new_shape = shape
        new_shape[-1] = hparams.num_blocks
        new_shape.append(int(hparams.z_size / hparams.num_blocks))
        c = tf.to_int32(tf.reshape(c, shape=new_shape))
        c = bit_to_int(
            c,
            nbits=int(hparams.z_size / hparams.num_blocks),
            base=2)
        c_hot = tf.one_hot(c, depth=hparams.block_v_size, axis=-1)
        c_hot_flat = tf.reshape(
            c_hot, shape=[-1, hparams.num_blocks, hparams.block_v_size])
        h1 = tf.matmul(tf.transpose(c_hot_flat, perm=[1, 0, 2]), means)
        h1 = tf.transpose(h1, perm=[1, 0, 2])
        new_shape = shape_x
        new_shape.append(hparams.hidden_size)
        h1 = tf.reshape(h1, new_shape)
      elif hparams.bottleneck_kind == "rounding":
        h1 = x

      h2 = tf.layers.dense(tf.nn.relu(h1), filter_size, name="vch2")
      return tf.layers.dense(tf.nn.relu(h2), hparams.hidden_size, name="vcfin")

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    z_size = hparams.z_size
    l = tf.constant(0.0)
    if hparams.bottleneck_kind == "dense":
      c = tf.layers.dense(x, z_size, name="vcc")
      h1 = tf.layers.dense(c, filter_size, name="vch1")
    if hparams.bottleneck_kind == "vae":
      c, l, _, _ = vae(x, z_size, "vae")
      h1 = tf.layers.dense(c, filter_size, name="vch1")
    if hparams.bottleneck_kind == "semhash":
      c = tf.layers.dense(x, z_size, name="vcc")
      y_clean = common_layers.saturating_sigmoid(c)
      if _DO_SUMMARIES:
        tf.summary.histogram("y_clean", tf.reshape(y_clean, [-1]))
      if hparams.noise_dev > 0 and hparams.mode == tf.estimator.ModeKeys.TRAIN:
        dev = hparams.noise_dev
        noise = tf.truncated_normal(common_layers.shape_list(c),
                                    mean=0.0, stddev=dev)
        y = common_layers.saturating_sigmoid(c + noise)
      else:
        y = y_clean
      d = tf.to_float(tf.less(0.5, y))
      y_discrete = tf.stop_gradient(d) + y - tf.stop_gradient(y)
      pd = common_layers.inverse_exp_decay(hparams.startup_steps * 2)
      pd *= hparams.d_mix
      pd = pd if hparams.mode == tf.estimator.ModeKeys.TRAIN else 1.0
      c = tf.where(tf.less(tf.random_uniform(
          [common_layers.shape_list(y)[0]]), pd), y_discrete, y)
      h1a = tf.layers.dense(c, filter_size, name="vch1a")
      h1b = tf.layers.dense(1.0 - c, filter_size, name="vch1b")
      h1 = h1a + h1b
      dx = tf.to_int32(tf.stop_gradient(d))
      c = bit_to_int(dx, z_size)
    if hparams.bottleneck_kind == "gumbel-softmax":
      _, hot, l = dae(x, hparams, name)
      c = tf.argmax(hot, axis=-1)
      h1 = tf.layers.dense(hot, hparams.hidden_size, name="dae_dense")
    if hparams.bottleneck_kind == "vq-vae":
      x_means_hot, x_means, q_loss, e_loss = kmeans(x, means, hparams)

      # Get the discrete latent represenation
      x_means_idx = tf.argmax(x_means_hot, axis=-1)

      # Get the binary representation
      x_means_bits = int_to_bit(
          x_means_idx,
          nbits=int(hparams.z_size / hparams.num_blocks),
          base=2)
      shape = common_layers.shape_list(x_means_bits)
      new_shape = shape[:-1]
      new_shape[-1] = hparams.z_size
      x_means_bits = tf.reshape(x_means_bits, shape=new_shape)
      c = bit_to_int(
          tf.to_int32(x_means_bits),
          nbits=hparams.z_size,
          base=2)

      # Update the ema variables
      if hparams.ema:
        tf.logging.info("Using EMA with beta = {}".format(hparams.beta))
        updated_ema_count = moving_averages.assign_moving_average(
            ema_count,
            tf.reduce_sum(
                tf.reshape(
                    x_means_hot,
                    shape=[-1, hparams.num_blocks, hparams.block_v_size]),
                axis=0),
            hparams.decay,
            zero_debias=False)

        x_means_hot_flat = tf.reshape(
            x_means_hot, shape=[-1, hparams.num_blocks, hparams.block_v_size])
        x_reshaped = hparams.reshape_fn(x, hparams)
        dw = tf.matmul(
            tf.transpose(x_means_hot_flat, perm=[1, 2, 0]),
            tf.transpose(x_reshaped, perm=[1, 0, 2]))
        updated_ema_means = moving_averages.assign_moving_average(
            ema_means, dw, hparams.decay, zero_debias=False)
        n = tf.reduce_sum(updated_ema_count, axis=-1, keep_dims=True)
        updated_ema_count = ((updated_ema_count + hparams.epsilon) /
                             (n + hparams.v_size * hparams.epsilon) * n)
        updated_ema_means /= tf.expand_dims(updated_ema_count, axis=-1)

        with tf.control_dependencies([e_loss]):
          update_means = tf.assign(means, updated_ema_means)
          with tf.control_dependencies([update_means]):
            l = hparams.beta * e_loss
      else:
        l = q_loss + hparams.beta * e_loss

      x_reshaped = hparams.reshape_fn(x, hparams)
      shape = common_layers.shape_list(x)
      x_means = tf.reshape(x_means, shape)
      x_reshaped = tf.reshape(x_reshaped, shape)
      h1 = x_reshaped + tf.stop_gradient(x_means - x_reshaped)

    if hparams.bottleneck_kind == "rounding":
      h = tf.layers.dense(x, 1, name="vcc")

      # Make h between 0 and 1
      h = tf.sigmoid(h)

      # Multiply by z_size to get it between [0, z_size]
      h *= hparams.v_size

      # Use the rounding bottleneck
      h1 = h + tf.stop_gradient(tf.round(h) - h)
      c = tf.squeeze(tf.round(h), axis=-1)
      c = tf.to_int32(c)
    h2 = tf.layers.dense(tf.nn.relu(h1), filter_size, name="vch2")
    res = tf.layers.dense(tf.nn.relu(h2), hparams.hidden_size, name="vcfin")
    return res, c, l, embed


def compress(x, c, is_2d, hparams, name):
  """Compress."""
  with tf.variable_scope(name):
    # Run compression by strided convs.
    cur = x
    k1 = (3, 3) if is_2d else (3, 1)
    k2 = (2, 2) if is_2d else (2, 1)
    cur = residual_conv(cur, hparams.num_compress_steps, k1, hparams, "rc")
    if c is not None and hparams.do_attend_compress:
      cur = attend(cur, c, hparams, "compress_attend")
    for i in xrange(hparams.num_compress_steps):
      if hparams.do_residual_compress:
        cur = residual_conv(cur, hparams.num_compress_steps, k1, hparams,
                            "rc_%d" % i)
      cur = common_layers.conv_block(
          cur, hparams.hidden_size, [((1, 1), k2)],
          strides=k2, name="compress_%d" % i)
    return cur


def encode(x, x_space, hparams, name):
  """Transformer preparations and encoder."""
  with tf.variable_scope(name):
    (encoder_input, encoder_self_attention_bias,
     ed) = transformer.transformer_prepare_encoder(x, x_space, hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.dropout)
    return transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams), ed


def decode_transformer(encoder_output,
                       encoder_decoder_attention_bias,
                       targets,
                       hparams,
                       name,
                       task=None):
  """Original Transformer decoder."""
  with tf.variable_scope(name):
    if task is None:
      task = hparams.task
    if task == "translate":
      targets = common_layers.flatten4d3d(targets)

      decoder_input, decoder_self_bias = (
          transformer.transformer_prepare_decoder(targets, hparams))

      decoder_input = tf.nn.dropout(decoder_input,
                                    1.0 - hparams.layer_prepostprocess_dropout)

      decoder_output = transformer.transformer_decoder(
          decoder_input,
          encoder_output,
          decoder_self_bias,
          encoder_decoder_attention_bias,
          hparams)
      decoder_output = tf.expand_dims(decoder_output, axis=2)
    else:
      assert task == "image"
      inputs = None
      # have to reshape targets as b, 32, 32, 3 * hidden size] beacuse otherwise
      # prepare_image will choke
      targets = tf.reshape(targets, [tf.shape(targets)[0], hparams.img_len,
                                     hparams.img_len,
                                     hparams.num_channels*hparams.hidden_size])

      # Prepare decoder inputs and bias.
      decoder_input, _, _, bias = cia.prepare_decoder(targets, hparams)
      # Add class label to decoder input.
      if not hparams.drop_inputs:
        decoder_input += tf.reshape(
            inputs,
            [common_layers.shape_list(targets)[0], 1, 1, hparams.hidden_size])
      decoder_output = cia.transformer_decoder_layers(
          decoder_input,
          None,
          bias,
          hparams.num_decoder_layers or hparams.num_hidden_layers,
          hparams,
          attention_type=hparams.dec_attention_type,
          name="decoder")
    decoder_output_shape = common_layers.shape_list(decoder_output)
    decoder_output = tf.reshape(decoder_output, [decoder_output_shape[0], -1, 1,
                                                 hparams.hidden_size])
    # Expand since t2t expects 4d tensors.
    return decoder_output


def multinomial_sample(x, vocab_size, temperature):
  """Multinomial sampling from a n-dimensional tensor."""
  if temperature > 0:
    samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]) / temperature, 1)
  else:
    samples = tf.argmax(x, axis=-1)
  reshaped_samples = tf.reshape(samples, common_layers.shape_list(x)[:-1])
  return tf.to_int32(reshaped_samples)


def ae_latent_softmax(latents_pred, latents_discrete, hparams):
  """Latent prediction and loss."""
  vocab_size = hparams.v_size
  if hparams.bottleneck_kind == "semhash":
    vocab_size = 2**hparams.z_size
  if hparams.num_decode_blocks < 2:
    latents_logits = tf.layers.dense(latents_pred, vocab_size,
                                     name="extra_logits")
    loss = None
    if latents_discrete is not None:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=latents_discrete, logits=latents_logits)
    sample = multinomial_sample(
        latents_logits, vocab_size, hparams.sampling_temp)
    return sample, loss

  # Multi-block case.
  vocab_bits = int(math.log(vocab_size, 2))
  assert vocab_size == 2**vocab_bits
  assert vocab_bits % hparams.num_decode_blocks == 0
  block_vocab_size = 2**(vocab_bits // hparams.num_decode_blocks)
  latents_logits = [
      tf.layers.dense(
          latents_pred, block_vocab_size, name="extra_logits_%d" % i)
      for i in xrange(hparams.num_decode_blocks)
  ]
  loss = None
  if latents_discrete is not None:
    losses = []
    for i in xrange(hparams.num_decode_blocks):
      d = tf.floormod(tf.floordiv(latents_discrete,
                                  block_vocab_size**i), block_vocab_size)
      losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=d, logits=latents_logits[i]))
    loss = sum(losses)
  samples = [multinomial_sample(l, block_vocab_size, hparams.sampling_temp)
             for l in latents_logits]
  sample = sum([s * block_vocab_size**i for i, s in enumerate(samples)])
  return sample, loss


def ae_latent_sample_beam(latents_dense_in, inputs, ed, embed, hparams):
  """Sample from the latent space in the autoencoder."""
  vocab_size = 2**hparams.z_size
  beam_size = 1  # TODO(lukaszkaiser): larger beam sizes seem to work bad.
  inputs = tf.tile(inputs, [beam_size, 1, 1])
  ed = tf.tile(ed, [beam_size, 1, 1, 1])

  def symbols_to_logits_fn(ids):
    """Go from ids to logits."""
    ids = tf.expand_dims(ids, axis=2)  # Ids start with added all-zeros.
    latents_discrete = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0]])

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
      latents_dense = embed(latents_discrete)
      latents_pred = decode_transformer(
          inputs, ed, latents_dense, hparams, "extra")
      logits = tf.layers.dense(latents_pred, vocab_size, name="extra_logits")
      current_output_position = common_layers.shape_list(ids)[1] - 1
      logits = logits[:, current_output_position, :, :]
    return tf.squeeze(logits, axis=[1])

  initial_ids = tf.zeros([tf.shape(latents_dense_in)[0]], dtype=tf.int32)
  length = tf.shape(latents_dense_in)[1]
  ids, _ = beam_search.beam_search(
      symbols_to_logits_fn, initial_ids, beam_size, length,
      vocab_size, alpha=0.0, eos_id=-1, stop_early=False)

  res = tf.expand_dims(ids[:, 0, :], axis=2)  # Pick first beam.
  return res[:, 1:]  # Remove the added all-zeros from ids.


def ae_latent_sample(latents_dense, inputs, ed, embed, iters, hparams):
  """Sample from the latent space in the autoencoder."""
  if hparams.num_decode_blocks < 2 and hparams.sampling_temp == 0.0:
    # TODO(lukaszkaiser): beam-search only works in non-blocked mode for now.
    tf.logging.info("Running beam-search for latents with beam size 1.")
    return ae_latent_sample_beam(latents_dense, inputs, ed, embed, hparams)
  latents_pred = decode_transformer(inputs, ed, latents_dense, hparams, "extra")
  latents_discrete, _ = ae_latent_softmax(latents_pred, None, hparams)

  def next_bit(latents_discrete, i):
    latents_discrete_prev = latents_discrete
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      latents_dense = embed(latents_discrete)
      latents_pred = decode_transformer(
          inputs, ed, latents_dense, hparams, "extra")
      latents_discrete, _ = ae_latent_softmax(latents_pred, None, hparams)
      return tf.concat([latents_discrete_prev[:, :(i+1), :],
                        latents_discrete[:, (i+1):, :]], axis=1)

  for i in xrange(iters):
    latents_discrete = next_bit(latents_discrete, i)
  return latents_discrete


def ae_transformer_internal(inputs,
                            targets,
                            target_space,
                            hparams,
                            cache=None,
                            predict_mask=1.0,
                            means=None,
                            ema_count=None,
                            ema_means=None):
  """AE Transformer, main step used for training."""
  # Summaries break with the do_refine cond, turn them off in that case.
  global _DO_SUMMARIES
  if hparams.do_refine:
    _DO_SUMMARIES = False

  # Prepare.
  if inputs is not None:
    batch_size = common_layers.shape_list(inputs)[0]
  else:
    batch_size = common_layers.shape_list(targets)[0]
  targets = tf.reshape(targets, [batch_size, -1, 1, hparams.hidden_size])

  # Encoder.
  if inputs is not None:
    inputs = common_layers.flatten4d3d(inputs)
    inputs, ed = encode(inputs, target_space, hparams, "input_enc")
  else:
    ed = None

  # Autoencoding.
  losses = {"extra": tf.constant(0.0), "latent_pred": tf.constant(0.0)}
  if hparams.do_ae:
    # flatten here
    original_targets_shape = tf.shape(targets)
    if hparams.task == "image":
      cia.maybe_reshape_4d_to_3d(targets, hparams)
    if hparams.task == "translate":
      max_targets_len_from_inputs = tf.concat([inputs, inputs], axis=1)
    else:
      assert hparams.task == "image"
      max_targets_len_from_inputs = targets
    targets, _ = common_layers.pad_to_same_length(
        targets, max_targets_len_from_inputs,
        final_length_divisible_by=2**hparams.num_compress_steps)
    targets_c = compress(targets, inputs, False, hparams, "compress")
    if hparams.mode != tf.estimator.ModeKeys.PREDICT:
      # Compress and bottleneck.
      latents_dense, latents_discrete, extra_loss, embed = bottleneck(
          targets_c, hparams, 2 * 2048, "vc", means, ema_count, ema_means)
      if _DO_SUMMARIES:
        tf.summary.histogram("b0", tf.reshape(latents_discrete[:, 0, :], [-1]))
      pc = common_layers.inverse_exp_decay(hparams.startup_steps)
      pc = pc if hparams.mode == tf.estimator.ModeKeys.TRAIN else 1.0
      cond = tf.less(tf.random_uniform([batch_size]), pc)
      latents_dense = tf.where(cond, latents_dense, targets_c)
      # TODO(lukaszkaiser): return extra losses batchwise, multiply before mean.
      losses["extra"] = extra_loss * tf.reduce_mean(tf.to_float(cond))
      # Extra loss predicting latent code from input. Discrete only.
      if hparams.bottleneck_kind not in ["dense", "vae"]:
        latents_pred = decode_transformer(
            inputs if inputs is not None else None,
            ed if inputs is not None else None,
            embed(latents_discrete), hparams, "extra",
            task="translate")
        _, latent_pred_loss = ae_latent_softmax(
            latents_pred, latents_discrete, hparams)
        losses["latent_pred"] = tf.reduce_mean(
            latent_pred_loss * 0.5 * tf.to_float(cond))
      else:
        inputs_c = decode_transformer(inputs, ed, targets_c, hparams, "dec_c")
        losses["latent_pred"] = tf.reduce_mean((inputs_c - targets_c)**2) * 20
        def bn_inputs():
          with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            bn, _, _, _ = bottleneck(inputs_c, hparams, 2 * 2048, "vc", means,
                                     ema_count, ema_means)
          return bn
        pbn = 0.8 if hparams.mode == tf.estimator.ModeKeys.TRAIN else 1.0
        inputs_c = tf.cond(tf.less(tf.random_uniform([]), pbn),
                           bn_inputs, lambda: inputs_c)
        ptc = 1.0 - common_layers.inverse_lin_decay(200000) * 0.5
        ptc = ptc if hparams.mode == tf.estimator.ModeKeys.TRAIN else 1.0
        latents_dense = tf.where(tf.less(tf.random_uniform([batch_size]), ptc),
                                 latents_dense, inputs_c)
    else:
      if hparams.bottleneck_kind in ["dense", "vae"]:
        inputs_c = decode_transformer(inputs, ed, targets_c, hparams, "dec_c")
        latents_dense, _, _, _ = bottleneck(inputs_c, hparams, 2 * 2048, "vc",
                                            means, ema_count, ema_means)
      else:
        latent_len = common_layers.shape_list(targets_c)[1]
        _, _, _, embed = bottleneck(targets_c, hparams, 2 * 2048, "vc", means,
                                    ema_count, ema_means)
        latents_dense = tf.zeros_like(targets_c[:, :latent_len, :, :])
        if cache is None:
          cache = ae_latent_sample(
              latents_dense, inputs, ed, embed, 16, hparams)
        latents_dense = embed(cache)
    # Postprocess.
    d = latents_dense
    pos = tf.get_variable("pos", [1, 1000, 1, hparams.hidden_size])
    pos = pos[:, :common_layers.shape_list(latents_dense)[1] + 1, :, :]
    latents_dense = tf.pad(latents_dense,
                           [[0, 0], [1, 0], [0, 0], [0, 0]]) + pos

    # Masking.
    if hparams.do_mask:
      masking = common_layers.inverse_lin_decay(hparams.mask_startup_steps)
      masking *= common_layers.inverse_exp_decay(
          hparams.mask_startup_steps // 4)  # Not much at start.
      if not hparams.do_refine:
        masking -= tf.random_uniform([]) * hparams.unmasked_percentage
      masking = tf.minimum(tf.maximum(masking, 0.0), 1.0)
      if hparams.mode == tf.estimator.ModeKeys.PREDICT:
        masking = predict_mask
      mask = tf.less(masking, tf.random_uniform(
          common_layers.shape_list(targets)[:-1]))
      mask = tf.expand_dims(tf.to_float(mask), 3)
      for i in xrange(hparams.num_compress_steps):
        j = hparams.num_compress_steps - i - 1
        d = residual_conv(d, 1, (3, 1), hparams, "decompress_rc_%d" % j)
        if hparams.do_attend_decompress:
          d = attend(d, inputs, hparams, "decompress_attend_%d" % j)
        d = decompress_step(d, hparams, i > 0, False, "decompress_%d" % j)
      # targets is always [batch, length, 1, depth]
      targets = mask * targets + (1.0 - mask) * d
      # reshape back to 4d here
      if hparams.task == "image":
        targets = tf.reshape(targets, original_targets_shape)
    if hparams.task == "translate":
      targets = tf.concat([tf.reverse(latents_dense, [1]), targets], axis=1)

  res = decode_transformer(inputs, ed, targets, hparams, "decoder")
  if hparams.do_ae:
    if hparams.task == "translate":
      res = res[:, common_layers.shape_list(latents_dense)[1]:, :, :]
    if hparams.do_mask and hparams.do_refine:
      def refine_res():
        # return residual_conv(res, 1, (5, 1), hparams, "refine")
        r, _ = encode(tf.squeeze(res, axis=[2]),
                      target_space, hparams, "refine_enc")
        return tf.expand_dims(r, axis=2)
      masked_batches = tf.reduce_sum(mask, axis=[1, 2, 3])
      all_masked = tf.less(masked_batches, 0.1)
      res = tf.where(all_masked, refine_res(), res)
    # We'll start training the extra model of latents after mask_startup_steps.
    latent_time = tf.less(hparams.mask_startup_steps,
                          tf.to_int32(tf.train.get_global_step()))
    losses["latent_pred"] *= tf.to_float(latent_time)
  return res, losses, cache


@registry.register_model
class TransformerAE(t2t_model.T2TModel):
  """Autoencoder-augmented Transformer."""

  def __init__(self, *args, **kwargs):
    super(TransformerAE, self).__init__(*args, **kwargs)
    self.predict_mask = 1.0

    # Define the embeddings if we are using vq-vae
    self.means = None
    self.ema_count = None
    self.ema_means = None
    if self._hparams.bottleneck_kind == "vq-vae":
      # Check that num_blocks exactly divides hidden_size and v_size
      assert self._hparams.hidden_size % self._hparams.num_blocks == 0
      assert self._hparams.v_size % self._hparams.num_blocks == 0

      self._hparams.block_dim = int(
          self._hparams.hidden_size // self._hparams.num_blocks)
      self._hparams.block_v_size = 2**(
          self._hparams.z_size / self._hparams.num_blocks)
      self._hparams.block_v_size = int(self._hparams.block_v_size)

      if self._hparams.reshape_method == "project":
        tf.logging.info("Using projections for decomposed vq-vae")
        tf.logging.info("Trainable projections = {}".format(
            self._hparams.trainable_projections))
        self._hparams.projection_tensors = tf.get_variable(
            name="projection",
            shape=[
                self._hparams.num_blocks, self._hparams.hidden_size,
                self._hparams.block_dim
            ],
            initializer=tf.random_normal_initializer(),
            trainable=self._hparams.trainable_projections)
        self._hparams.reshape_fn = project_hidden
      elif self._hparams.reshape_method == "slice":
        tf.logging.info("Using slices for decomposed vq-vae")
        self._hparams.reshape_fn = slice_hidden
      else:
        raise ValueError("Unknown reshape method")

      self.means = tf.get_variable(
          name="means",
          shape=[
              self._hparams.num_blocks, self._hparams.block_v_size,
              self._hparams.block_dim
          ],
          initializer=tf.uniform_unit_scaling_initializer())

      # Create the shadow variables if we are using EMA
      if self._hparams.ema:
        self.ema_count = tf.get_variable(
            "ema_count", [self._hparams.num_blocks, self._hparams.block_v_size],
            initializer=tf.constant_initializer(0))
        with tf.colocate_with(self.means):
          self.ema_means = tf.get_variable(
              "ema_means", initializer=self.means.initialized_value())

  @property
  def has_input(self):
    return self._problem_hparams.input_modality

  def body(self, features):
    inputs = features["inputs"] if "inputs" in features else None
    if self._hparams.drop_inputs:
      inputs = None
    reuse = "cache_raw" in features
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      res, loss, _ = ae_transformer_internal(
          inputs,
          features["targets"],
          features["target_space_id"],
          self._hparams,
          features.get("cache_raw", None),
          predict_mask=self.predict_mask,
          means=self.means,
          ema_count=self.ema_count,
          ema_means=self.ema_means)
      return res, loss

  def prepare_features_for_infer(self, features):
    if self._hparams.do_mask or not self._hparams.do_ae:
      return features
    beam_batch_size = self._decode_hparams.beam_size
    beam_batch_size *= self._decode_hparams.batch_size
    inputs = tf.zeros([beam_batch_size, 1, 1, self._hparams.hidden_size])
    inputs = inputs if "inputs" in features else None
    if self._hparams.drop_inputs or not self.has_input:
      inputs = None
    targets = tf.zeros([beam_batch_size, 1, 1, self._hparams.hidden_size])
    with tf.variable_scope("body"):
      _, _, cache = ae_transformer_internal(
          inputs, targets, features["target_space_id"], self._hparams,
          self.means, self.ema_count, self.ema_means)
    features["cache_raw"] = cache

  def infer(self, features=None, decode_length=50, beam_size=1, top_beams=1,
            alpha=0.0):
    """Produce predictions from the model."""
    if not self._hparams.do_mask:
      return super(TransformerAE, self).infer(
          features, decode_length, beam_size, top_beams, alpha)["outputs"]
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Create an initial targets tensor.
    if "partial_targets" in features:
      initial_output = tf.convert_to_tensor(features["partial_targets"])
    else:
      batch_size = common_layers.shape_list(features["inputs"])[0]
      length = common_layers.shape_list(features["inputs"])[1]
      target_length = tf.to_int32(2.0 * tf.to_float(length))
      initial_output = tf.zeros((batch_size, target_length, 1, 1),
                                dtype=tf.int64)

    features["targets"] = initial_output
    logits, _ = self(features)  # pylint: disable=not-callable
    samples = tf.argmax(logits, axis=-1)

    # More steps.
    self.predict_mask = 0.0  # Use the provided targets this time.
    how_many_more_steps = 0  # Set to 1 or more for Gibbs-like sampling.
    for _ in xrange(how_many_more_steps):
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        features["targets"] = samples
        logits, _ = self(features)  # pylint: disable=not-callable
        samples = tf.argmax(logits, axis=-1)

    self.predict_mask = 1.0
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return samples


@registry.register_hparams
def transformer_ae_small():
  """Set of hyperparameters."""
  hparams = transformer.transformer_small()
  hparams.batch_size = 2048
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 4000
  hparams.num_hidden_layers = 3
  hparams.hidden_size = 384
  hparams.filter_size = 2048
  hparams.label_smoothing = 0.0
  hparams.optimizer = "Adam"  # Can be unstable, maybe try Adam.
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.997  # Needs tuning, try 0.98 to 0.999.
  hparams.add_hparam("z_size", 14)
  hparams.add_hparam("noise_dev", 0.5)
  hparams.add_hparam("d_mix", 0.5)
  # Bottleneck kinds supported: dense, vae, semhash, gumbel-softmax, vq-vae.
  hparams.add_hparam("bottleneck_kind", "semhash")
  hparams.add_hparam("num_blocks", 1)
  hparams.add_hparam("num_decode_blocks", 1)
  # Reshape method for decomposed vq-vae: slice, project
  hparams.add_hparam("reshape_method", "slice")
  hparams.add_hparam("trainable_projections", False)
  hparams.add_hparam("unmasked_percentage", 0.1)
  hparams.add_hparam("do_ae", True)
  hparams.add_hparam("do_mask", True)
  hparams.add_hparam("do_refine", False)
  hparams.add_hparam("do_attend_compress", False)
  hparams.add_hparam("do_attend_decompress", True)
  hparams.add_hparam("do_residual_compress", False)
  hparams.add_hparam("drop_inputs", False)
  hparams.add_hparam("v_size", 1024*64)
  hparams.add_hparam("max_context_length", 64)
  hparams.add_hparam("num_compress_steps", 3)
  hparams.add_hparam("startup_steps", 10000)
  hparams.add_hparam("mask_startup_steps", 50000)
  hparams.add_hparam("kmeans_lr_factor", 0.002)
  hparams.add_hparam("z_dropout", 0.1)
  hparams.add_hparam("is_2d", 0)
  hparams.add_hparam("use_gumbel_softmax", True)
  hparams.add_hparam("softmax_k", 0)
  hparams.add_hparam("decode_autoregressive", True)
  hparams.add_hparam("do_vae", True)
  hparams.add_hparam("bit_vae", True)
  hparams.add_hparam("beta", 0.25)
  hparams.add_hparam("epsilon", 1e-5)
  hparams.add_hparam("decay", 0.999)
  hparams.add_hparam("ema", True)
  hparams.add_hparam("random_top_k", 1)
  hparams.kl_warmup_steps = 150000
  hparams.force_full_predict = True

  # task params
  hparams.add_hparam("task", "translate")  # translate or image tasks supported
  return hparams


@registry.register_hparams
def imagetransformer_ae_cifar():
  """Hyperparameters for CIFAR-10 experiments."""
  hparams = transformer_ae_small()
  hparams.filter_size = 512
  hparams.num_compress_steps = 3
  hparams.v_size = 1024 * 64
  hparams.startup_steps = 10000
  hparams.kmeans_lr_factor = 0.0
  hparams.is_2d = 0
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  hparams.hidden_size = 512
  hparams.batch_size = 1
  hparams.max_length = 256
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.initializer_gain = 0.2
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.label_smoothing = 0.0
  hparams.norm_type = "layer"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.num_heads = 8
  hparams.task = "image"
  hparams.ffn_layer = "conv_hidden_relu"
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.attention_dropout = 0.0
  hparams.relu_dropout = 0.
  hparams.pos = "timing"  # timing, none
  hparams.nbr_decoder_problems = 1
  hparams.num_output_layers = 3
  hparams.add_hparam("block_size", 1)

  # dilated attention based flags
  hparams.add_hparam("gap_sizes", [2, 4, 8, 16, 32, 64, 2, 4, 8, 16, 32, 64])
  hparams.add_hparam("dilated_attention", False)

  # image size related flags
  # assuming that the image has same height and width
  hparams.add_hparam("img_len", 32)
  hparams.add_hparam("num_channels", 3)
  # Local attention params
  hparams.add_hparam("local_and_global_att", False)
  hparams.add_hparam("block_length", 256)
  hparams.add_hparam("block_width", 128)
  hparams.num_encoder_layers = 4
  hparams.num_decoder_layers = 12
  hparams.sep_rgb_embed = False
  hparams.add_hparam("dec_attention_type", cia.AttentionType.LOCAL_1D)
  hparams.add_hparam("block_rastor_scan", False)

  # multipos attention params
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)

  hparams.add_hparam("unconditional", False)  # unconditional generation

  hparams.target_modality = "image:channel_embeddings_bottom"
  hparams.drop_inputs = True
  hparams.do_attend_compress = False
  hparams.do_attend_decompress = False
  return hparams


@registry.register_hparams
def transformer_ae_base():
  """Set of hyperparameters."""
  hparams = transformer_ae_small()
  hparams.batch_size = 2048
  hparams.hidden_size = 512
  hparams.filter_size = 4096
  hparams.num_hidden_layers = 6
  return hparams
