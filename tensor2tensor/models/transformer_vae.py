# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


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


def interleave(x, y, axis=1):
  x = tf.expand_dims(x, axis=axis+1)
  y = tf.expand_dims(y, axis=axis+1)
  return tf.concat([x, y], axis=axis+1)


def decompress_step(source, c, hparams, first_relu, is_2d, name):
  """Decompression function."""
  with tf.variable_scope(name):
    shape = tf.shape(source)
    if c is not None:
      source = attend(source, c, hparams, "decompress_attend")
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
  x_shape = tf.shape(x)
  x_flat = tf.reshape(x, [-1, x.get_shape().as_list()[-1]])
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
    gumbel_samples = gumbel_sample(tf.shape(m))
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
      ret = tf.reshape(maxvhot, tf.shape(s))  # Just hot on eval/infer.
    return m, ret, d_dev * 5.0 + tf.reduce_mean(kl) * 0.002


def vae(x, z_size, name):
  with tf.variable_scope(name):
    mu = tf.layers.dense(x, z_size, name="mu")
    log_sigma = tf.layers.dense(x, z_size, name="log_sigma")
    shape = tf.shape(x)
    epsilon = tf.random_normal([shape[0], shape[1], 1, z_size])
    z = mu + tf.exp(log_sigma / 2) * epsilon
    kl = 0.5 * tf.reduce_mean(
        tf.exp(log_sigma) + tf.square(mu) - 1. - log_sigma, axis=-1)
    return z, tf.reduce_mean(kl), mu, log_sigma


def bit_vae(x, hparams, name):
  with tf.variable_scope(name):
    bity = tf.layers.dense(x, hparams.z_size, name="bity")
    dev = common_layers.inverse_lin_decay(hparams.startup_steps) * 1.5
    noise = tf.random_normal(tf.shape(bity), mean=0.0, stddev=dev)
    y = common_layers.saturating_sigmoid(bity + noise)
    tf.summary.histogram("bit", tf.reshape(y, [-1]))
    def discrete_y():
      d = tf.to_float(tf.less(0.5, y))
      return tf.stop_gradient(d) + y - tf.stop_gradient(y)
    y = tf.cond(tf.less(tf.train.get_global_step(), hparams.startup_steps),
                lambda: y, discrete_y)
    # Flatten and predict for loss.
    y_flat = tf.reshape(y, [-1, hparams.z_size, 1, 1])
    hsize = hparams.hidden_size
    hparams.hidden_size = hsize // 2
    emb0 = tf.get_variable("emb0", [hparams.hidden_size])
    emb1 = tf.get_variable("emb1", [hparams.hidden_size])
    emb0 = tf.reshape(emb0, [1, 1, 1, hparams.hidden_size])
    emb1 = tf.reshape(emb0, [1, 1, 1, hparams.hidden_size])
    y_emb = y_flat * emb1 + (1 - y_flat) * emb0
    y_logit = decode(None, None, y_emb, None, None, hparams, "dbit")
    hparams.hidden_size = hsize
    y_pred = tf.nn.log_softmax(tf.layers.dense(y_logit, 2, name="y_pred"))
    y_flat = tf.reshape(y_flat, [-1])
    y_pred = tf.reshape(y_pred, [-1, 2])
    loss = - (y_flat * y_pred[:, 1] + (1 - y_flat) * y_pred[:, 0])
    # Get the final z and return.
    z = tf.layers.dense(y, hparams.z_size, name="after_bit")
    return z, tf.reduce_mean(loss)


def nearest(x, means, hparams):
  """Find the nearest means to elements in x."""
  x, means = tf.stop_gradient(x), tf.stop_gradient(means)
  means = tf.nn.l2_normalize(means, dim=1)
  x_flat = tf.reshape(x, [-1, hparams.hidden_size])
  # dist = tf.reduce_sum(tf.square(x_flat - tf.expand_dims(means, 0)), axis=2)
  dist = - tf.matmul(x_flat, means, transpose_b=True)
  _, nearest_idx = tf.nn.top_k(- dist, k=1)
  nearest_hot = tf.one_hot(tf.squeeze(nearest_idx, axis=1), hparams.v_size)
  nearest_hot = tf.reshape(nearest_hot, [tf.shape(x)[0], tf.shape(x)[1],
                                         tf.shape(x)[2], hparams.v_size])
  return tf.stop_gradient(nearest_hot)


def kmeans(x, means, hparams, name):
  with tf.variable_scope(name):
    x_means_hot = nearest(x, means, hparams)
    x_means = tf.gather(means, tf.argmax(x_means_hot, axis=-1))
    kl = tf.reduce_sum(tf.square(x - x_means), axis=-1)
    return x_means_hot, tf.reduce_mean(kl)  # * 10.0


def bit_to_int(x_bit, nbits):
  """Turn x_bit representing numbers bitwise (lower-endian) to int tensor."""
  x_l = tf.stop_gradient(tf.reshape(x_bit, [-1, nbits]))
  x_labels = []
  for i in range(nbits):
    x_labels.append(x_l[:, i] * 2**i)
  res = sum(x_labels)
  return tf.to_int32(tf.reshape(res, tf.shape(x_bit)[:-1]))


def int_to_bit(x_int, nbits):
  """Turn x_int representing numbers into a bitwise (lower-endian) tensor."""
  x_l = tf.expand_dims(x_int, axis=-1)
  x_labels = []
  for i in range(nbits):
    x_labels.append(tf.floormod(tf.floordiv(x_l, 2**i), 2))
  res = tf.concat(x_labels, axis=-1)
  return tf.to_float(res)


def bottleneck(x, hparams, filter_size, name):
  """Bottleneck."""
  def embed1(x):
    if hparams.bottleneck_kind == "semhash":
      c = int_to_bit(x, c_size)
      h1a = tf.layers.dense(c, filter_size, name="vch1a")
      h1b = tf.layers.dense(1.0 - c, filter_size, name="vch1b")
      return h1a + h1b
    elif hparams.bottleneck_kind == "gumbel-softmax":
      hot = tf.one_hot(x, hparams.v_size)
      with tf.variable_scope(name, reuse=True):
        return tf.layers.dense(hot, hparams.hidden_size, name="dae_dense")

  def embed(x):
    with tf.variable_scope(name, reuse=True):
      h1 = embed1(x)
      h2 = tf.layers.dense(tf.nn.relu(h1), filter_size, name="vch2")
      res = tf.layers.dense(tf.nn.relu(h2), hparams.hidden_size, name="vcfin")
    return res

  with tf.variable_scope(name):
    c_size = hparams.c_size
    l = tf.constant(0.0)
    if hparams.bottleneck_kind == "dense":
      c = tf.layers.dense(x, c_size, name="vcc")
      h1 = tf.layers.dense(c, filter_size, name="vch1")
    if hparams.bottleneck_kind == "semhash":
      c = tf.layers.dense(x, c_size, name="vcc")
      y_clean = common_layers.saturating_sigmoid(c)
      tf.summary.histogram("y_clean", tf.reshape(y_clean, [-1]))
      # l = tf.reduce_mean(y_clean * (1.0 - y_clean))
      if hparams.noise_dev > 0 and hparams.mode == tf.estimator.ModeKeys.TRAIN:
        dev = hparams.noise_dev
        noise = tf.truncated_normal(tf.shape(c), mean=0.0, stddev=dev)
        y = common_layers.saturating_sigmoid(c + noise)
      else:
        y = y_clean
      d = tf.to_float(tf.less(0.5, y))
      y_discrete = tf.stop_gradient(d) + y - tf.stop_gradient(y)
      pd = common_layers.inverse_exp_decay(hparams.startup_steps * 2)
      pd *= hparams.d_mix
      pd = pd if hparams.mode == tf.estimator.ModeKeys.TRAIN else 1.0
      c = tf.cond(tf.less(tf.random_uniform([]), pd),
                  lambda: y_discrete, lambda: y)
      h1a = tf.layers.dense(c, filter_size, name="vch1a")
      h1b = tf.layers.dense(1.0 - c, filter_size, name="vch1b")
      h1 = h1a + h1b
      dx = tf.to_int32(tf.stop_gradient(d))
      c = bit_to_int(dx, c_size)
    if hparams.bottleneck_kind == "gumbel-softmax":
      _, hot, l = dae(x, hparams, name)
      c = tf.argmax(hot, axis=-1)
      h1 = tf.layers.dense(hot, hparams.hidden_size, name="dae_dense")
    h2 = tf.layers.dense(tf.nn.relu(h1), filter_size, name="vch2")
    res = tf.layers.dense(tf.nn.relu(h2), hparams.hidden_size, name="vcfin")
    return res, c, l, embed


def compress(x, c, is_2d, hparams, name):
  """Compress."""
  with tf.variable_scope(name):
    # Run compression by strided convs.
    cur = x
    k1 = (3, 3) if is_2d else (3, 1)
    cur = residual_conv(cur, hparams.num_compress_steps, k1, hparams, "rc")
    k2 = (2, 2) if is_2d else (2, 1)
    for i in xrange(hparams.num_compress_steps):
      if c is not None:
        cur = attend(cur, c, hparams, "compress_attend_%d" % i)
      cur = common_layers.conv_block(
          cur, hparams.hidden_size, [((1, 1), k2)],
          strides=k2, name="compress_%d" % i)
    return cur


def mix(x1, x2, steps, min_prob=0.0, max_prob=1.0, mode="lin", simple=False):
  """Mix starting with x2, mixing mixing, going towards x1."""
  if mode == "lin":
    alpha_p = common_layers.inverse_lin_decay(steps)
  else:
    alpha_p = common_layers.inverse_exp_decay(steps)
  alpha_p = alpha_p * (max_prob - min_prob) + min_prob
  if simple:
    return alpha_p * x1 + (1.0 - alpha_p) * x2
  alpha = tf.random_uniform(tf.shape(x1))
  alpha = tf.to_float(tf.less(alpha, alpha_p))
  return alpha * x1 + (1.0 - alpha) * x2


def encode(x, x_space, hparams, name):
  """Transformer preparations and encoder."""
  with tf.variable_scope(name):
    (encoder_input, encoder_self_attention_bias,
     ed) = transformer.transformer_prepare_encoder(x, x_space, hparams)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.dropout)
    return transformer.transformer_encoder(
        encoder_input, encoder_self_attention_bias, hparams), ed


def decode(cond_vec, cond_add, gold, c, ed, hparams, name):
  """Transformer decoder."""
  with tf.variable_scope(name):
    drop_gold = tf.nn.dropout(gold, 1.0 - hparams.layer_prepostprocess_dropout)
    decoder_input = common_layers.shift_right(drop_gold, pad_value=cond_vec)
    if cond_add is not None:
      decoder_input += cond_add
    decoder_input = tf.squeeze(decoder_input, axis=2)
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
    bias = common_attention.attention_bias_lower_triangle(tf.shape(gold)[1])
    if c is not None and len(c.get_shape()) > 3:
      c = tf.squeeze(c, axis=2)
    return transformer.transformer_decoder(decoder_input, c, bias, ed, hparams)


def decode_transformer(encoder_output,
                       encoder_decoder_attention_bias,
                       targets,
                       hparams,
                       name):
  """Original Transformer decoder."""
  with tf.variable_scope(name):
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_bias = transformer.transformer_prepare_decoder(
        targets, hparams)

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer.transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_bias,
        encoder_decoder_attention_bias,
        hparams)

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)


def expand_batch(x, mul):
  """Expand on batch by mul times."""
  cx = tf.expand_dims(x, axis=1)
  x_shape = x.get_shape().as_list()
  batch_mul = tf.to_int32(mul)
  cx += tf.zeros([1, batch_mul, 1, 1, 1])
  mid_shape = [tf.shape(x)[2]] if len(x_shape) > 3 else []
  end_shape = [x_shape[-1]] if x_shape[-1] else [tf.shape(x)[-1]]
  res_shape = [-1, tf.shape(x)[1]] + mid_shape + end_shape
  return tf.reshape(cx, res_shape)


def ae_compress(x, is_2d, hparams, name, reuse=None):
  """Compress, then AE."""
  with tf.variable_scope(name, reuse=reuse):
    cur = compress(x, None, is_2d, hparams, "compress")
    # Convolve and ReLu to get state.
    cur = common_layers.conv_block(
        cur, hparams.hidden_size, [((1, 1), (1, 1))], name="mid_conv")
    means_size = hparams.z_size if hparams.do_vae else hparams.v_size
    means = tf.get_variable("z_to_dense", [means_size, hparams.hidden_size])
    if hparams.do_vae:
      if hparams.bit_vae:
        hot, loss = bit_vae(cur, hparams, "bvae")
      else:
        hot, loss, _, _ = vae(cur, hparams.z_size, "vae")
      return cur, hot, loss
    if hparams.use_gumbel_softmax:
      _, hot, loss = dae(cur, hparams, "dae")
      return cur, hot, loss
    # Using k-means part. L2-normalizing to use fast cosine distance.
    cur = mix(tf.nn.l2_normalize(cur, dim=3), cur,
              hparams.startup_steps // 3, mode="exp", simple=True)
    cur_n = hparams.kmeans_lr_factor * cur
    cur_n += (1.0 - hparams.kmeans_lr_factor) * tf.stop_gradient(cur)
    hot, loss = kmeans(cur_n, means, hparams, name="kmeans")
    # We need a linear layer to undo the l2-normalization.
    cur = tf.layers.dense(cur, hparams.hidden_size, name="unnormalize")
    return cur, hot, loss


def ae_embed(hot, hparams, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    means_size = hparams.z_size if hparams.do_vae else hparams.v_size
    means = tf.get_variable("z_to_dense", [means_size, hparams.hidden_size])
    hot_flat = tf.reshape(hot, [-1, means_size])
    emb = tf.matmul(hot_flat, means)
    emb = tf.reshape(emb, [tf.shape(hot)[0], tf.shape(hot)[1],
                           tf.shape(hot)[2], hparams.hidden_size])
    if hparams.use_gumbel_softmax or hparams.do_vae:
      return emb
    return tf.layers.dense(emb, hparams.hidden_size,
                           name="unnormalize", reuse=reuse)


def ae_decompress(z, ae, x, is_2d, hparams, name, reuse=None):
  """Decompress from z, leaking from ae."""
  with tf.variable_scope(name + "_decompress", reuse=reuse):
    if hparams.use_gumbel_softmax or hparams.do_vae:
      # Leak at the beginning to help train.
      z = mix(z, ae, hparams.startup_steps)
    else:
      # Gradients flow to ae while the value is z.
      z = tf.stop_gradient(z) + ae - tf.stop_gradient(ae)
    # Leak during training to keep the full dense autoencoder.
    prob_z = common_layers.inverse_exp_decay(hparams.startup_steps) * 0.8
    prob_z = prob_z if hparams.mode == tf.contrib.learn.ModeKeys.TRAIN else 1.0
    z = tf.cond(tf.less(tf.random_uniform([]), prob_z),
                lambda: z, lambda: ae)

    # Dropout for better autoencoding.
    z = tf.nn.dropout(z, keep_prob=1.0 - hparams.z_dropout)

    # Decompress.
    d = z
    k = (3, 3) if is_2d else (3, 1)
    for i in xrange(hparams.num_compress_steps):
      j = hparams.num_compress_steps - i - 1
      d = residual_conv(d, 1, k, hparams, "decompress_rc_%d" % j)
      d = decompress_step(d, None, hparams, i > 0, is_2d, "decompress_%d" % j)

    # Autoregressive part.
    if hparams.decode_autoregressive:
      k = 2**(hparams.num_compress_steps * (2 if is_2d else 1))
      x_batch = tf.reshape(x, [-1, k, 1, hparams.hidden_size])
      x_batch = tf.stop_gradient(x_batch)
      z_batch = tf.reshape(z, [-1, 1, 1, hparams.hidden_size])
      d_batch = tf.reshape(d, [-1, k, 1, hparams.hidden_size])
      dec_batch = decode(z_batch, d_batch, x_batch, None, None, hparams, "dar")
    else:  # For non-autoregressive.
      dec_batch = d
    z = tf.reshape(dec_batch, [-1, tf.shape(x)[1], tf.shape(x)[2],
                               hparams.hidden_size])
    if is_2d:
      z = tf.layers.dense(z, hparams.hidden_size * 3)
  return z


def ffn(x, hparams, name):
  with tf.variable_scope(name):
    y = transformer.transformer_ffn_layer(
        common_layers.layer_preprocess(x, hparams), hparams)
    return common_layers.layer_postprocess(x, y, hparams)


def multinomial_sample(x, vocab_size, temperature):
  """Multinomial sampling from a n-dimensional tensor."""
  samples = tf.multinomial(tf.reshape(x, [-1, vocab_size]) / temperature, 1)
  reshaped_samples = tf.reshape(samples, tf.shape(x)[:-1])
  return tf.to_int32(reshaped_samples)


def ae_latent_sample(t_c, inputs, ed, embed, iters, hparams):
  """Sample from the latent space in the autoencoder."""
  t_pred = decode_transformer(inputs, ed, t_c, hparams, "extra")
  t_pred = tf.layers.dense(t_pred, 2**16, name="extra_logits")
  t_bit = multinomial_sample(t_pred, 2**16, hparams.sampling_temp)

  def next_bit(t_bit, i):
    t_bit_prev = t_bit
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      t_c = embed(t_bit)
      t_pred = decode_transformer(inputs, ed, t_c, hparams, "extra")
      t_pred = tf.layers.dense(t_pred, 2**16, name="extra_logits")
      t_bit = multinomial_sample(t_pred, 2**16, hparams.sampling_temp)
      return tf.concat([t_bit_prev[:, :(i+1), :],
                        t_bit[:, (i+1):, :]], axis=1)

  for i in xrange(iters):
    t_bit = next_bit(t_bit, i)
  return t_bit


def ae_transformer_internal(inputs, targets, target_space, hparams,
                            beam_size, cache=None, predict_mask=1.0):
  """AE Transformer, main step used for training."""
  hparams.z_size = hparams.hidden_size
  with tf.variable_scope("ae_transformer"):
    # Prepare inputs, targets, k.
    orig_targets = targets
    batch_size = tf.shape(orig_targets)[0]
    targets = tf.reshape(targets, [batch_size, -1, 1, hparams.hidden_size])
    k = hparams.num_compress_steps

    # Encoder.
    if inputs is not None:
      inputs = common_layers.flatten4d3d(inputs)
      inputs, ed = encode(inputs, target_space, hparams, "input_enc")
    else:
      ed = None

    # Autoencoding.
    losses = {"vc": tf.constant(0.0), "sm": tf.constant(0.0)}
    if hparams.do_ae:
      targets, _ = common_layers.pad_to_same_length(
          targets, targets, final_length_divisible_by=2**k)
      targets_c = compress(targets, None, False, hparams, "compress")
      if hparams.mode != tf.estimator.ModeKeys.PREDICT:
        # Compress and bottleneck.
        t_c, t_bit, vc_loss, _ = bottleneck(targets_c, hparams, 2*2048, "vc")
        tf.summary.histogram("bit0", tf.reshape(t_bit[:, 0, :], [-1]))
        pc = common_layers.inverse_exp_decay(hparams.startup_steps) * 0.95
        pc = pc if hparams.mode == tf.estimator.ModeKeys.TRAIN else 1.0
        cond = tf.less(tf.random_uniform([]), pc)
        t_c = tf.cond(cond, lambda: t_c, lambda: targets_c)
        losses["vc"] = vc_loss * tf.to_float(cond)
        # Extra loss predicting latent code from input.
        t_pred = decode_transformer(
            inputs, ed, tf.stop_gradient(t_c), hparams, "extra")
        t_pred = tf.layers.dense(t_pred, 2**16, name="extra_logits")
        losses["sm"] = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=t_bit, logits=t_pred)
        losses["sm"] = tf.reduce_mean(losses["sm"]) * 0.5 * tf.to_float(cond)
      else:
        latent_len = tf.shape(targets_c)[1]
        _, _, _, embed = bottleneck(targets_c, hparams, 2*2048, "vc")
        t_c = tf.zeros_like(targets_c[:, :latent_len, :, :])
        if cache is None:
          cache = ae_latent_sample(t_c, inputs, ed, embed, 8, hparams)
          cache = cache[0, :, :]
          cache = tf.reshape(cache, [1, latent_len, 1])
          cache = tf.tile(cache, [beam_size, 1, 1])
        t_c = embed(cache)
      # Postprocess.
      d = t_c
      pos = tf.get_variable("pos", [1, 1000, 1, hparams.hidden_size])
      pos = pos[:, :tf.shape(t_c)[1] + 1, :, :]
      t_c = tf.pad(t_c, [[0, 0], [1, 0], [0, 0], [0, 0]]) + pos

      # Masking.
      if hparams.do_mask:
        masking = common_layers.inverse_lin_decay(100000)
        masking *= common_layers.inverse_exp_decay(25000)  # Not much at start.
        masking -= tf.random_uniform([]) * 0.3
        masking = tf.minimum(tf.maximum(masking, 0.0), 1.0)
        if hparams.mode == tf.estimator.ModeKeys.PREDICT:
          masking = predict_mask
        mask = tf.less(masking, tf.random_uniform(tf.shape(targets)[:-1]))
        mask = tf.expand_dims(tf.to_float(mask), 3)
        for i in xrange(hparams.num_compress_steps):
          j = hparams.num_compress_steps - i - 1
          d = residual_conv(d, 1, (3, 1), hparams, "decompress_rc_%d" % j)
          d = decompress_step(d, None, hparams,
                              i > 0, False, "decompress_%d" % j)
        noise = d  # tf.random_uniform(tf.shape(targets))
        targets = mask * targets + (1.0 - mask) * noise
      targets = tf.concat([tf.reverse(t_c, [1]), targets], axis=1)

    res = decode_transformer(inputs, ed, targets, hparams, "decoder")
    if hparams.do_ae:
      res = res[:, tf.shape(t_c)[1]:, :, :]
    return res, losses, cache


@registry.register_model
class TransformerAE(t2t_model.T2TModel):
  """Autoencoder-augmented Transformer."""

  def __init__(self, *args, **kwargs):
    super(TransformerAE, self).__init__(*args, **kwargs)
    self.predict_mask = 1.0

  @property
  def has_input(self):
    return self._problem_hparams.input_modality

  def model_fn_body(self, features):
    inputs = features["inputs"] if "inputs" in features else None
    if self._hparams.drop_inputs:
      inputs = None
    reuse = "cache_raw" in features
    beam_size = self._decode_hparams.beam_size
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      res, loss, _ = ae_transformer_internal(
          inputs, features["targets"], features["target_space_id"],
          self._hparams, beam_size, features.get("cache_raw", None),
          predict_mask=self.predict_mask)
      return res, loss

  def prepare_features_for_infer(self, features):
    if not self._hparams.do_ae:
      return features
    beam_size = self._decode_hparams.beam_size
    inputs = tf.zeros([beam_size, 1, 1, self._hparams.hidden_size])
    inputs = inputs if "inputs" in features else None
    if self._hparams.drop_inputs or not self.has_input:
      inputs = None
    targets = tf.zeros([beam_size, 1, 1, self._hparams.hidden_size])
    with tf.variable_scope("body"):
      _, _, cache = ae_transformer_internal(
          inputs, targets, features["target_space_id"],
          self._hparams, beam_size)
    features["cache_raw"] = cache

  def infer(self, features=None, decode_length=50, beam_size=1, top_beams=1,
            alpha=0.0):
    """Produce predictions from the model."""
    if not self._hparams.do_mask:
      return super(TransformerAE, self).infer(
          features, decode_length, beam_size, top_beams, alpha)
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
      batch_size = tf.shape(features["inputs"])[0]
      length = tf.shape(features["inputs"])[1]
      target_length = tf.to_int32(1.3 * tf.to_float(length))
      initial_output = tf.zeros((batch_size, target_length, 1, 1),
                                dtype=tf.int64)

    features["targets"] = initial_output
    sharded_logits, _ = self.model_fn(features, False, force_full_predict=True)
    sharded_samples = self._data_parallelism(tf.argmax, sharded_logits, 4)
    samples = tf.concat(sharded_samples, 0)

    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return samples


@registry.register_hparams
def transformer_ae_small():
  """Set of hyperparameters."""
  hparams = transformer.transformer_small()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 4000
  hparams.num_hidden_layers = 3
  hparams.hidden_size = 384
  hparams.filter_size = 2048
  hparams.label_smoothing = 0.0
  hparams.add_hparam("c_size", 16)
  hparams.add_hparam("noise_dev", 1.0)
  hparams.add_hparam("d_mix", 0.5)
  # Bottleneck kinds supported: dense, semhash, gumbel-softmax.
  hparams.add_hparam("bottleneck_kind", "semhash")
  hparams.add_hparam("do_ae", True)
  hparams.add_hparam("do_mask", True)
  hparams.add_hparam("drop_inputs", False)
  hparams.add_hparam("z_size", 128)
  hparams.add_hparam("v_size", 1024*64)
  hparams.add_hparam("max_context_length", 64)
  hparams.add_hparam("num_compress_steps", 3)
  hparams.add_hparam("kl_steps", 35000)
  hparams.add_hparam("startup_steps", 10000)
  hparams.add_hparam("kmeans_lr_factor", 0.002)
  hparams.add_hparam("z_dropout", 0.1)
  hparams.add_hparam("is_2d", 0)
  hparams.add_hparam("use_gumbel_softmax", True)
  hparams.add_hparam("softmax_k", 0)
  hparams.add_hparam("decode_autoregressive", True)
  hparams.add_hparam("do_vae", True)
  hparams.add_hparam("bit_vae", True)
  return hparams


@registry.register_hparams
def transformer_ae_cifar():
  """Hyperparameters for CIFAR-10 experiments."""
  hparams = transformer_ae_small()
  hparams.hidden_size = 256
  hparams.filter_size = 512
  hparams.z_size = 256  # 64
  hparams.z_size2 = 0  # 16
  hparams.batch_size = 1024 * 4
  hparams.num_compress_steps = 2
  hparams.v_size = 1024 * 16
  hparams.kl_warmup_steps = 150000
  hparams.startup_steps = 20000
  hparams.kmeans_lr_factor = 0.0
  hparams.is_2d = 1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  hparams.ffn_layer = "conv_hidden_relu_with_sepconv"
  return hparams


@registry.register_hparams
def transformer_ae_base():
  """Set of hyperparameters."""
  hparams = transformer_ae_small()
  hparams.batch_size = 1024
  hparams.hidden_size = 512
  hparams.filter_size = 4096
  hparams.num_hidden_layers = 6
  return hparams
