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

"""Discretization bottlenecks used to train discrete latent variables."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial  # pylint: disable=g-importing-member

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import common_layers

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.training import moving_averages  # pylint: disable=g-direct-tensorflow-import


def project_hidden(x, projection_tensors, hidden_size, num_blocks):
  """Project encoder hidden state under num_blocks using projection tensors.

  Args:
    x: Encoder hidden state of shape [batch_size, latent_dim,  hidden_size].
    projection_tensors: Projection tensors used to project the hidden state.
    hidden_size: Dimension of the latent space.
    num_blocks: Number of blocks in DVQ.

  Returns:
    x_projected: Projected states of shape [batch_size, latent_dim, num_blocks,
      hidden_size / num_blocks].
  """
  batch_size, latent_dim, _ = common_layers.shape_list(x)
  x = tf.reshape(x, shape=[1, -1, hidden_size])
  x_tiled = tf.reshape(
      tf.tile(x, multiples=[num_blocks, 1, 1]),
      shape=[num_blocks, -1, hidden_size])
  x_projected = tf.matmul(x_tiled, projection_tensors)
  x_projected = tf.transpose(x_projected, perm=[1, 0, 2])
  x_4d = tf.reshape(x_projected, [batch_size, latent_dim, num_blocks, -1])
  return x_4d


def slice_hidden(x, hidden_size, num_blocks):
  """Slice encoder hidden state under num_blocks.

  Args:
    x: Encoder hidden state of shape [batch_size, latent_dim, hidden_size].
    hidden_size: Dimension of the latent space.
    num_blocks: Number of blocks in DVQ.

  Returns:
    Sliced states of shape [batch_size, latent_dim, num_blocks, block_dim].
  """
  batch_size, latent_dim, _ = common_layers.shape_list(x)
  block_dim = hidden_size // num_blocks
  x_sliced = tf.reshape(x,
                        shape=[batch_size, latent_dim, num_blocks, block_dim])
  return x_sliced


def nearest_neighbor(x,
                     means,
                     block_v_size,
                     random_top_k=1,
                     soft_em=False,
                     num_samples=1,
                     sum_over_latents=False,
                     summary=True):
  """Find the nearest element in means to elements in x.

  Args:
    x: Continuous encodings of shape [batch_size, latent_dim, num_blocks,
      block_dim].
    means: Embedding table of shape [num_blocks, block_v_size, block_dim].
    block_v_size: Number of table entries per block.
    random_top_k: Noisy top-k if this is bigger than 1.
    soft_em: If True then use soft EM rather than hard EM.
    num_samples: Number of samples to take in soft EM.
    sum_over_latents: Whether to sum over non-batch dimensions when calculating
      negative entropy loss. Used only when doing soft EM.
    summary: If True then record summary histogram of entropies.

  Returns:
    Tensor with nearest element in mean encoded in one-hot notation
    and distances.
  """
  batch_size, latent_dim, num_blocks, block_dim = common_layers.shape_list(x)
  x = tf.reshape(x, [batch_size * latent_dim, num_blocks, block_dim])
  x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keep_dims=True)
  scalar_prod = tf.matmul(
      tf.transpose(x, perm=[1, 0, 2]), tf.transpose(means, perm=[0, 2, 1]))
  scalar_prod = tf.transpose(scalar_prod, perm=[1, 0, 2])
  dist = x_norm_sq + tf.transpose(
      means_norm_sq, perm=[2, 0, 1]) - 2 * scalar_prod

  # computing cluster probabilities
  if soft_em:
    num_blocks = common_layers.shape_list(dist)[1]
    nearest_idx = tf.stack(
        [
            tf.multinomial(-dist[:, i, :], num_samples=num_samples)
            for i in range(num_blocks)
        ],
        axis=1)
    nearest_hot = tf.one_hot(nearest_idx, depth=block_v_size)
    neg_q_entropy = tf.reduce_sum(
        nearest_hot * tf.expand_dims(tf.nn.log_softmax(-dist), 2), axis=2)
    if sum_over_latents:
      neg_q_entropy = tf.reduce_sum(neg_q_entropy, [1, 2])
    neg_q_entropy = tf.reduce_mean(neg_q_entropy, axis=0)
    nearest_hot = tf.reduce_mean(nearest_hot, axis=-2)
    if summary:
      tf.summary.histogram("neg_q_entropy", tf.reshape(neg_q_entropy, [-1]))
  else:
    neg_q_entropy = 0.
    if random_top_k > 1:
      _, top_k_idx = tf.nn.top_k(-dist, k=random_top_k)
      nearest_idx = tf.gather(
          top_k_idx,
          tf.random_uniform(
              [1], minval=0, maxval=random_top_k - 1, dtype=tf.int32),
          axis=-1)
    else:
      nearest_idx = tf.argmax(-dist, axis=-1)
    nearest_hot = tf.one_hot(nearest_idx, block_v_size)
  return nearest_hot, neg_q_entropy


def embedding_lookup(x,
                     means,
                     num_blocks,
                     block_v_size,
                     bottleneck_kind="dvq",
                     random_top_k=1,
                     soft_em=False,
                     num_samples=1,
                     do_hard_gumbel_softmax=False,
                     temperature_warmup_steps=150000,
                     num_flows=0,
                     approximate_gs_entropy=False,
                     sum_over_latents=False):
  """Compute nearest neighbors and loss for training the embeddings via DVQ.

  Args:
    x: Continuous encodings of shape [batch_size, latent_dim, num_blocks,
      block_dim].
    means: Embedding table of shape [num_blocks, block_v_size, block_dim].
    num_blocks: Number of blocks in DVQ.
    block_v_size: Number of table entries per block.
    bottleneck_kind: Discrete bottleneck type.
    random_top_k: Noisy top-k if this is bigger than 1.
    soft_em: If True then use soft EM rather than hard EM.
    num_samples: Number of samples to use for soft EM.
    do_hard_gumbel_softmax: Whether to use hard or soft Gumbel-Softmax samples
      for gumbel-softmax-dvq bottleneck.
    temperature_warmup_steps: Number of steps it takes to decay temperature to
      0. Used only if bottleneck_kind is gumbel-softmax-dvq.
    num_flows: Number of inverse autoregressive flows for gumbel-softmax-dvq
      bottleneck.
    approximate_gs_entropy: Whether to approximate the Gumbel-Softmax density
      as a categorical distribution when calculating the sample entropy. Used
      only if bottleneck_kind is gumbel-softmax-dvq.
    sum_over_latents: Whether to sum over non-batch dimensions when calculating
      negative entropy loss. Used only if soft EM or when bottleneck_kind is
      gumbel-softmax-dvq.

  Returns:
    x_means_hot: The nearest neighbor in one hot form, with shape
      [batch_size * latent_dim, num_blocks, block_v_size].
    x_means: The nearest neighbor itself, with shape [batch_size * latent_dim,
      num_blocks, block_dim].
    q_loss: Scalar Tensor representing codebook loss.
    e_loss: Scalar Tensor representing commitment loss.
    neg_q_entropy: Scalar Tensor representing negative entropy of variational
      approximation (0 if it is deterministic).
  """
  if bottleneck_kind == "gumbel-softmax-dvq":
    x_means_hot, neg_q_entropy = gumbel_softmax_nearest_neighbor_dvq(
        x,
        means,
        block_v_size,
        hard=do_hard_gumbel_softmax,
        num_samples=num_samples,
        temperature_warmup_steps=temperature_warmup_steps,
        num_flows=num_flows,
        approximate_gs_entropy=approximate_gs_entropy,
        sum_over_latents=sum_over_latents)
  else:
    x_means_hot, neg_q_entropy = nearest_neighbor(
        x,
        means,
        block_v_size,
        random_top_k,
        soft_em=soft_em,
        num_samples=num_samples,
        sum_over_latents=sum_over_latents)
  x_means_hot_flat = tf.reshape(x_means_hot, [-1, num_blocks, block_v_size])
  x_means = tf.matmul(tf.transpose(x_means_hot_flat, perm=[1, 0, 2]), means)
  x_means = tf.transpose(x_means, [1, 0, 2])
  batch_size, latent_dim, num_blocks, block_dim = common_layers.shape_list(x)
  x = tf.reshape(x, [batch_size * latent_dim, num_blocks, block_dim])

  # Currently, we use the mean scaling for the commitment loss, as opposed to
  # summing across all non-batch dimensions.
  q_loss = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(x), x_means))
  e_loss = tf.reduce_mean(tf.squared_difference(x, tf.stop_gradient(x_means)))
  return x_means_hot, x_means, q_loss, e_loss, neg_q_entropy


def bit_to_int(x_bit, num_bits, base=2):
  """Turn x_bit representing numbers bitwise (lower-endian) to int tensor.

  Args:
    x_bit: Tensor containing numbers in a particular base to be converted to
      int.
    num_bits: Number of bits in the representation.
    base: Base of the representation.

  Returns:
    Integer representation of this number.
  """
  x_l = tf.stop_gradient(tf.to_int32(tf.reshape(x_bit, [-1, num_bits])))
  x_labels = [
      x_l[:, i] * tf.to_int32(base)**tf.to_int32(i) for i in range(num_bits)]
  res = sum(x_labels)
  return tf.to_int32(tf.reshape(res, common_layers.shape_list(x_bit)[:-1]))


def int_to_bit(x_int, num_bits, base=2):
  """Turn x_int representing numbers into a bitwise (lower-endian) tensor.

  Args:
    x_int: Tensor containing integer to be converted into base notation.
    num_bits: Number of bits in the representation.
    base: Base of the representation.

  Returns:
    Corresponding number expressed in base.
  """
  x_l = tf.to_int32(tf.expand_dims(x_int, axis=-1))
  x_labels = [tf.floormod(
      tf.floordiv(tf.to_int32(x_l), tf.to_int32(base)**i), tf.to_int32(base))
              for i in range(num_bits)]
  res = tf.concat(x_labels, axis=-1)
  return tf.to_float(res)


def int_to_bit_embed(x_int, num_bits, embedding_size, base=2):
  """Turn x_int into a bitwise (lower-endian) tensor and embed densly."""
  shape = common_layers.shape_list(x_int)
  inputs = int_to_bit(x_int, num_bits, base=base)
  inputs = tf.reshape(inputs, shape[:-1] + [shape[-1] * 8])
  inputs = 2.0 * tf.to_float(inputs) - 1.0  # Move from 0/1 to -1/1.
  return tf.layers.dense(inputs, embedding_size, name="int_to_bit_embed")


def embed(x,
          hidden_size,
          z_size,
          filter_size,
          bottleneck_kind="dvq",
          soft_em=False,
          num_blocks=2,
          num_residuals=1,
          block_v_size=None,
          means=None,
          name=None):
  """Embedding function that takes discrete latent and returns embedding.

  Args:
    x: Input to the discretization bottleneck.
    hidden_size: Dimension of the latent state.
    z_size: Number of bits, where discrete codes range from 1 to 2**z_size.
    filter_size: Dimension to project embedding by. Used only if bottleneck_kind
      is semhash.
    bottleneck_kind: Kind of discretization bottleneck to use; one of dvq,
      semhash, gumbel-softmax (Default: dvq).
    soft_em: If True then it uses a multi-sample version of EM (Default: False).
    num_blocks: Number of blocks in DVQ (Default: 2).
    num_residuals: Number of residuals (Default: 1).
    block_v_size: Number of embedding entries per block (Default: None).
    means: The embedding table for dvq (Default: None).
    name: Name for the bottleneck scope.

  Returns:
    Continuous embedding to be passed on to the decoder.

  Raises:
    ValueError: For unknown or missing arguments.
  """
  with tf.variable_scope(name, default_name="embed", reuse=tf.AUTO_REUSE):
    if bottleneck_kind == "semhash":
      c = int_to_bit(x, z_size)
      h1a = tf.layers.dense(c, filter_size, name="vch1a")
      h1b = tf.layers.dense(1.0 - c, filter_size, name="vch1b")
      h1 = h1a + h1b
    elif bottleneck_kind == "gumbel-softmax":
      hot = tf.one_hot(x, 2**z_size)
      h1 = tf.layers.dense(hot, hidden_size, name="dae_dense")
    elif bottleneck_kind in ["dvq", "gumbel-softmax-dvq"]:
      if block_v_size is None:
        raise ValueError("Bottleneck kind is dvq but block_v_size is None.")

      if soft_em:
        assert num_residuals == 1
        x_hot_flat = tf.reshape(x, shape=[-1, num_blocks, block_v_size])
        h1 = tf.matmul(tf.transpose(x_hot_flat, perm=[1, 0, 2]), means[0])
        h1 = tf.transpose(h1, perm=[1, 0, 2])
        new_shape = common_layers.shape_list(x)
        new_shape[-1] = hidden_size
        h1 = tf.reshape(h1, shape=new_shape)
      else:
        shape_x = common_layers.shape_list(x)
        x_flat = tf.reshape(x, [-1, 1])
        c = int_to_bit(x_flat, num_bits=z_size, base=2)
        shape = common_layers.shape_list(c)
        new_shape = shape
        new_shape[-1] = num_residuals
        new_shape.append(num_blocks)
        new_shape.append(int(z_size / (num_residuals * num_blocks)))
        c = tf.to_int32(tf.reshape(c, shape=new_shape))
        h1_shape = shape_x
        h1_shape.append(hidden_size)
        h1 = tf.zeros(dtype=tf.float32, shape=h1_shape)
        for i in range(num_residuals):
          c_residual = bit_to_int(
              c[:, :, i, :, :],
              num_bits=int(z_size / (num_residuals * num_blocks)),
              base=2)
          c_hot = tf.one_hot(c_residual, depth=block_v_size, axis=-1)
          c_hot_flat = tf.reshape(c_hot, shape=[-1, num_blocks, block_v_size])
          h1_residual = tf.matmul(
              tf.transpose(c_hot_flat, perm=[1, 0, 2]), means[i])
          h1_residual = tf.transpose(h1_residual, perm=[1, 0, 2])
          h1_residual = tf.reshape(h1_residual, shape=h1_shape)
          h1 += h1_residual
    elif bottleneck_kind == "rounding":
      h1 = x
    else:
      raise ValueError("Unknown bottleneck kind.")

    return h1


def vae(x, z_size, name=None):
  """Simple variational autoencoder without discretization.

  Args:
    x: Input to the discretization bottleneck.
    z_size: Number of bits, where discrete codes range from 1 to 2**z_size.
    name: Name for the bottleneck scope.

  Returns:
    Embedding function, latent, loss, mu and log_simga.
  """
  with tf.variable_scope(name, default_name="vae"):
    mu = tf.layers.dense(x, z_size, name="mu")
    log_sigma = tf.layers.dense(x, z_size, name="log_sigma")
    shape = common_layers.shape_list(x)
    epsilon = tf.random_normal([shape[0], shape[1], 1, z_size])
    z = mu + tf.exp(log_sigma / 2) * epsilon
    kl = 0.5 * tf.reduce_mean(
        tf.expm1(log_sigma) + tf.square(mu) - log_sigma, axis=-1)
    free_bits = z_size // 4
    kl_loss = tf.reduce_mean(tf.maximum(kl - free_bits, 0.0))
    return z, kl_loss, mu, log_sigma


def top_k_softmax(x, k):
  """Calculate softmax(x), select top-k and rescale to sum to 1.

  Args:
    x: Input to softmax over.
    k: Number of top-k to select.

  Returns:
    softmax(x) and maximum item.
  """
  x = tf.nn.softmax(x)
  top_x, _ = tf.nn.top_k(x, k=k + 1)
  min_top = tf.reduce_min(top_x, axis=-1, keep_dims=True)
  x = tf.nn.relu((x - min_top) + 1e-12)
  x /= tf.reduce_sum(x, axis=-1, keep_dims=True)
  return x, tf.reduce_max(top_x, axis=-1)


def gumbel_sample(shape):
  """Sample from the Gumbel distribution, protect from overflows.

  Args:
    shape: Shape of Gumbel samples.

  Returns:
    Noise drawn from Gumbel distribution.
  """
  uniform_samples = tf.random_uniform(shape, minval=0.00001, maxval=0.99998)
  return -tf.log(-tf.log(uniform_samples))


def gumbel_softmax(x,
                   z_size,
                   mode,
                   softmax_k=0,
                   temperature_warmup_steps=150000,
                   summary=True,
                   name=None):
  """Gumbel softmax discretization bottleneck.

  Args:
    x: Input to the discretization bottleneck.
    z_size: Number of bits, where discrete codes range from 1 to 2**z_size.
    mode: tf.estimator.ModeKeys.
    softmax_k: If > 0 then do top-k softmax.
    temperature_warmup_steps: Number of steps it takes to decay temperature to
      0.
    summary: Whether to write summaries.
    name: Name for the bottleneck scope.

  Returns:
    Embedding function, discrete code, and loss.
  """
  with tf.variable_scope(name, default_name="gumbel_softmax"):
    m = tf.layers.dense(x, 2**z_size, name="mask")
    if softmax_k > 0:
      m, kl = top_k_softmax(m, softmax_k)
      return m, m, 1.0 - tf.reduce_mean(kl)
    logsm = tf.nn.log_softmax(m)

    # Gumbel-softmax sample.
    gumbel_samples = gumbel_sample(common_layers.shape_list(m))
    steps = temperature_warmup_steps
    gumbel_samples *= common_layers.inverse_exp_decay(steps // 5) * 0.5
    temperature = 1.2 - common_layers.inverse_lin_decay(steps)

    # 10% of the time keep reasonably high temperature to keep learning.
    temperature = tf.cond(
        tf.less(tf.random_uniform([]), 0.9), lambda: temperature,
        lambda: tf.random_uniform([], minval=0.5, maxval=1.0))
    s = tf.nn.softmax((logsm + gumbel_samples) / temperature)
    m = tf.nn.softmax(m)
    kl = -tf.reduce_max(logsm, axis=-1)

    if summary:
      tf.summary.histogram("max-log", tf.reshape(kl, [-1]))

    # Calculate the argmax and construct hot vectors.
    maxvec = tf.reshape(tf.argmax(m, axis=-1), [-1])
    maxvhot = tf.stop_gradient(tf.one_hot(maxvec, 2**z_size))

    # Add losses that prevent too few being used.
    distrib = tf.reshape(logsm, [-1, 2**z_size]) * maxvhot
    d_mean = tf.reduce_mean(distrib, axis=[0], keep_dims=True)
    d_variance = tf.reduce_mean(
        tf.squared_difference(distrib, d_mean), axis=[0])
    d_dev = -tf.reduce_mean(d_variance)
    ret = s

    if mode != tf.estimator.ModeKeys.TRAIN:
      ret = tf.reshape(maxvhot, common_layers.shape_list(s))  # Just hot @eval.
    return m, ret, d_dev * 5.0 + tf.reduce_mean(kl) * 0.002


def discrete_bottleneck(inputs,
                        hidden_size,
                        z_size,
                        filter_size,
                        mode=None,
                        bottleneck_kind="dvq",
                        num_blocks=2,
                        num_residuals=1,
                        reshape_method="slice",
                        projection_tensors=None,
                        beta=0.25,
                        ema=True,
                        means=None,
                        ema_count=None,
                        ema_means=None,
                        epsilon=1e-5,
                        decay=0.999,
                        random_top_k=1,
                        soft_em=False,
                        num_samples=1,
                        softmax_k=0,
                        temperature_warmup_steps=150000,
                        do_hard_gumbel_softmax=False,
                        num_flows=0,
                        approximate_gs_entropy=False,
                        sum_over_latents=False,
                        discrete_mix=0.5,
                        noise_dev=1.,
                        startup_steps=50000,
                        summary=True,
                        name=None,
                        cond=True):
  """Discretization bottleneck.

  Args:
    inputs: Input to the bottleneck, a Tensor of shape [..., channels].
    hidden_size: Dimension of the dense output.
    z_size: Number of bits, where discrete codes range from 1 to 2**z_size.
    filter_size: Filter size in the embedding function.
    mode: tf.estimator.ModeKeys.
    bottleneck_kind: Kind of discretization bottleneck. One of dense, dvq
      (decomposed vector quantization), gumbel-softmax, gumbel-softmax-dvq,
      semhash, or vae.
    num_blocks: Number of blocks. Used only if bottleneck_kind is DVQ.
    num_residuals: Number of residual units used to compute nearest
      neighbors. Used only if bottleneck_kind is DVQ.
    reshape_method: Method to reshape. Used only if bottleneck_kind is DVQ.
    projection_tensors: If the reshape method is project, then these are the
      tensors used to project.
    beta: Scale factor for codebook loss and EMA. Used only if bottleneck_kind
      is DVQ.
    ema: Whether to update embeddings using exponential moving averages. Used
      only if bottleneck_kind is DVQ.
    means: The embedding table. Used only if ema is True.
    ema_count: Table of counts for each embedding corresponding to how many
      examples in a batch it was the closest to. Used only if ema is True.
    ema_means: Exponentially averaged version of the embeddings. Used only if
      ema is True.
    epsilon: Small value to avoid dividing by zero in EMA update. Used only if
      ema is True.
    decay: Decay factor for the exponential moving average. Used only if ema is
      True.
    random_top_k: Noisy top-k. Used only if bottleneck_kind is DVQ.
    soft_em: Whether to use soft EM or hard EM. Used only if bottleneck_kind is
      DVQ.
    num_samples: Number of samples for soft EM. Used only if soft_em is True.
    softmax_k: If > 0 then do top-k softmax. Used only if bottleneck_kind
      is gumbel-softmax.
    temperature_warmup_steps: Number of steps it takes to decay temperature to
      0. Used only if bottleneck_kind is gumbel-softmax or gumbel-softmax-dvq.
    do_hard_gumbel_softmax: Whether to use hard or soft Gumbel-Softmax
      samples. Used only if bottleneck_kind is gumbel-softmax-dvq.
    num_flows: Number of inverse autoregresive flows. Used only if
      bottleneck_kind is gumbel-softmax-dvq.
    approximate_gs_entropy: Whether to approximate the Gumbel-Softmax density
      as a categorical distribution when calculating the sample entropy. Used
      only if bottleneck_kind is gumbel-softmax-dvq.
    sum_over_latents: Whether to sum over all non-batch dimensions before
      taking mean of entropy loss term. Used only if bottleneck kind is DVQ
      or gumbel-softmax-dvq.
    discrete_mix: Factor for mixing discrete and non-discrete input. Used only
      if bottleneck_kind is semhash.
    noise_dev: Noise stddev. Used only if bottleneck_kind is semhash.
    startup_steps: Number of steps after which latent predictor is trained. Used
      only if bottleneck_kind is semhash.
    summary: Whether to write summaries.
    name: Name for the bottleneck scope.
    cond: A tf.bool condition on whether to update the codebook.

  Returns:
    outputs_dense: Tensor of shape [..., output_dim]. The output dimension is
      hidden_size if bottleneck_kind is gumbel-softmax, DVQ; filter_size if
      bottleneck_kind is dense, semhash, vae. If bottleneck_kind is DVQ,
      outputs_dense represents the codebook (means) indexed by outputs_discrete.
    outputs_discrete: Tensor of shape [...]. Discrete codes, each an index in
      [0, 2**z_size). It uses the hot representation if soft_em is True.
    extra_loss: Scalar Tensor. Sum of codebook and commitment losses if
      bottleneck_kind is DVQ; else zero.
    embed_fn: Function embed with arguments partially filled in.
    neg_q_entropy: Scalar Tensor representing negative entropy of variational
      approximation (0 if it is deterministic).

  Raises:
    ValueError: If projection_tensors is None for reshape_method project, or
    ema_count or ema_means is None if ema is True, or unknown args.
  """
  if bottleneck_kind in ["dvq", "gumbel-softmax-dvq"]:
    assert means is not None
    if hidden_size % num_blocks != 0:
      raise ValueError("num_blocks does not divide hidden size")

    if z_size % num_residuals != 0:
      raise ValueError("num_residuals does not divide embedding table size")
    z_size_per_residual = int(z_size / num_residuals)

    if z_size_per_residual % num_blocks != 0:
      raise ValueError("num_blocks does not divide embedding table size")
    block_v_size = 2**int(z_size_per_residual / num_blocks)

    if ema:
      if ema_count is None:
        raise ValueError("ema_count is None but ema is True")
      if ema_means is None:
        raise ValueError("ema_means is None but ema is True")
  else:
    block_v_size = None

  with tf.variable_scope(
      name, default_name="discrete_bottleneck", reuse=tf.AUTO_REUSE):
    embed_fn = partial(
        embed,
        hidden_size=hidden_size,
        z_size=z_size,
        filter_size=filter_size,
        bottleneck_kind=bottleneck_kind,
        soft_em=soft_em,
        num_blocks=num_blocks,
        num_residuals=num_residuals,
        block_v_size=block_v_size,
        means=means,
        name=name)

    if bottleneck_kind == "dense":
      # Note discrete output is continuous here.
      outputs_discrete = tf.layers.dense(inputs, z_size, name="vcc")
      outputs_dense = tf.layers.dense(
          outputs_discrete, filter_size, name="vch1")
      extra_loss = tf.constant(0.0)
      neg_q_entropy = tf.constant(0.0)
    elif bottleneck_kind in ["dvq", "gumbel-softmax-dvq"]:
      inputs_3d = inputs
      if len(inputs.shape) == 4:
        inputs_3d = tf.squeeze(inputs, axis=2)
      if reshape_method == "slice":
        x_reshaped = slice_hidden(
            inputs_3d, hidden_size=hidden_size, num_blocks=num_blocks)
      elif reshape_method == "project":
        if projection_tensors is None:
          raise ValueError(
              "Projection tensors is None for reshape_method project")
        x_reshaped = project_hidden(
            inputs_3d,
            projection_tensors=projection_tensors,
            hidden_size=hidden_size,
            num_blocks=num_blocks)
      else:
        raise ValueError("Unknown reshape_method")

      x_res = tf.reshape(x_reshaped,
                         [-1] + common_layers.shape_list(x_reshaped)[2:])
      x_means_hot = []
      x_means = 0
      extra_loss = 0
      for i in range(num_residuals):
        x_means_hot_res, x_means_res, q_loss_res, e_loss_res, neg_q_entropy = (
            embedding_lookup(
                x_reshaped,
                means=means[i],
                num_blocks=num_blocks,
                block_v_size=block_v_size,
                bottleneck_kind=bottleneck_kind,
                random_top_k=random_top_k,
                soft_em=soft_em,
                num_samples=num_samples,
                temperature_warmup_steps=temperature_warmup_steps,
                do_hard_gumbel_softmax=do_hard_gumbel_softmax,
                num_flows=num_flows,
                approximate_gs_entropy=approximate_gs_entropy,
                sum_over_latents=sum_over_latents))
        # Update the EMA variables.
        if ema:
          tf.logging.info("Using EMA with beta = {}".format(beta))
          updated_ema_count_res = moving_averages.assign_moving_average(
              ema_count[i],
              tf.where(cond,
                       tf.reduce_sum(
                           tf.reshape(x_means_hot_res,
                                      shape=[-1, num_blocks, block_v_size]),
                           axis=0), ema_count[i]),
              decay,
              zero_debias=False)

          dw = tf.matmul(
              tf.transpose(x_means_hot_res, perm=[1, 2, 0]),
              tf.transpose(x_res, perm=[1, 0, 2]))

          updated_ema_means_res = moving_averages.assign_moving_average(
              ema_means[i], tf.where(cond, dw, ema_means[i]),
              decay, zero_debias=False)
          n = tf.reduce_sum(updated_ema_count_res, axis=-1, keep_dims=True)
          updated_ema_count_res = (
              (updated_ema_count_res + epsilon) / (n + 2**z_size * epsilon) * n)
          # pylint: disable=g-no-augmented-assignment
          updated_ema_means_res = updated_ema_means_res / tf.expand_dims(
              updated_ema_count_res, axis=-1)
          # pylint: enable=g-no-augmented-assignment

          with tf.control_dependencies([e_loss_res]):
            update_means_res = tf.assign(means[i],
                                         tf.where(cond,
                                                  updated_ema_means_res,
                                                  means[i]))
            with tf.control_dependencies([update_means_res]):
              extra_loss += beta * e_loss_res
        else:
          extra_loss += q_loss_res + beta * e_loss_res

        # Update the residuals.
        x_res -= x_means_res
        x_means += x_means_res
        x_means_hot.append(x_means_hot_res)

      # Get the discrete latent representation.
      x_means_hot = tf.stack(x_means_hot, axis=1)
      x_means_idx = tf.argmax(x_means_hot, axis=-1)

      # Get the binary representation.
      x_means_bits = int_to_bit(
          x_means_idx,
          num_bits=int(z_size / (num_residuals * num_blocks)),
          base=2)
      shape = common_layers.shape_list(x_means_bits)
      new_shape = shape[:-2]
      new_shape[-1] = z_size
      x_means_bits = tf.reshape(x_means_bits, shape=new_shape)
      outputs_discrete = bit_to_int(
          tf.to_int32(x_means_bits), num_bits=z_size, base=2)

      # Adjust shape of discrete outputs.
      inputs_shape = common_layers.shape_list(inputs)
      outputs_discrete = tf.reshape(outputs_discrete, inputs_shape[:-1])

      # If we're using soft EM then set discretes to the hot representation.
      if soft_em:
        outputs_discrete = x_means_hot
        outputs_discrete = tf.reshape(outputs_discrete,
                                      inputs_shape[:-1] + [block_v_size])

      # Reshape assuming hidden_size == inputs_shape[:-1].
      x_means = tf.reshape(x_means, inputs_shape)
      outputs_dense = inputs + tf.stop_gradient(x_means - inputs)
    elif bottleneck_kind == "gumbel-softmax":
      _, outputs_hot, extra_loss = gumbel_softmax(
          inputs,
          z_size=z_size,
          mode=mode,
          softmax_k=softmax_k,
          temperature_warmup_steps=temperature_warmup_steps,
          summary=summary,
          name=name)
      outputs_discrete = tf.argmax(outputs_hot, axis=-1)
      outputs_dense = tf.layers.dense(
          outputs_hot, hidden_size, name="dae_dense")
      neg_q_entropy = tf.constant(0.0)
    elif bottleneck_kind == "semhash":
      outputs_discrete = tf.layers.dense(inputs, z_size, name="vcc")
      y_clean = common_layers.saturating_sigmoid(outputs_discrete)
      if summary:
        tf.summary.histogram("y_clean", tf.reshape(y_clean, [-1]))
      if noise_dev > 0 and mode == tf.estimator.ModeKeys.TRAIN:
        noise = tf.truncated_normal(
            common_layers.shape_list(outputs_discrete),
            mean=0.0,
            stddev=noise_dev)
        y = common_layers.saturating_sigmoid(outputs_discrete + noise)
      else:
        y = y_clean
      d = tf.to_float(tf.less(0.5, y))
      y_discrete = tf.stop_gradient(d) + y - tf.stop_gradient(y)
      pd = common_layers.inverse_exp_decay(startup_steps * 2)
      pd *= discrete_mix
      pd = pd if mode == tf.estimator.ModeKeys.TRAIN else 1.0
      c = tf.where(
          tf.less(tf.random_uniform([common_layers.shape_list(y)[0]]), pd),
          y_discrete, y)
      outputs_dense_a = tf.layers.dense(c, filter_size, name="vch1a")
      outputs_dense_b = tf.layers.dense(1.0 - c, filter_size, name="vch1b")
      outputs_dense = outputs_dense_a + outputs_dense_b
      dx = tf.to_int32(tf.stop_gradient(d))
      outputs_discrete = bit_to_int(dx, z_size)
      extra_loss = tf.constant(0.0)
      neg_q_entropy = tf.constant(0.0)
    elif bottleneck_kind == "vae":
      outputs_discrete, extra_loss, _, _ = vae(inputs, z_size, name="vae")
      outputs_dense = tf.layers.dense(
          outputs_discrete, filter_size, name="vch1")
      neg_q_entropy = tf.constant(0.0)
    else:
      raise ValueError("Unknown discretization method.")

  return outputs_dense, outputs_discrete, extra_loss, embed_fn, neg_q_entropy


def predict_bits_with_lstm(prediction_source, state_size, total_num_bits,
                           target_bits=None, extra_inputs=None,
                           bits_at_once=8, temperature=1.0, dropout=0.1):
  """Predict a sequence of bits (a latent) with LSTM, both training and infer.

  Given a tensor on which the predictions are based (prediction_source), we use
  a single-layer LSTM with state of size state_size to predict total_num_bits,
  which we predict in groups of size bits_at_once. During training, we use
  target_bits as input to the LSTM (teacher forcing) and return the target_bits
  together with the prediction loss. During inference, we sample with the given
  temperature and return the predicted sequence and loss 0.

  Args:
    prediction_source: a Tensor of shape [batch_size, ...] used to create
      the initial state and the first input to the LSTM.
    state_size: python integer, the size of the LSTM state.
    total_num_bits: python integer, how many bits in total to predict.
    target_bits: a tensor of shape [batch_size, total_num_bits] used during
      training as the target to predict; each element should be -1 or 1.
    extra_inputs: a Tensor [batch_size, total_num_bits // bits_at_once, d]
      of additional inputs, passed as additional LSTM inputs.
    bits_at_once: pytho integer, how many bits to predict at once.
    temperature: python float, temperature used for sampling during inference.
    dropout: float, the amount of dropout to aply during training (0.1 default).

  Returns:
    a pair (bits, loss) with the predicted bit sequence, which is a Tensor of
    shape [batch_size, total_num_bits] with elements either -1 or 1, and a loss
    used to train the predictions against the provided target_bits.
  """

  with tf.variable_scope("predict_bits_with_lstm"):
    # Layers and cell state creation.
    lstm_cell = tf.nn.rnn_cell.LSTMCell(state_size)
    discrete_predict = tf.layers.Dense(2**bits_at_once, name="discrete_predict")
    discrete_embed = tf.layers.Dense(state_size, name="discrete_embed")
    batch_size = common_layers.shape_list(prediction_source)[0]
    layer_pred = tf.layers.flatten(prediction_source)
    first_lstm_input = tf.layers.dense(layer_pred, state_size, name="istate")
    c_state = tf.layers.dense(layer_pred, state_size, name="cstate")
    m_state = tf.layers.dense(layer_pred, state_size, name="mstate")
    state = (c_state, m_state)

    # Prediction mode if no targets are given.
    if target_bits is None:
      outputs = []
      lstm_input = first_lstm_input
      for i in range(total_num_bits // bits_at_once):
        if extra_inputs is not None:
          lstm_input = tf.concat([lstm_input, extra_inputs[:, i, :]], axis=1)
        output, state = lstm_cell(lstm_input, state)
        discrete_logits = discrete_predict(output)
        discrete_samples = common_layers.sample_with_temperature(
            discrete_logits, temperature)
        outputs.append(tf.expand_dims(discrete_samples, axis=1))
        lstm_input = discrete_embed(tf.one_hot(discrete_samples, 256))
      outputs = tf.concat(outputs, axis=1)
      outputs = int_to_bit(outputs, bits_at_once)
      outputs = tf.reshape(outputs, [batch_size, total_num_bits])
      return 2 * outputs - 1, 0.0

    # Training mode, calculating loss.
    assert total_num_bits % bits_at_once == 0
    target_bits = tf.reshape(tf.maximum(tf.stop_gradient(target_bits), 0), [
        batch_size, total_num_bits // bits_at_once, bits_at_once])
    target_ints = bit_to_int(target_bits, bits_at_once)
    tf.summary.histogram("target_integers", tf.reshape(target_ints, [-1]))
    target_hot = tf.one_hot(target_ints, 2**bits_at_once, axis=-1)
    target_embedded = discrete_embed(target_hot)
    target_embedded = tf.nn.dropout(target_embedded, 1.0 - dropout)
    teacher_input = tf.concat(
        [tf.expand_dims(first_lstm_input, axis=1), target_embedded], axis=1)
    outputs = []
    for i in range(total_num_bits // bits_at_once):
      lstm_input = teacher_input[:, i, :]
      if extra_inputs is not None:
        lstm_input = tf.concat([lstm_input, extra_inputs[:, i, :]], axis=1)
      output, state = lstm_cell(lstm_input, state)
      outputs.append(tf.expand_dims(output, axis=1))
    outputs = tf.concat(outputs, axis=1)
    outputs = tf.nn.dropout(outputs, 1.0 - dropout)
    d_int_pred = discrete_predict(outputs)
    pred_loss = tf.losses.sparse_softmax_cross_entropy(
        logits=d_int_pred, labels=target_ints)
    pred_loss = tf.reduce_mean(pred_loss)
    return d_int_pred, pred_loss


# New API for discretization bottlenecks:
# * Each method is separate and provides 2 functions:
# * The [method]_bottleneck function returns discretized state.
# * The [method]_unbottleneck function moves from discretized state to dense.


def get_vq_codebook(codebook_size, hidden_size):
  """Get lookup table for VQ bottleneck."""
  with tf.variable_scope("vq", reuse=tf.AUTO_REUSE):
    means = tf.get_variable(
        name="means",
        shape=[codebook_size, hidden_size],
        initializer=tf.uniform_unit_scaling_initializer())

    ema_count = tf.get_variable(
        name="ema_count",
        shape=[codebook_size],
        initializer=tf.constant_initializer(0),
        trainable=False)

    with tf.colocate_with(means):
      ema_means = tf.get_variable(
          name="ema_means",
          initializer=means.initialized_value(),
          trainable=False)

  return means, ema_means, ema_count


def vq_nearest_neighbor(x, means,
                        soft_em=False, num_samples=10, temperature=None):
  """Find the nearest element in means to elements in x."""
  bottleneck_size = common_layers.shape_list(means)[0]
  x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keepdims=True)
  scalar_prod = tf.matmul(x, means, transpose_b=True)
  dist = x_norm_sq + tf.transpose(means_norm_sq) - 2 * scalar_prod
  if soft_em:
    x_means_idx = tf.multinomial(-dist, num_samples=num_samples)
    x_means_hot = tf.one_hot(
        x_means_idx, depth=common_layers.shape_list(means)[0])
    x_means_hot = tf.reduce_mean(x_means_hot, axis=1)
  else:
    if temperature is None:
      x_means_idx = tf.argmax(-dist, axis=-1)
    else:
      x_means_idx = tf.multinomial(- dist / temperature, 1)
      x_means_idx = tf.squeeze(x_means_idx, axis=-1)
    if (common_layers.should_generate_summaries() and
        not common_layers.is_xla_compiled()):
      tf.summary.histogram("means_idx", tf.reshape(x_means_idx, [-1]))
    x_means_hot = tf.one_hot(x_means_idx, bottleneck_size)
  x_means_hot_flat = tf.reshape(x_means_hot, [-1, bottleneck_size])
  x_means = tf.matmul(x_means_hot_flat, means)
  e_loss = tf.reduce_mean(tf.squared_difference(x, tf.stop_gradient(x_means)))
  return x_means_hot, e_loss, dist


def vq_discrete_bottleneck(x,
                           bottleneck_bits,
                           beta=0.25,
                           decay=0.999,
                           epsilon=1e-5,
                           soft_em=False,
                           num_samples=10):
  """Simple vector quantized discrete bottleneck."""
  bottleneck_size = 2**bottleneck_bits
  x_means_hot, e_loss, _ = vq_body(
      x,
      bottleneck_size,
      beta=beta,
      decay=decay,
      epsilon=epsilon,
      soft_em=soft_em,
      num_samples=num_samples)
  return x_means_hot, e_loss


def vq_body(x,
            codebook_size,
            beta=0.25,
            decay=0.999,
            epsilon=1e-5,
            soft_em=False,
            num_samples=10,
            temperature=None,
            do_update=True):
  """Discretize each x into one of codebook_size codes."""
  x_shape = common_layers.shape_list(x)
  hidden_size = x_shape[-1]
  means, ema_means, ema_count = get_vq_codebook(codebook_size, hidden_size)
  x = tf.reshape(x, [-1, hidden_size])
  x_means_hot, e_loss, distances = vq_nearest_neighbor(
      x, means, soft_em=soft_em, num_samples=num_samples,
      temperature=temperature)

  def loss_with_update():
    """Update the ema variables and return loss triggering the update."""
    updated_ema_count = moving_averages.assign_moving_average(
        ema_count,
        tf.reduce_sum(tf.reshape(x_means_hot, shape=[-1, codebook_size]),
                      axis=0),
        decay,
        zero_debias=False)

    dw = tf.matmul(x_means_hot, x, transpose_a=True)
    updated_ema_means = tf.identity(
        moving_averages.assign_moving_average(
            ema_means, dw, decay, zero_debias=False))
    n = tf.reduce_sum(updated_ema_count, axis=-1, keepdims=True)
    updated_ema_count = (
        (updated_ema_count + epsilon) / (n + codebook_size * epsilon) * n)
    updated_ema_means /= tf.expand_dims(updated_ema_count, axis=-1)
    with tf.control_dependencies([e_loss]):
      update_means = means.assign(updated_ema_means)
      with tf.control_dependencies([update_means]):
        return beta * e_loss

  # Loss, also do update if requested.
  if do_update:
    loss = loss_with_update()
  else:
    loss = tf.cond(do_update, loss_with_update, lambda: beta * e_loss)

  d = tf.reshape(x_means_hot, x_shape[:-1] + [codebook_size])
  return d, loss, distances


def vq_loss(x,
            targets,
            codebook_size,
            beta=0.25,
            decay=0.999,
            epsilon=1e-5,
            soft_em=False,
            num_samples=10,
            temperature=None,
            do_update=True):
  """Compute the loss of large vocab tensors using a VQAE codebook.

  Args:
    x: Tensor of inputs to be quantized to nearest code
    targets: Tensor of target indices to target codes
    codebook_size: Size of quantization codebook
    beta: scalar float for moving averages
    decay: scalar float for moving averages
    epsilon: scalar float for moving averages
    soft_em: boolean, whether to apply a soft sampling procedure
    num_samples: if soft_em, number of samples to take
    temperature: temperature if we want to sample nearest neighbors or None
    do_update: whether to update the means; True by default, can be a Tensor

  Returns:
    discrete_x: one-hot Tensor indicating which codebook element is closest to x
    x_means: Tensor, on the forward pass: closest codebook element to x, on the
      backwards pass: soft convex-combination of codebook elements by proximity
      to x
    target_means: the codebook elements corresponding to the targets
    code_loss: loss driving x closer to its nearest codebook element
    targets_loss: cross-entropy loss driving x closer to code corresponding to
      target
  """
  x_shape = common_layers.shape_list(x)
  target_shape = common_layers.shape_list(targets)
  hidden_size = x_shape[-1]
  means, _, _ = get_vq_codebook(codebook_size, hidden_size)
  x = tf.reshape(x, [-1, hidden_size])
  targets = tf.reshape(targets, [-1])
  one_hot_targets = tf.one_hot(targets, codebook_size)
  target_means = tf.matmul(one_hot_targets, means)

  discrete_x, code_loss, distances = vq_body(
      x,
      codebook_size,
      beta=beta,
      decay=decay,
      epsilon=epsilon,
      soft_em=soft_em,
      num_samples=num_samples,
      temperature=temperature,
      do_update=do_update)

  logits = -distances
  targets_loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=targets)
  targets_loss = tf.reduce_mean(targets_loss)

  x_means = tf.matmul(discrete_x, means)
  x_means = x + tf.stop_gradient(x_means - x)

  discrete_x = tf.reshape(discrete_x, x_shape[:-1] + [codebook_size])
  target_means = tf.reshape(target_means, target_shape + [hidden_size])
  return discrete_x, x_means, target_means, code_loss, targets_loss


def vq_discrete_unbottleneck(x, hidden_size):
  """Simple undiscretization from vector quantized representation."""
  x_shape = common_layers.shape_list(x)
  x = tf.to_float(x)
  bottleneck_size = common_layers.shape_list(x)[-1]
  means, _, _ = get_vq_codebook(bottleneck_size, hidden_size)
  result = tf.matmul(tf.reshape(x, [-1, x_shape[-1]]), means)
  return tf.reshape(result, x_shape[:-1] + [hidden_size])


def gumbel_softmax_nearest_neighbor_dvq(x,
                                        means,
                                        block_v_size,
                                        hard=False,
                                        temperature_init=1.2,
                                        num_samples=1,
                                        temperature_warmup_steps=150000,
                                        summary=True,
                                        num_flows=0,
                                        approximate_gs_entropy=False,
                                        sum_over_latents=False):
  """Sample from Gumbel-Softmax and compute neighbors and losses.

  Args:
    x: A `float`-like `Tensor` of shape [batch_size, latent_dim, num_blocks,
      block_dim] containing the latent vectors to be compared to the codebook.
    means: Embedding table of shape [num_blocks, block_v_size, block_dim].
    block_v_size: Number of discrete codes per block.
    hard: Determines whether we take hard or soft Gumbel-Softmax samples
      (Default: False).
    temperature_init: Initial temperature used for Gumbel-Softmax samples,
      after it which it decays to 0 (Default: 1.2).
    num_samples: Number of samples drawn for each latent (Default: 1).
    temperature_warmup_steps: Number of steps it takes to decay temperature to 0
      (Default: 150000).
    summary: When `True`, we save histogram summaries of the KL term (Default:
      True).
    num_flows: Number of inverse autoregressive flows with Gumbel-Softmax
      samples.
    approximate_gs_entropy: When `True`, we approximate Gumbel-Softmax
      density as categorical when calculating sample entropy (Default: False).
    sum_over_latents: Whether to sum over non-batch dimensions when calculating
      negative entropy loss.

  Returns:
    x_means_assignments: A `float`-like `Tensor` containing the codebook
      assignments, averaged over samples, with shape [batch_size * latent_dim,
      num_blocks, block_v_size].
    neg_q_entropy: The negative entropy of the variational distribution,
      averaged over samples.
  """
  batch_size, latent_dim, num_blocks, block_dim = common_layers.shape_list(x)

  # Combine latent_dim and batch_size for computing distances.
  x = tf.reshape(x, [-1, num_blocks, block_dim])

  # Compute distances using (x - means)**2 = x**2 + means**2 - 2*x*means.
  x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keepdims=True)
  means_norm_sq = tf.transpose(means_norm_sq, perm=[2, 0, 1])
  scalar_prod = tf.matmul(
      tf.transpose(x, perm=[1, 0, 2]), tf.transpose(means, perm=[0, 2, 1]))
  scalar_prod = tf.transpose(scalar_prod, perm=[1, 0, 2])
  dist = x_norm_sq + means_norm_sq - 2 * scalar_prod

  # IAF requires latents to have their own dimension, so reshape dist from
  # [batch_size * latent_dim, num_blocks, block_v_size] to
  # [batch_size * num_blocks, latent_dim, block_v_size].
  dist = tf.reshape(dist, [batch_size, latent_dim, num_blocks, -1])
  dist = tf.reshape(
      tf.transpose(dist, perm=[0, 2, 1, 3]), [-1, latent_dim, block_v_size])
  log_class_probs = tf.nn.log_softmax(-dist)

  sample_shape = [num_samples] + common_layers.shape_list(dist)
  gumbel_samples = gumbel_sample(sample_shape)

  # Temperature decays linearly.
  temperature = temperature_init - common_layers.inverse_lin_decay(
      temperature_warmup_steps)

  # 10% of the time keep reasonably high temperature to keep learning.
  temperature = tf.cond(
      tf.less(tf.random_uniform([]), 0.9), lambda: temperature,
      lambda: tf.random_uniform([], minval=0.5, maxval=1.0))

  gumbel_softmax_samples = tf.nn.softmax(
      (tf.expand_dims(log_class_probs, 0) + gumbel_samples) / temperature)
  q_samples = tf.clip_by_value(gumbel_softmax_samples, 1e-6, 1 - 1e-6)

  if approximate_gs_entropy:
    q_dist = tfp.distributions.Multinomial(total_count=1.0, logits=-dist)
  else:
    q_dist = tfp.distributions.RelaxedOneHotCategorical(
        temperature, logits=-dist)

  # Take mean over samples to approximate entropy.
  neg_q_entropy = tf.reduce_mean(q_dist.log_prob(q_samples), 0)
  if summary:
    tf.summary.histogram("neg_q_entropy", tf.reshape(neg_q_entropy, [-1]))
  if sum_over_latents:
    neg_q_entropy = tf.reshape(neg_q_entropy,
                               [batch_size, num_blocks, latent_dim])
    neg_q_entropy = tf.reduce_sum(neg_q_entropy, [1, 2])
  neg_q_entropy = tf.reduce_mean(neg_q_entropy)

  if num_flows > 0:
    hparams = iaf_hparams(hidden_size=512, filter_size=4096)
    q_samples = tf.reshape(q_samples, [-1, latent_dim, block_v_size])
    for flow in range(num_flows):
      shifted_samples = tf.pad(q_samples, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

      # Project samples from  [batch_size, latent_size, block_v_size] to
      # [batch_size, latent_size, hidden_size].
      shifted_samples = common_layers.dense(shifted_samples,
                                            hparams.hidden_size)
      # TODO(vafa): Include masking as a flag.
      mask = True
      if mask:
        attention_type = cia.AttentionType.LOCAL_1D
      else:
        attention_type = cia.AttentionType.GLOBAL
      ffn_output = cia.transformer_decoder_layers(
          inputs=shifted_samples,
          encoder_output=None,
          num_layers=6,
          hparams=hparams,
          attention_type=attention_type,
          name="transformer_" + str(flow))

      # Project samples back to [batch_size, latent_size, block_v_size].
      ffn_output = common_layers.dense(ffn_output, block_v_size)
      log_pi = tf.nn.log_softmax(ffn_output)

      # Flow 1: Adding log_pi to q_samples and dividing by the temperature.
      # Note that we drop the last dimension of q_samples for centered-softmax,
      # which we can do without recalculating probabilities because the last
      # dimension of log_pi and q_samples are deterministic given the others.
      # Flow 2: Centered-softmax.
      chained_bijectors = tfp.bijectors.Chain([
          tfp.bijectors.SoftmaxCentered(),
          tfp.bijectors.Affine(
              shift=log_pi[:, :, :-1],
              scale_identity_multiplier=1. / temperature)
      ])
      q_samples = chained_bijectors.forward(q_samples[:, :, :-1])
      log_det = chained_bijectors.inverse_log_det_jacobian(
          q_samples, event_ndims=1)
      log_det = tf.reshape(log_det,
                           [num_samples, batch_size, num_blocks, latent_dim])
      if sum_over_latents:
        log_det = tf.reduce_sum(log_det, axis=[2, 3])
      neg_q_entropy += tf.reduce_mean(log_det)

    q_samples = tf.reshape(
        q_samples,
        [num_samples, batch_size * num_blocks, latent_dim, block_v_size])

  if hard:
    x_means_idx = tf.argmax(q_samples, -1)

    # Take average of one-hot vectors over samples.
    x_means_hot = tf.reduce_mean(tf.one_hot(x_means_idx, block_v_size), 0)
    x_means_assignments = (
        tf.reduce_mean(q_samples, 0) +
        tf.stop_gradient(x_means_hot - tf.reduce_mean(q_samples, 0)))
  else:
    x_means_assignments = tf.reduce_mean(gumbel_softmax_samples, 0)

  # Reshape assignments to [batch_size * latent_dim, num_blocks,
  # block_v_size]. We have to transpose between reshapes to make sure the
  # dimensions have the correct interpretation.
  x_means_assignments = tf.reshape(
      x_means_assignments, [batch_size, num_blocks, latent_dim, block_v_size])
  x_means_assignments = tf.transpose(x_means_assignments, [0, 2, 1, 3])
  x_means_assignments = tf.reshape(
      x_means_assignments, [batch_size * latent_dim, num_blocks, block_v_size])

  return x_means_assignments, neg_q_entropy


def gumbel_softmax_discrete_bottleneck(x,
                                       bottleneck_bits,
                                       beta=0.25,
                                       decay=0.999,
                                       epsilon=1e-5,
                                       temperature_warmup_steps=150000,
                                       hard=False,
                                       summary=True):
  """VQ-VAE using Gumbel-Softmax.

  Different from `gumbel_softmax()` function as
  this function calculates the KL by using the discrete entropy
  instead of taking the argmax, and it also uses an exponential moving average
  to update the codebook while the `gumbel_softmax()` function includes no
  codebook update.

  Args:
    x: A `float`-like `Tensor` containing the latent vectors to be compared to
      the codebook, whose squared difference is used as the Gumbel-Softmax
      logits.
    bottleneck_bits: An `int` that sets the size of the bottleneck in `log_2`.
    beta: Beta factor for commitment loss (Default: 0.25).
    decay: Decay factor for exponential moving average (Default: 0.999).
    epsilon: Small value to avoid dividing by zero in EMA update
      (Default: 1e-5).
    temperature_warmup_steps: Number of steps it takes to decay temperature to 0
      (Default: 150000).
    hard: When `True`, we use hard Gumbel-Softmax samples and force
      discrete latents by taking the argmax. When `False`, we use soft samples,
      which we treat as codebook weights (Default: False).
    summary: When `True`, we save histogram summaries of the KL term (Default:
      True).

  Returns:
    x_means_assignments: A `float`-like `Tensor` containing the codebook
      assignments. When `hard == True`, this is one-hot, containing the arg-max
      of the Gumbel-Softmax samples (and we use the straightthrough gradient).
      Otherwise, it contains the Gumbel-Softmax samples exactly, which are
      values from the `(K-1)`-simplex where `K` is the bottleneck size.
    loss: The loss, which is the sum of the KL between the Gumbel-Softmax and
      the uniform prior and the commitment loss multiplied by the beta factor.
      We approximate the KL by using the entropy of a categorical distribution
      instead of the Gumbel Softmax.

  """
  bottleneck_size = 2**bottleneck_bits
  x_shape = common_layers.shape_list(x)
  hidden_size = x_shape[-1]
  means, ema_means, ema_count = get_vq_codebook(bottleneck_size, hidden_size)
  x = tf.reshape(x, [-1, hidden_size])

  bottleneck_size = common_layers.shape_list(means)[0]
  x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keepdims=True)
  scalar_prod = tf.matmul(x, means, transpose_b=True)
  dist = x_norm_sq + tf.transpose(means_norm_sq) - 2 * scalar_prod

  class_probs = tf.nn.softmax(dist)
  log_class_probs = tf.nn.log_softmax(dist)
  gumbel_samples = gumbel_sample(common_layers.shape_list(dist))
  steps = temperature_warmup_steps
  gumbel_samples *= common_layers.inverse_exp_decay(steps // 5) * 0.5
  temperature = 1.2 - common_layers.inverse_lin_decay(steps)

  # 10% of the time keep reasonably high temperature to keep learning.
  temperature = tf.cond(
      tf.less(tf.random_uniform([]), 0.9), lambda: temperature,
      lambda: tf.random_uniform([], minval=0.5, maxval=1.0))
  gumbel_softmax_samples = tf.nn.softmax(
      (log_class_probs + gumbel_samples) / temperature)

  # Calculate KL between q and a uniform prior.
  kl = tf.reduce_sum(
      class_probs * (log_class_probs - tf.log(1.0 / bottleneck_size)), -1)
  if summary:
    tf.summary.histogram("KL", tf.reshape(kl, [-1]))

  # Straight-through gradient estimation when we're using hard assignments.
  if hard:
    x_means_idx = tf.reshape(tf.argmax(gumbel_softmax_samples, axis=-1), [-1])
    x_means_hot = tf.one_hot(x_means_idx, bottleneck_size)
    x_means_assignments = gumbel_softmax_samples + tf.stop_gradient(
        x_means_hot - gumbel_softmax_samples)
  else:
    x_means_assignments = gumbel_softmax_samples
  x_means_assignments_flat = tf.reshape(x_means_assignments,
                                        [-1, bottleneck_size])
  x_means = tf.matmul(x_means_assignments_flat, means)
  commitment_loss = tf.reduce_mean(
      tf.squared_difference(x, tf.stop_gradient(x_means)))

  # Update the ema variables.
  updated_ema_count = moving_averages.assign_moving_average(
      ema_count,
      tf.reduce_sum(
          tf.reshape(x_means_assignments, shape=[-1, bottleneck_size]), axis=0),
      decay,
      zero_debias=False)

  dw = tf.matmul(x_means_assignments, x, transpose_a=True)
  updated_ema_means = tf.identity(
      moving_averages.assign_moving_average(
          ema_means, dw, decay, zero_debias=False))
  n = tf.reduce_sum(updated_ema_count, axis=-1, keepdims=True)
  updated_ema_count = (
      (updated_ema_count + epsilon) / (n + bottleneck_size * epsilon) * n)
  updated_ema_means /= tf.expand_dims(updated_ema_count, axis=-1)
  with tf.control_dependencies([commitment_loss]):
    update_means = means.assign(updated_ema_means)
    with tf.control_dependencies([update_means]):
      loss = beta * commitment_loss

  # Add KL loss.
  loss += tf.reduce_mean(kl)

  x_means_assignments = tf.reshape(x_means_assignments,
                                   x_shape[:-1] + [bottleneck_size])
  return x_means_assignments, loss


def tanh_discrete_bottleneck(x, bottleneck_bits, bottleneck_noise,
                             discretize_warmup_steps, mode):
  """Simple discretization through tanh, flip bottleneck_noise many bits."""
  x = tf.layers.dense(x, bottleneck_bits, name="tanh_discrete_bottleneck")
  d0 = tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x))) - 1.0
  if mode == tf.estimator.ModeKeys.TRAIN:
    x += tf.truncated_normal(
        common_layers.shape_list(x), mean=0.0, stddev=0.2)
  x = tf.tanh(x)
  d = x + tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
  if mode == tf.estimator.ModeKeys.TRAIN:
    noise = tf.random_uniform(common_layers.shape_list(x))
    noise = 2.0 * tf.to_float(tf.less(bottleneck_noise, noise)) - 1.0
    d *= noise
  d = common_layers.mix(d, x, discretize_warmup_steps,
                        mode == tf.estimator.ModeKeys.TRAIN)
  return d, d0


def tanh_discrete_unbottleneck(x, hidden_size):
  """Simple un-discretization from tanh."""
  x = tf.layers.dense(x, hidden_size, name="tanh_discrete_unbottleneck")
  return x


def isemhash_bottleneck(x,
                        bottleneck_bits,
                        bottleneck_noise,
                        discretize_warmup_steps,
                        mode,
                        isemhash_noise_dev=0.5,
                        isemhash_mix_prob=0.5):
  """Improved semantic hashing bottleneck."""
  with tf.variable_scope("isemhash_bottleneck"):
    x = tf.layers.dense(x, bottleneck_bits, name="dense")
    y = common_layers.saturating_sigmoid(x)
    if isemhash_noise_dev > 0 and mode == tf.estimator.ModeKeys.TRAIN:
      noise = tf.truncated_normal(
          common_layers.shape_list(x), mean=0.0, stddev=isemhash_noise_dev)
      y = common_layers.saturating_sigmoid(x + noise)
    d = tf.to_float(tf.less(0.5, y)) + y - tf.stop_gradient(y)
    d = 2.0 * d - 1.0  # Move from [0, 1] to [-1, 1].
    if mode == tf.estimator.ModeKeys.TRAIN:  # Flip some bits.
      noise = tf.random_uniform(common_layers.shape_list(x))
      noise = 2.0 * tf.to_float(tf.less(bottleneck_noise, noise)) - 1.0
      d *= noise
      d = common_layers.mix(
          d,
          2.0 * y - 1.0,
          discretize_warmup_steps,
          mode == tf.estimator.ModeKeys.TRAIN,
          max_prob=isemhash_mix_prob)
    return d, 0.0


def isemhash_unbottleneck(x, hidden_size, isemhash_filter_size_multiplier=1.0):
  """Improved semantic hashing un-bottleneck."""
  filter_size = int(hidden_size * isemhash_filter_size_multiplier)
  x = 0.5 * (x - 1.0)  # Move from [-1, 1] to [0, 1].
  with tf.variable_scope("isemhash_unbottleneck"):
    h1a = tf.layers.dense(x, filter_size, name="hidden1a")
    h1b = tf.layers.dense(1.0 - x, filter_size, name="hidden1b")
    h2 = tf.layers.dense(tf.nn.relu(h1a + h1b), filter_size, name="hidden2")
    return tf.layers.dense(tf.nn.relu(h2), hidden_size, name="final")


def parametrized_bottleneck(x, hparams):
  """Meta-function calling all the above bottlenecks with hparams."""
  if hparams.bottleneck_kind == "tanh_discrete":
    d, _ = tanh_discrete_bottleneck(
        x, hparams.bottleneck_bits, hparams.bottleneck_noise * 0.5,
        hparams.discretize_warmup_steps, hparams.mode)
    return d, 0.0
  if hparams.bottleneck_kind == "isemhash":
    return isemhash_bottleneck(
        x, hparams.bottleneck_bits, hparams.bottleneck_noise * 0.5,
        hparams.discretize_warmup_steps, hparams.mode,
        hparams.isemhash_noise_dev, hparams.isemhash_mix_prob)
  if hparams.bottleneck_kind == "vq":
    return vq_discrete_bottleneck(x, hparams.bottleneck_bits, hparams.vq_beta,
                                  hparams.vq_decay, hparams.vq_epsilon)
  if hparams.bottleneck_kind == "em":
    return vq_discrete_bottleneck(
        x,
        hparams.bottleneck_bits,
        hparams.vq_beta,
        hparams.vq_decay,
        hparams.vq_epsilon,
        soft_em=True,
        num_samples=hparams.vq_num_samples)
  if hparams.bottleneck_kind == "gumbel_softmax":
    return gumbel_softmax_discrete_bottleneck(
        x,
        hparams.bottleneck_bits,
        hparams.vq_beta,
        hparams.vq_decay,
        hparams.vq_epsilon,
        hparams.temperature_warmup_steps,
        hard=False,
        summary=True)

  raise ValueError(
      "Unsupported hparams.bottleneck_kind %s" % hparams.bottleneck_kind)


def parametrized_unbottleneck(x, hidden_size, hparams):
  """Meta-function calling all the above un-bottlenecks with hparams."""
  if hparams.bottleneck_kind == "tanh_discrete":
    return tanh_discrete_unbottleneck(x, hidden_size)
  if hparams.bottleneck_kind == "isemhash":
    return isemhash_unbottleneck(x, hidden_size,
                                 hparams.isemhash_filter_size_multiplier)
  if hparams.bottleneck_kind in ["vq", "em", "gumbel_softmax"]:
    return vq_discrete_unbottleneck(x, hidden_size)
  raise ValueError(
      "Unsupported hparams.bottleneck_kind %s" % hparams.bottleneck_kind)


def iaf_hparams(hidden_size=512, filter_size=4096):
  """Create hyperpameters for inverse autoregressive flows.

  Args:
    hidden_size: Width of attention layers and neural network output layer.
    filter_size: Hidden layer width for neural network.

  Returns:
    hparams: Hyperpameters with basic presets for inverse autoregressive flows.
  """
  hparams = common_hparams.basic_params1()

  # Attention hyperparameters.
  hparams.hidden_size = hidden_size
  hparams.add_hparam("attention_key_channels", None)
  hparams.add_hparam("attention_value_channels", None)
  hparams.add_hparam("num_heads", 4)
  hparams.add_hparam("attention_dropout", 0.1)
  hparams.add_hparam("shared_rel", False)
  hparams.add_hparam("block_width", 1)
  hparams.add_hparam("block_length", 1)
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)

  # Preprocessing and postprocesing hyperparameters.
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.norm_type = "layer"
  hparams.norm_epsilon = 1e-06
  hparams.layer_prepostprocess_dropout_broadcast_dims = ""
  hparams.layer_postprocess_sequence = "da"

  # Feedforward neural network hyperparameters.
  hparams.add_hparam("filter_size", filter_size)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")
  hparams.add_hparam("relu_dropout", 0.1)
  return hparams
