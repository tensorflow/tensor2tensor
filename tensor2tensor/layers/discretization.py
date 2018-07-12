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
"""Discretization bottlenecks used to train discrete latent variables."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from tensor2tensor.layers import common_layers

import tensorflow as tf

from tensorflow.python.training import moving_averages


def project_hidden(x, projection_tensors, hidden_size, num_blocks):
  """Project encoder hidden state into block_dim using projection tensors.

  Args:
    x: Encoder hidden state of shape [-1, hidden_size].
    projection_tensors: Projection tensors used to project the hidden state.
    hidden_size: Dimension of the latent space.
    num_blocks: Number of blocks in DVQ.

  Returns:
    Projected states of shape [-1, num_blocks, block_dim].
  """
  x = tf.reshape(x, shape=[1, -1, hidden_size])
  x_tiled = tf.reshape(
      tf.tile(x, multiples=[num_blocks, 1, 1]),
      shape=[num_blocks, -1, hidden_size])
  x_projected = tf.matmul(x_tiled, projection_tensors)
  x_projected = tf.transpose(x_projected, perm=[1, 0, 2])
  return x_projected


def slice_hidden(x, hidden_size, num_blocks):
  """Slice encoder hidden state into block_dim.

  Args:
    x: Encoder hidden state of shape [-1, hidden_size].
    hidden_size: Dimension of the latent space.
    num_blocks: Number of blocks in DVQ.

  Returns:
    Sliced states of shape [-1, num_blocks, block_dim].
  """
  block_dim = int(hidden_size // num_blocks)
  x_sliced = tf.reshape(x, shape=[-1, num_blocks, block_dim])
  return x_sliced


def nearest_neighbor(x,
                     means,
                     block_v_size,
                     random_top_k=1,
                     soft_em=False,
                     num_samples=1):
  """Find the nearest element in means to elements in x.

  Args:
    x: Batch of encoder continuous latent states sliced/projected into shape
      [-1, num_blocks, block_dim].
    means: Embedding table of shpae [num_blocks, block_v_size, block_dim].
    block_v_size: Number of table entries per block.
    random_top_k: Noisy top-k if this is bigger than 1 (Default: 1).
    soft_em: If True then use soft EM rather than hard EM (Default: False).
    num_samples: Number of samples to take in soft EM (Default: 1).

  Returns:
    Tensor with nearest element in mean encoded in one-hot notation
    and distances.
  """
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
    nearest_hot = tf.reduce_mean(nearest_hot, axis=-2)
  else:
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
  return nearest_hot


def embedding_lookup(x,
                     means,
                     num_blocks,
                     block_v_size,
                     random_top_k=1,
                     soft_em=False,
                     num_samples=1):
  """Compute nearest neighbors and loss for training the embeddings via DVQ.

  Args:
    x: Batch of encoder continuous latent states sliced/projected into shape
      [-1, num_blocks, block_dim].
    means: Embedding table of shape [num_blocks, block_v_size, block_dim].
    num_blocks: Number of blocks in DVQ.
    block_v_size: Number of table entries per block.
    random_top_k: Noisy top-k if this is bigger than 1 (Default: 1).
    soft_em: If True then use soft EM rather than hard EM (Default: False).
    num_samples: Number of samples to use for soft EM (Default: 1).

  Returns:
    The nearest neighbor in one hot form, the nearest neighbor itself, the
    commitment loss, embedding training loss and distances.
  """
  x_means_hot = nearest_neighbor(
      x,
      means,
      block_v_size,
      random_top_k,
      soft_em=soft_em,
      num_samples=num_samples)
  x_means_hot_flat = tf.reshape(x_means_hot, [-1, num_blocks, block_v_size])
  x_means = tf.matmul(tf.transpose(x_means_hot_flat, perm=[1, 0, 2]), means)
  x_means = tf.transpose(x_means, [1, 0, 2])
  q_loss = tf.reduce_mean(tf.square((tf.stop_gradient(x) - x_means)))
  e_loss = tf.reduce_mean(tf.square(x - tf.stop_gradient(x_means)))
  return x_means_hot, x_means, q_loss, e_loss


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
  x_labels = []
  for i in range(num_bits):
    x_labels.append(x_l[:, i] * tf.to_int32(base)**tf.to_int32(i))
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
  x_labels = []
  for i in range(num_bits):
    x_labels.append(
        tf.floormod(
            tf.floordiv(tf.to_int32(x_l),
                        tf.to_int32(base)**i), tf.to_int32(base)))
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
          name,
          bottleneck_kind="dvq",
          soft_em=False,
          num_blocks=2,
          num_residuals=1,
          block_v_size=None,
          means=None):
  """Embedding function that takes discrete latent and returns embedding.

  Args:
    x: Input to the discretization bottleneck.
    hidden_size: Dimension of the latent state.
    z_size: Number of bits used to produce discrete code; discrete codes range
      from 1 to 2**z_size.
    filter_size: Filter size to be used for the embedding function.
    name: Name for the bottleneck scope.
    bottleneck_kind: Kind of discretization bottleneck to use; one of dvq,
      semhash, gumbel-softmax (Default: dvq).
    soft_em: If True then it uses a multi-sample version of EM (Default: False).
    num_blocks: Number of blocks in DVQ (Default: 2).
    num_residuals: Number of residuals (Default: 1).
    block_v_size: Number of embedding entries per block (Default: None).
    means: The embedding table for dvq (Default: None).

  Returns:
    Continuous embedding to be passed on to the decoder.

  Raises:
    ValueError: For unknown or missing arguments.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    if bottleneck_kind == "semhash":
      c = int_to_bit(x, z_size)
      h1a = tf.layers.dense(c, filter_size, name="vch1a")
      h1b = tf.layers.dense(1.0 - c, filter_size, name="vch1b")
      h1 = h1a + h1b
    elif bottleneck_kind == "gumbel-softmax":
      hot = tf.one_hot(x, 2**z_size)
      h1 = tf.layers.dense(hot, hidden_size, name="dae_dense")
    elif bottleneck_kind == "dvq":
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


def vae(x, name, z_size):
  """Simple variational autoencoder without discretization.

  Args:
    x: Input to the discretization bottleneck.
    name: Name for the bottleneck scope.
    z_size: Number of bits used to produce discrete code; discrete codes range
      from 1 to 2**z_size.

  Returns:
    Embedding function, latent, loss, mu and log_simga.
  """
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
                   name,
                   z_size,
                   mode,
                   softmax_k=0,
                   kl_warmup_steps=150000,
                   summary=True):
  """Gumbel softmax discretization bottleneck.

  Args:
    x: Input to the discretization bottleneck.
    name: Name for the bottleneck scope.
    z_size: Number of bits used to produce discrete code; discrete codes range
      from 1 to 2**z_size.
    mode: Mode represents whether we are training or testing for bottlenecks
      that differ in behavior (Default: None).
    softmax_k: If > 1 then do top-k softmax (Default: 0).
    kl_warmup_steps: Number of steps for kl warmup (Default: 150000).
    summary: If True, then write summaries (Default: True).

  Returns:
    Embedding function, discrete code and loss.
  """
  with tf.variable_scope(name):
    m = tf.layers.dense(x, 2**z_size, name="mask")
    if softmax_k > 0:
      m, kl = top_k_softmax(m, softmax_k)
      return m, m, 1.0 - tf.reduce_mean(kl)
    logsm = tf.nn.log_softmax(m)

    # Gumbel-softmax sample.
    gumbel_samples = gumbel_sample(common_layers.shape_list(m))
    steps = kl_warmup_steps
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
    d_variance = tf.reduce_mean(tf.square(distrib - d_mean), axis=[0])
    d_dev = -tf.reduce_mean(d_variance)
    ret = s

    if mode != tf.contrib.learn.ModeKeys.TRAIN:
      ret = tf.reshape(maxvhot, common_layers.shape_list(s))  # Just hot @eval.
    return m, ret, d_dev * 5.0 + tf.reduce_mean(kl) * 0.002


def discrete_bottleneck(x,
                        hidden_size,
                        z_size,
                        filter_size,
                        name,
                        mode=None,
                        startup_steps=50000,
                        bottleneck_kind="dvq",
                        num_blocks=2,
                        num_residuals=1,
                        reshape_method="slice",
                        projection_tensors=None,
                        means=None,
                        beta=0.25,
                        noise_dev=1.,
                        decay=0.999,
                        discrete_mix=0.5,
                        random_top_k=1,
                        soft_em=False,
                        num_samples=1,
                        epsilon=1e-5,
                        softmax_k=0,
                        kl_warmup_steps=150000,
                        ema=True,
                        ema_count=None,
                        ema_means=None,
                        summary=True):
  """Discretization bottleneck for latent variables.

  Args:
    x: Input to the discretization bottleneck.
    hidden_size: Dimension of the latent state.
    z_size: Number of bits used to produce discrete code; discrete codes range
      from 1 to 2**z_size.
    filter_size: Filter size to be used for the embedding function.
    name: Name for the bottleneck scope.
    mode: Mode represents whether we are training or testing for bottlenecks
      that differ in behavior (Default: None).
    startup_steps: Number of steps after which latent predictor is trained
      (Default: 50000).
    bottleneck_kind: Kind of discretization bottleneck to use; one of dvq,
      semhash, gumbel-softmax (Default: dvq).
    num_blocks: Number of blocks to use for decomposed vector
      quantization (Default: 2).
    num_residuals: Number of residual units used to compute nearest
      neighbors (Default: 1).
    reshape_method: Method to reshape for DVQ (Default: slice).
    projection_tensors: If the reshape method is project, then these are the
      tensors used to project (Default: None).
    means: The embedding table for dvq (Default: None).
    beta: Beta factor for the DVQ loss (Default: 0.25).
    noise_dev: Stddev for noise added for semhash (Default: 0).
    decay: Decay factor for the exponential moving average (Default: 0.999).
    discrete_mix: Factor for mixing discrete and non-discrete input for semhash
      (Default: 0.5).
    random_top_k: Noisy top-k for DVQ (Default: 1).
    soft_em: If True then use soft EM rather than hard EM (Default: False).
    num_samples: Number of samples for soft EM (Default: 1).
    epsilon: Epsilon parameter for DVQ (Default: 1e-5).
    softmax_k: If > 1 then do top-k softmax (Default: 0).
    kl_warmup_steps: Number of steps for kl warmup (Default: 150000).
    ema: If True update embeddings using exponential moving averages (Default:
      True).
    ema_count: Table of counts for each embedding corresponding to how many
      examples in a batch it was the closest to (Default: None).
    ema_means: Exponentially averaged version of the embeddings (Default: None).
    summary: If True, then write summaries (Default: True).

  Returns:
    Embedding to pass to the decoder, discrete latent, loss, and the embedding
    function.

  Raises:
    ValueError: If projection_tensors is None for reshape_method project, or
    ema_count or ema_means is None if we are using ema, or unknown args.
  """
  block_v_size = None
  if bottleneck_kind == "dvq":
    # Define the dvq parameters
    assert means is not None

    # Check block dimensions add up
    if hidden_size % num_blocks != 0:
      raise ValueError("num_blocks does not divide hidden size")

    if z_size % num_residuals != 0:
      raise ValueError("num_residuals does not divide embedding table size")

    z_size_per_residual = int(z_size / num_residuals)

    if z_size_per_residual % num_blocks != 0:
      raise ValueError("num_blocks does not divide embedding table size")

    block_v_size = 2**(z_size_per_residual / num_blocks)
    block_v_size = int(block_v_size)

    # Set the reshape method corresponding to projections or slices
    if reshape_method == "slice":
      reshape_fn = partial(
          slice_hidden, hidden_size=hidden_size, num_blocks=num_blocks)
    elif reshape_method == "project":
      if projection_tensors is None:
        raise ValueError(
            "Projection tensors is None for reshape_method project")
      reshape_fn = partial(
          project_hidden,
          projection_tensors=projection_tensors,
          hidden_size=hidden_size,
          num_blocks=num_blocks)
    else:
      raise ValueError("Unknown reshape_method")

    # Check if the ema settings make sense
    if ema:
      if ema_count is None:
        raise ValueError("ema_count is None but ema is True")
      if ema_means is None:
        raise ValueError("ema_means is None but ema is True")

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    l = tf.constant(0.0)
    if bottleneck_kind == "dense":
      c = tf.layers.dense(x, z_size, name="vcc")
      h1 = tf.layers.dense(c, filter_size, name="vch1")
    elif bottleneck_kind == "vae":
      c, l, _, _ = vae(x, z_size, "vae")
      h1 = tf.layers.dense(c, filter_size, name="vch1")
    elif bottleneck_kind == "semhash":
      c = tf.layers.dense(x, z_size, name="vcc")
      y_clean = common_layers.saturating_sigmoid(c)
      if summary:
        tf.summary.histogram("y_clean", tf.reshape(y_clean, [-1]))
      if noise_dev > 0 and mode == tf.estimator.ModeKeys.TRAIN:
        noise = tf.truncated_normal(
            common_layers.shape_list(c), mean=0.0, stddev=noise_dev)
        y = common_layers.saturating_sigmoid(c + noise)
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
      h1a = tf.layers.dense(c, filter_size, name="vch1a")
      h1b = tf.layers.dense(1.0 - c, filter_size, name="vch1b")
      h1 = h1a + h1b
      dx = tf.to_int32(tf.stop_gradient(d))
      c = bit_to_int(dx, z_size)
    elif bottleneck_kind == "gumbel-softmax":
      _, hot, l = gumbel_softmax(x, name, z_size, mode, softmax_k,
                                 kl_warmup_steps, summary)
      c = tf.argmax(hot, axis=-1)
      h1 = tf.layers.dense(hot, hidden_size, name="dae_dense")
    elif bottleneck_kind == "dvq":
      x_reshaped = reshape_fn(x)
      x_res = x_reshaped
      x_means_hot = []
      x_means = 0
      l = 0
      for i in range(num_residuals):
        x_means_hot_res, x_means_res, q_loss_res, e_loss_res = embedding_lookup(
            x_res, means[i], num_blocks, block_v_size, random_top_k, soft_em,
            num_samples)
        # Update the ema variables
        if ema:
          tf.logging.info("Using EMA with beta = {}".format(beta))
          updated_ema_count_res = moving_averages.assign_moving_average(
              ema_count[i],
              tf.reduce_sum(
                  tf.reshape(
                      x_means_hot_res, shape=[-1, num_blocks, block_v_size]),
                  axis=0),
              decay,
              zero_debias=False)

          dw = tf.matmul(
              tf.transpose(x_means_hot_res, perm=[1, 2, 0]),
              tf.transpose(x_res, perm=[1, 0, 2]))

          updated_ema_means_res = moving_averages.assign_moving_average(
              ema_means[i], dw, decay, zero_debias=False)
          n = tf.reduce_sum(updated_ema_count_res, axis=-1, keep_dims=True)
          updated_ema_count_res = ((updated_ema_count_res + epsilon) /
                                   (n + 2**z_size * epsilon) * n)
          # pylint: disable=g-no-augmented-assignment
          updated_ema_means_res = updated_ema_means_res / tf.expand_dims(
              updated_ema_count_res, axis=-1)
          # pylint: enable=g-no-augmented-assignment

          with tf.control_dependencies([e_loss_res]):
            update_means_res = tf.assign(means[i], updated_ema_means_res)
            with tf.control_dependencies([update_means_res]):
              l += beta * e_loss_res
        else:
          l += q_loss_res + beta * e_loss_res

        # Update the residuals
        x_res -= x_means_res
        x_means += x_means_res
        x_means_hot.append(x_means_hot_res)

      # Get the discrete latent representation
      x_means_hot = tf.stack(x_means_hot, axis=1)
      x_means_idx = tf.argmax(x_means_hot, axis=-1)

      # Get the binary representation
      x_means_bits = int_to_bit(
          x_means_idx,
          num_bits=int(z_size / (num_residuals * num_blocks)),
          base=2)
      shape = common_layers.shape_list(x_means_bits)
      new_shape = shape[:-2]
      new_shape[-1] = z_size
      x_means_bits = tf.reshape(x_means_bits, shape=new_shape)
      c = bit_to_int(tf.to_int32(x_means_bits), num_bits=z_size, base=2)

      # Adjust shape of c
      shape_x = common_layers.shape_list(x)
      new_shape = shape_x[:-1]
      c = tf.reshape(c, new_shape)

      # If we are doing soft EM then c is x_means_hot
      if soft_em:
        c = x_means_hot
        new_shape.append(block_v_size)
        c = tf.reshape(c, new_shape)

      x_means = tf.reshape(x_means, shape_x)
      x_reshaped = tf.reshape(x_reshaped, shape_x)
      h1 = x_reshaped + tf.stop_gradient(x_means - x_reshaped)
    else:
      raise ValueError("Unknown discretization method.")

    res = h1

    embed_fn = partial(
        embed,
        hidden_size=hidden_size,
        z_size=z_size,
        filter_size=filter_size,
        name=name,
        bottleneck_kind=bottleneck_kind,
        soft_em=soft_em,
        num_blocks=num_blocks,
        num_residuals=num_residuals,
        block_v_size=block_v_size,
        means=means)
    return res, c, l, embed_fn


# New API for discretization bottlenecks:
# * Each method is separate and provides 2 functions:
# * The [method]_bottleneck function returns discretized state.
# * The [method]_unbottleneck function moves from discretized state to dense.


def get_vq_bottleneck(bottleneck_size, hidden_size):
  """Get lookup table for VQ bottleneck."""
  with tf.variable_scope("vq", reuse=tf.AUTO_REUSE):
    means = tf.get_variable(
        name="means",
        shape=[bottleneck_size, hidden_size],
        initializer=tf.uniform_unit_scaling_initializer())

    ema_count = tf.get_variable(
        name="ema_count",
        shape=[bottleneck_size],
        initializer=tf.constant_initializer(0),
        trainable=False)

    with tf.colocate_with(means):
      ema_means = tf.get_variable(
          name="ema_means",
          initializer=means.initialized_value(),
          trainable=False)

  return means, ema_means, ema_count


def vq_nearest_neighbor(x, means, soft_em=False, num_samples=10):
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
    x_means_idx = tf.argmax(-dist, axis=-1)
    x_means_hot = tf.one_hot(x_means_idx, bottleneck_size)
  x_means_hot_flat = tf.reshape(x_means_hot, [-1, bottleneck_size])
  x_means = tf.matmul(x_means_hot_flat, means)
  e_loss = tf.reduce_mean(tf.square(x - tf.stop_gradient(x_means)))
  return x_means_hot, e_loss


def vq_discrete_bottleneck(x,
                           bottleneck_bits,
                           beta=0.25,
                           decay=0.999,
                           epsilon=1e-5,
                           soft_em=False,
                           num_samples=10):
  """Simple vector quantized discrete bottleneck."""
  bottleneck_size = 2**bottleneck_bits
  x_shape = common_layers.shape_list(x)
  hidden_size = x_shape[-1]
  means, ema_means, ema_count = get_vq_bottleneck(bottleneck_size, hidden_size)
  x = tf.reshape(x, [-1, hidden_size])
  x_means_hot, e_loss = vq_nearest_neighbor(
      x, means, soft_em=soft_em, num_samples=num_samples)

  # Update the ema variables
  updated_ema_count = moving_averages.assign_moving_average(
      ema_count,
      tf.reduce_sum(
          tf.reshape(x_means_hot, shape=[-1, bottleneck_size]), axis=0),
      decay,
      zero_debias=False)

  dw = tf.matmul(x_means_hot, x, transpose_a=True)
  updated_ema_means = tf.identity(moving_averages.assign_moving_average(
      ema_means, dw, decay, zero_debias=False))
  n = tf.reduce_sum(updated_ema_count, axis=-1, keepdims=True)
  updated_ema_count = (
      (updated_ema_count + epsilon) / (n + bottleneck_size * epsilon) * n)
  updated_ema_means /= tf.expand_dims(updated_ema_count, axis=-1)
  with tf.control_dependencies([e_loss]):
    update_means = means.assign(updated_ema_means)
    with tf.control_dependencies([update_means]):
      loss = beta * e_loss

  d = tf.reshape(x_means_hot, x_shape[:-1] + [bottleneck_size])
  return d, loss


def vq_discrete_unbottleneck(x, hidden_size):
  """Simple undiscretization from vector quantized representation."""
  x_shape = common_layers.shape_list(x)
  x = tf.to_float(x)
  bottleneck_size = common_layers.shape_list(x)[-1]
  means, _, _ = get_vq_bottleneck(bottleneck_size, hidden_size)
  result = tf.matmul(tf.reshape(x, [-1, x_shape[-1]]), means)
  return tf.reshape(result, x_shape[:-1] + [hidden_size])


def gumbel_softmax_discrete_bottleneck(x,
                                       bottleneck_bits,
                                       beta=0.25,
                                       decay=0.999,
                                       epsilon=1e-5,
                                       startup_steps=15000,
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
    startup_steps: Number of steps for KL warmup (Default: 25000).
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
  means, ema_means, ema_count = get_vq_bottleneck(bottleneck_size, hidden_size)
  x = tf.reshape(x, [-1, hidden_size])

  bottleneck_size = common_layers.shape_list(means)[0]
  x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keepdims=True)
  scalar_prod = tf.matmul(x, means, transpose_b=True)
  dist = x_norm_sq + tf.transpose(means_norm_sq) - 2 * scalar_prod

  class_probs = tf.nn.softmax(dist)
  log_class_probs = tf.nn.log_softmax(dist)
  gumbel_samples = gumbel_sample(common_layers.shape_list(dist))
  gumbel_samples *= common_layers.inverse_exp_decay(startup_steps // 5) * 0.5
  temperature = 1.2 - common_layers.inverse_lin_decay(startup_steps)

  # 10% of the time keep reasonably high temperature to keep learning.
  temperature = tf.cond(
      tf.less(tf.random_uniform([]), 0.9), lambda: temperature,
      lambda: tf.random_uniform([], minval=0.5, maxval=1.0))
  gumbel_softmax_samples = tf.nn.softmax(
      (log_class_probs + gumbel_samples) / temperature)

  # Calculate KL between q and a uniform prior.
  kl = tf.reduce_sum(class_probs * (log_class_probs -
                                    tf.log(1.0/bottleneck_size)), -1)
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
  x_means_assignments_flat = tf.reshape(
      x_means_assignments, [-1, bottleneck_size])
  x_means = tf.matmul(x_means_assignments_flat, means)
  commitment_loss = tf.reduce_mean(tf.square(x - tf.stop_gradient(x_means)))

  # Update the ema variables.
  updated_ema_count = moving_averages.assign_moving_average(
      ema_count,
      tf.reduce_sum(
          tf.reshape(x_means_assignments, shape=[-1, bottleneck_size]), axis=0),
      decay,
      zero_debias=False)

  dw = tf.matmul(x_means_assignments, x, transpose_a=True)
  updated_ema_means = tf.identity(moving_averages.assign_moving_average(
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

  x_means_assignments = tf.reshape(
      x_means_assignments, x_shape[:-1] + [bottleneck_size])
  return x_means_assignments, loss


def tanh_discrete_bottleneck(x, bottleneck_bits, bottleneck_noise,
                             discretize_warmup_steps, mode):
  """Simple discretization through tanh, flip bottleneck_noise many bits."""
  x = tf.tanh(tf.layers.dense(x, bottleneck_bits,
                              name="tanh_discrete_bottleneck"))
  d = x + tf.stop_gradient(2.0 * tf.to_float(tf.less(0.0, x)) - 1.0 - x)
  if mode == tf.estimator.ModeKeys.TRAIN:
    noise = tf.random_uniform(common_layers.shape_list(x))
    noise = 2.0 * tf.to_float(tf.less(bottleneck_noise, noise)) - 1.0
    d *= noise
  d = common_layers.mix(d, x, discretize_warmup_steps,
                        mode == tf.estimator.ModeKeys.TRAIN)
  return d, 0.0


def tanh_discrete_unbottleneck(x, hidden_size):
  """Simple un-discretization from tanh."""
  x = tf.layers.dense(x, hidden_size, name="tanh_discrete_unbottleneck")
  return x


def isemhash_bottleneck(x, bottleneck_bits, bottleneck_noise,
                        discretize_warmup_steps, mode,
                        isemhash_noise_dev=0.5, isemhash_mix_prob=0.5):
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
      d = common_layers.mix(d, 2.0 * y - 1.0, discretize_warmup_steps,
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
    return tanh_discrete_bottleneck(
        x, hparams.bottleneck_bits, hparams.bottleneck_noise * 0.5,
        hparams.discretize_warmup_steps, hparams.mode)
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
    return gumbel_softmax_discrete_bottleneck(x, hparams.bottleneck_bits,
                                              hparams.vq_beta, hparams.vq_decay,
                                              hparams.vq_epsilon,
                                              hparams.startup_steps, hard=False,
                                              summary=True)

  raise ValueError("Unsupported hparams.bottleneck_kind %s"
                   % hparams.bottleneck_kind)


def parametrized_unbottleneck(x, hidden_size, hparams):
  """Meta-function calling all the above un-bottlenecks with hparams."""
  if hparams.bottleneck_kind == "tanh_discrete":
    return tanh_discrete_unbottleneck(x, hidden_size)
  if hparams.bottleneck_kind == "isemhash":
    return isemhash_unbottleneck(
        x, hidden_size, hparams.isemhash_filter_size_multiplier)
  if hparams.bottleneck_kind in ["vq", "em", "gumbel_softmax"]:
    return vq_discrete_unbottleneck(x, hidden_size)
  raise ValueError("Unsupported hparams.bottleneck_kind %s"
                   % hparams.bottleneck_kind)
