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

"""Discretization bottlenecks used to train discrete latent variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial
# Dependency imports
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
                     inv_temp=1.0,
                     ema_count=None,
                     c_probs=None):
  """Find the nearest element in means to elements in x.

  Args:
    x: Batch of encoder continuous latent states sliced/projected into shape
      [-1, num_blocks, block_dim].
    means: Embedding table of shpae [num_blocks, block_v_size, block_dim].
    block_v_size: Number of table entries per block.
    random_top_k: Noisy top-k if this is bigger than 1 (Default: 1).
    soft_em: If True then use soft EM rather than hard EM (Default: False).
    inv_temp: Inverse temperature for soft EM (Default: 1.)
    ema_count: Table of counts for each embedding corresponding to how many
      examples in a batch it was the closest to (Default: None).
    c_probs: Precomputed probablities of clusters may be given, for example in
      the case of smoothed l0 priors.
  Returns:
    Tensor with nearest element in mean encoded in one-hot notation.
  """
  x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keep_dims=True)
  means_norm_sq = tf.reduce_sum(tf.square(means), axis=-1, keep_dims=True)
  scalar_prod = tf.matmul(
      tf.transpose(x, perm=[1, 0, 2]), tf.transpose(means, perm=[0, 2, 1]))
  scalar_prod = tf.transpose(scalar_prod, perm=[1, 0, 2])
  dist = x_norm_sq + tf.transpose(
      means_norm_sq, perm=[2, 0, 1]) - 2 * scalar_prod
  # computing cluster probabilities
  if soft_em or c_probs is not None:
    if c_probs is not None:
      # expand dims to match inv temp
      c_probs = tf.expand_dims(c_probs, 0)
    else:
      ema_count = tf.expand_dims(ema_count+1., 0)
      c_probs = ema_count / tf.reduce_sum(ema_count, 2, keepdims=True)
  if soft_em:
    nearest_hot = tf.nn.softmax(-inv_temp * dist, axis=-1) * c_probs
    nearest_hot /= tf.reduce_sum(nearest_hot, 2, keepdims=True)
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
                     inv_temp=1.0,
                     ema_count=None,
                     c_probs=None):
  """Compute nearest neighbors and loss for training the embeddings via DVQ.

  Args:
    x: Batch of encoder continuous latent states sliced/projected into shape
      [-1, num_blocks, block_dim].
    means: Embedding table of shape [num_blocks, block_v_size, block_dim].
    num_blocks: Number of blocks in DVQ.
    block_v_size: Number of table entries per block.
    random_top_k: Noisy top-k if this is bigger than 1 (Default: 1).
    soft_em: If True then use soft EM rather than hard EM (Default: False).
    inv_temp: Inverse temperature for soft EM (Default: 1.)
    ema_count: Table of counts for each embedding corresponding to how many
      examples in a batch it was the closest to (Default: None).
    c_probs: precomputed cluster probabilities might be passed, for example in
      the case of smoothed L0.

  Returns:
    The nearest neighbor in one hot form, the nearest neighbor itself, the
    commitment loss, embedding training loss.
  """
  x_means_hot = nearest_neighbor(x, means, block_v_size, random_top_k, soft_em,
                                 inv_temp, ema_count, c_probs)
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


def embed(x,
          hidden_size,
          z_size,
          filter_size,
          name,
          bottleneck_kind='dvq',
          num_blocks=2,
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
      semhash, gumbel-softmax.
    num_blocks: Number of blocks in DVQ.
    block_v_size: Number of embedding entries per block.
    means: The embedding table for dvq (Default: None).

  Returns:
    Continuous embedding to be passed on to the decoder.

  Raises:
    ValueError: For unknown or missing arguments.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    if bottleneck_kind == 'semhash':
      c = int_to_bit(x, z_size)
      h1a = tf.layers.dense(c, filter_size, name='vch1a')
      h1b = tf.layers.dense(1.0 - c, filter_size, name='vch1b')
      h1 = h1a + h1b
    elif bottleneck_kind == 'gumbel-softmax':
      hot = tf.one_hot(x, 2**z_size)
      h1 = tf.layers.dense(hot, hidden_size, name='dae_dense')
    elif bottleneck_kind == 'dvq':
      if block_v_size is None:
        raise ValueError('Bottleneck kind is dvq but block_v_size is None.')

      shape_x = common_layers.shape_list(x)
      x_flat = tf.reshape(x, [-1, 1])
      c = int_to_bit(x_flat, num_bits=z_size, base=2)
      shape = common_layers.shape_list(c)
      new_shape = shape
      new_shape[-1] = num_blocks
      new_shape.append(int(z_size / num_blocks))
      c = tf.to_int32(tf.reshape(c, shape=new_shape))
      c = bit_to_int(c, num_bits=int(z_size / num_blocks), base=2)
      c_hot = tf.one_hot(c, depth=block_v_size, axis=-1)
      c_hot_flat = tf.reshape(c_hot, shape=[-1, num_blocks, block_v_size])
      h1 = tf.matmul(tf.transpose(c_hot_flat, perm=[1, 0, 2]), means)
      h1 = tf.transpose(h1, perm=[1, 0, 2])
      new_shape = shape_x
      new_shape.append(hidden_size)
      h1 = tf.reshape(h1, new_shape)
    elif bottleneck_kind == 'rounding':
      h1 = x
    else:
      raise ValueError('Unknown bottleneck kind.')

    h2 = tf.layers.dense(tf.nn.relu(h1), filter_size, name='vch2')
    return tf.layers.dense(tf.nn.relu(h2), hidden_size, name='vcfin')


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
    mu = tf.layers.dense(x, z_size, name='mu')
    log_sigma = tf.layers.dense(x, z_size, name='log_sigma')
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
    m = tf.layers.dense(x, 2**z_size, name='mask')
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
      tf.summary.histogram('max-log', tf.reshape(kl, [-1]))

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
                        bottleneck_kind='dvq',
                        num_blocks=2,
                        reshape_method='slice',
                        projection_tensors=None,
                        means=None,
                        beta=0.25,
                        noise_dev=1.,
                        decay=0.999,
                        discrete_mix=0.5,
                        random_top_k=1,
                        soft_em=False,
                        inv_temp=1.0,
                        epsilon=1e-5,
                        softmax_k=0,
                        kl_warmup_steps=150000,
                        ema=True,
                        ema_count=None,
                        ema_means=None,
                        summary=True,
                        dp_strength=1.0,
                        dp_decay=1.0,
                        dp_alpha=0.5,
                        slo=False,
                        slo_alpha=10,
                        slo_beta=0.5,
                        c_logits=None):
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
    num_blocks: Number of blocks to use for decomposed vector quantization.
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
    inv_temp: Inverse temperature for soft EM (Default: 1.)
    epsilon: Epsilon parameter for DVQ (Default: 1e-5).
    softmax_k: If > 1 then do top-k softmax (Default: 0).
    kl_warmup_steps: Number of steps for kl warmup (Default: 150000).
    ema: If True update embeddings using exponential moving averages (Default:
      True).
    ema_count: Table of counts for each embedding corresponding to how many
      examples in a batch it was the closest to (Default: None).
    ema_means: Exponentially averaged version of the embeddings (Default: None).
    summary: If True, then write summaries (Default: True).
    dp_strength: Strength of Dirichlet Process loss prior (Default: 1.0).
    dp_decay: Decay the dp_strength using an exponential decay using this
      term (Default: 1.0).
    dp_alpha: Alpha term (pseudo-count) in Dirichlet Process (Default: 0.5).
    slo: Smoothed L0
    slo_alpha: alpha for smoothed L0
    slo_beta: beta for smoothed L0
    c_logits: a [num_blocks, block_size] tensor of logits for
      computing cluster probabilities.

  Returns:
    Embedding to pass to the decoder, discrete latent, loss, and the embedding
    function.

  Raises:
    ValueError: If projection_tensors is None for reshape_method project, or
    ema_count or ema_means is None if we are using ema, or unknown args.
  """
  block_v_size = None
  if bottleneck_kind == 'dvq':
    # Define the dvq parameters
    assert means is not None

    # Check block dimensions add up
    if hidden_size % num_blocks != 0:
      raise ValueError('num_blocks does not divide hidden size')

    if 2**z_size % num_blocks != 0:
      raise ValueError('num_blocks does not divide embedding table size')

    block_v_size = 2**(z_size / num_blocks)
    block_v_size = int(block_v_size)

    # Set the reshape method corresponding to projections or slices
    if reshape_method == 'slice':
      reshape_fn = partial(
          slice_hidden, hidden_size=hidden_size, num_blocks=num_blocks)
    elif reshape_method == 'project':
      if projection_tensors is None:
        raise ValueError(
            'Projection tensors is None for reshape_method project')
      reshape_fn = partial(
          project_hidden,
          projection_tensors=projection_tensors,
          hidden_size=hidden_size,
          num_blocks=num_blocks)
    else:
      raise ValueError('Unknown reshape_method')

    # Check if the ema settings make sense
    if ema:
      if ema_count is None:
        raise ValueError('ema_count is None but ema is True')
      if ema_means is None:
        raise ValueError('ema_means is None but ema is True')

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    l = tf.constant(0.0)
    if bottleneck_kind == 'dense':
      c = tf.layers.dense(x, z_size, name='vcc')
      h1 = tf.layers.dense(c, filter_size, name='vch1')
    elif bottleneck_kind == 'vae':
      c, l, _, _ = vae(x, z_size, 'vae')
      h1 = tf.layers.dense(c, filter_size, name='vch1')
    elif bottleneck_kind == 'semhash':
      c = tf.layers.dense(x, z_size, name='vcc')
      y_clean = common_layers.saturating_sigmoid(c)
      if summary:
        tf.summary.histogram('y_clean', tf.reshape(y_clean, [-1]))
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
      h1a = tf.layers.dense(c, filter_size, name='vch1a')
      h1b = tf.layers.dense(1.0 - c, filter_size, name='vch1b')
      h1 = h1a + h1b
      dx = tf.to_int32(tf.stop_gradient(d))
      c = bit_to_int(dx, z_size)
    elif bottleneck_kind == 'gumbel-softmax':
      _, hot, l = gumbel_softmax(x, name, z_size, mode, softmax_k,
                                 kl_warmup_steps, summary)
      c = tf.argmax(hot, axis=-1)
      h1 = tf.layers.dense(hot, hidden_size, name='dae_dense')
    elif bottleneck_kind == 'dvq':
      c_probs = None
      if c_logits is not None:
        c_probs = tf.nn.softmax(c_logits, axis=-1)
      x_reshaped = reshape_fn(x)
      x_means_hot, x_means, q_loss, e_loss = embedding_lookup(
          x_reshaped, means, num_blocks, block_v_size, random_top_k, soft_em,
          inv_temp, ema_count, c_probs)

      # Get the discrete latent represenation
      x_means_idx = tf.argmax(x_means_hot, axis=-1)

      # Get the binary representation
      x_means_bits = int_to_bit(
          x_means_idx, num_bits=int(z_size / num_blocks), base=2)
      shape = common_layers.shape_list(x_means_bits)
      new_shape = shape[:-1]
      new_shape[-1] = z_size
      x_means_bits = tf.reshape(x_means_bits, shape=new_shape)
      c = bit_to_int(tf.to_int32(x_means_bits), num_bits=z_size, base=2)

      # Adjust shape of c
      shape_x = common_layers.shape_list(x)
      new_shape = shape_x[:-1]
      c = tf.reshape(c, new_shape)

      # Update the ema variables
      if ema:
        tf.logging.info('Using EMA with beta = {}'.format(beta))
        updated_ema_count = moving_averages.assign_moving_average(
            ema_count,
            tf.reduce_sum(
                tf.reshape(x_means_hot, shape=[-1, num_blocks, block_v_size]),
                axis=0),
            decay,
            zero_debias=False)

        # Adding a term that puts a Dirichlet prior over cluster probabilities
        # Hopefully it'll encourage rich get richer behaviors
        dp_prior_loss = 0.
        slo_loss = 0.
        if dp_strength > 0.0:
          # Decay dp_strength over time to make it less important
          dp_strength = tf.train.exponential_decay(
              dp_strength,
              global_step=tf.to_int32(tf.train.get_global_step()),
              decay_steps=20000,
              decay_rate=dp_decay)
          dp_count = ema_count + dp_alpha
          p = dp_count / tf.reduce_sum(dp_count, 1, keepdims=True)
          dp_prior_loss = tf.log(p)
          dp_prior_loss = -1.0 * tf.reduce_sum(dp_prior_loss)
          dp_prior_loss /= (num_blocks * block_v_size)

        # if using smoothed L0
        if slo:
          # expected log likelihood
          ell = tf.reduce_sum(ema_count * tf.log(c_probs))
          # the prior component in the loss for MAP EM.
          slo_prior = slo_alpha * tf.reduce_sum(tf.exp(-1.*c_probs/slo_beta))
          slo_loss = -1. * (ell + slo_prior)/(num_blocks * block_v_size)
        x_means_hot_flat = tf.reshape(
            x_means_hot, shape=[-1, num_blocks, block_v_size])
        dw = tf.matmul(
            tf.transpose(x_means_hot_flat, perm=[1, 2, 0]),
            tf.transpose(x_reshaped, perm=[1, 0, 2]))
        updated_ema_means = moving_averages.assign_moving_average(
            ema_means, dw, decay, zero_debias=False)
        n = tf.reduce_sum(updated_ema_count, axis=-1, keep_dims=True)
        updated_ema_count = ((updated_ema_count + epsilon) /
                             (n + 2**z_size * epsilon) * n)
        updated_ema_means /= tf.expand_dims(updated_ema_count, axis=-1)

        with tf.control_dependencies([e_loss]):
          update_means = tf.assign(means, updated_ema_means)
          with tf.control_dependencies([update_means]):
            l = beta * e_loss + dp_strength * dp_prior_loss + slo_loss
      else:
        l = q_loss + beta * e_loss

      x_means = tf.reshape(x_means, shape_x)
      x_reshaped = tf.reshape(x_reshaped, shape_x)
      h1 = x_reshaped + tf.stop_gradient(x_means - x_reshaped)
    else:
      raise ValueError('Unknown discretization method.')

    h2 = tf.layers.dense(tf.nn.relu(h1), filter_size, name='vch2')
    res = tf.layers.dense(tf.nn.relu(h2), hidden_size, name='vcfin')

    embed_fn = partial(
        embed,
        hidden_size=hidden_size,
        z_size=z_size,
        filter_size=filter_size,
        name=name,
        bottleneck_kind=bottleneck_kind,
        num_blocks=num_blocks,
        block_v_size=block_v_size,
        means=means)
    return res, c, l, embed_fn
