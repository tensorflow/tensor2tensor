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
"""Clean discrete bottleneck as in https://arxiv.org/abs/1805.11063."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial

from tensor2tensor.layers import common_layers

import tensorflow as tf

from tensorflow.python.training import moving_averages


class DiscreteBottleneck(object):
  """Discrete bottleneck class."""

  def __init__(self, hparams):
    self.hparams = hparams

    # Set the discretization bottleneck specific things here
    self.hparams.z_size_per_residual = self.hparams.z_size // \
                                       self.hparams.num_residuals
    self.hparams.block_dim = int(
        self.hparams.hidden_size // self.hparams.num_blocks)
    self.hparams.block_v_size = 2**(
        self.hparams.z_size_per_residual / self.hparams.num_blocks)
    self.hparams.block_v_size = int(self.hparams.block_v_size)
    self.hparams.means = tf.get_variable(
        name="means",
        shape=[
            self.hparams.num_residuals, self.hparams.num_blocks,
            self.hparams.block_v_size, self.hparams.block_dim
        ],
        initializer=tf.uniform_unit_scaling_initializer())
    tf.logging.info("Done creating means")

    # Create the shadow variables if we are using EMA
    self.hparams.ema_count = None
    self.hparams.ema_means = None
    if self.hparams.ema:
      self.hparams.ema_count = []
      self.hparams.ema_means = []
      for i in range(hparams.num_residuals):
        ema_count_i = tf.get_variable(
            "ema_count_{}".format(i),
            [self.hparams.num_blocks, self.hparams.block_v_size],
            initializer=tf.constant_initializer(0),
            trainable=False)
        self.hparams.ema_count.append(ema_count_i)

      with tf.colocate_with(self.hparams.means):
        self.ema_means = []
        for i in range(hparams.num_residuals):
          ema_means_i = tf.get_variable(
              "ema_means_{}".format(i),
              initializer=self.hparams.means.initialized_value()[i],
              trainable=False)
          self.hparams.ema_means.append(ema_means_i)

  def slice_hidden(self, x):
    """Slice encoder hidden state into block_dim.

    Args:
        x: Encoder hidden state of shape [-1, hidden_size].

    Returns:
        Sliced states of shape [-1, num_blocks, block_dim].
    """
    x_sliced = tf.reshape(
        x, shape=[-1, self.hparams.num_blocks, self.hparams.block_dim])
    return x_sliced

  def nearest_neighbor(self, x, means):
    """Find the nearest element in means to elements in x.

    Args:
        x: Batch of encoder continuous latent states sliced/projected into
           shape [-1, num_blocks, block_dim].
        means: Embedding means of shape.

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

    if self.hparams.soft_em:
      nearest_idx = tf.stack(
          [
              tf.multinomial(
                  -dist[:, i, :], num_samples=self.hparams.num_samples)
              for i in range(self.hparams.num_blocks)
          ],
          axis=1)
      nearest_hot = tf.one_hot(nearest_idx, depth=self.hparams.block_v_size)
      nearest_hot = tf.reduce_mean(nearest_hot, axis=-2)
    else:
      if self.hparams.random_top_k > 1:
        _, top_k_idx = tf.nn.top_k(-dist, k=self.hparams.random_top_k)
        nearest_idx = tf.gather(
            top_k_idx,
            tf.random_uniform(
                [1],
                minval=0,
                maxval=self.hparams.random_top_k - 1,
                dtype=tf.int32),
            axis=-1)
      else:
        if self.hparams.use_scales:
          dist /= tf.reshape(self.hparams.scales,
                             [1, 1, self.hparams.moe_num_experts])
        nearest_idx = tf.argmax(-dist, axis=-1)
      nearest_hot = tf.one_hot(nearest_idx, self.hparams.block_v_size)
    return nearest_hot

  def embedding_lookup(self, x, means):
    """Compute nearest neighbors and loss for training the embeddings.

    Args:
        x: Batch of encoder continuous latent states sliced/projected into
        shape
        [-1, num_blocks, block_dim].
        means: Embedding means.

    Returns:
        The nearest neighbor in one hot form, the nearest neighbor
        itself, the
        commitment loss, embedding training loss.
    """
    x_means_hot = self.nearest_neighbor(x, means)
    x_means_hot_flat = tf.reshape(
        x_means_hot, [-1, self.hparams.num_blocks, self.hparams.block_v_size])
    x_means = tf.matmul(tf.transpose(x_means_hot_flat, perm=[1, 0, 2]), means)
    x_means = tf.transpose(x_means, [1, 0, 2])
    q_loss = tf.reduce_mean(tf.square((tf.stop_gradient(x) - x_means)))
    e_loss = tf.reduce_mean((x - tf.stop_gradient(x_means))**2)
    return x_means_hot, x_means, q_loss, e_loss

  def bit_to_int(self, x_bit, num_bits, base=2):
    """Turn x_bit representing numbers bitwise (lower-endian) to int tensor.

    Args:
        x_bit: Tensor containing numbers in a particular base to be
        converted to
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

  def int_to_bit(self, x_int, num_bits, base=2):
    """Turn x_int representing numbers into a bitwise (lower-endian) tensor.

    Args:
        x_int: Tensor containing integer to be converted into base
        notation.
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

  def embed(self, x, scope="bottleneck"):
    """Embedding function that takes discrete latent and returns embedding.

    Args:
        x: Input to the discretization bottleneck.
        scope: Scope name of the function.

    Returns:
        Continuous embedding to be passed on to the decoder.

    Raises:
        ValueError: For unknown or missing arguments.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      shape_x = common_layers.shape_list(x)
      x_flat = tf.reshape(x, [-1, 1])
      c = self.int_to_bit(x_flat, num_bits=self.hparams.z_size, base=2)
      shape = common_layers.shape_list(c)
      new_shape = shape
      new_shape[-1] = self.hparams.num_residuals
      new_shape.append(self.hparams.num_blocks)
      new_shape.append(
          int(self.hparams.z_size /
              (self.hparams.num_residuals * self.hparams.num_blocks)))
      c = tf.to_int32(tf.reshape(c, shape=new_shape))
      h1_shape = shape_x
      h1_shape.append(self.hparams.hidden_size)
      h1 = tf.zeros(dtype=tf.float32, shape=h1_shape)
      for i in range(self.hparams.num_residuals):
        c_residual = self.bit_to_int(
            c[:, :, i, :, :],
            num_bits=int(
                self.hparams.z_size /
                (self.hparams.num_residuals * self.hparams.num_blocks)),
            base=2)
        c_hot = tf.one_hot(c_residual, depth=self.hparams.block_v_size, axis=-1)
        c_hot_flat = tf.reshape(
            c_hot,
            shape=[-1, self.hparams.num_blocks, self.hparams.block_v_size])
        h1_residual = tf.matmul(
            tf.transpose(c_hot_flat, perm=[1, 0, 2]), self.hparams.means[i])
        h1_residual = tf.transpose(h1_residual, perm=[1, 0, 2])
        h1_residual = tf.reshape(h1_residual, shape=h1_shape)
        h1 += h1_residual

      # Add Gaussian noise
      h1_shape[0] = self.hparams.batch_size
      h2 = tf.layers.dense(
          tf.nn.relu(h1), self.hparams.filter_size, name="vch2")
      res = tf.layers.dense(
          tf.nn.relu(h2), self.hparams.hidden_size, name="vcfin")
      return res

  def discrete_bottleneck(self, x, scope="bottleneck"):
    """Discretization bottleneck for latent variables.

    Args:
        x: Input to the discretization bottleneck.
        scope: Scope of the function.

    Returns:
        Embedding to pass to the decoder, discrete latent, loss, and the
        embedding
        function.

    Raises:
        ValueError: If projection_tensors is None for reshape_method
        project, or
        ema_count or ema_means is None if we are using ema, or unknown
        args.
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      x_reshaped = self.slice_hidden(x)
      x_res = x_reshaped
      x_means_hot = []
      x_means = 0
      loss = 0
      for i in range(self.hparams.num_residuals):
        x_means_hot_res, x_means_res, q_loss_res, e_loss_res = \
            self.embedding_lookup(x_reshaped, self.hparams.means[i])

        # Update the ema variables
        if self.hparams.ema:
          tf.logging.info("Using EMA with beta = {}".format(self.hparams.beta))
          updated_ema_count_res = \
              moving_averages.assign_moving_average(
                  self.hparams.ema_count[i],
                  tf.reduce_sum(
                      tf.reshape(
                          x_means_hot_res,
                          shape=[-1, self.hparams.num_blocks,
                                 self.hparams.block_v_size]),
                      axis=0),
                  self.hparams.decay,
                  zero_debias=False)

          dw = tf.matmul(
              tf.transpose(x_means_hot_res, perm=[1, 2, 0]),
              tf.transpose(x_res, perm=[1, 0, 2]))

          updated_ema_means_res = \
              moving_averages.assign_moving_average(
                  self.hparams.ema_means[i], dw, self.hparams.decay,
                  zero_debias=False)
          n = tf.reduce_sum(updated_ema_count_res, axis=-1, keep_dims=True)
          updated_ema_count_res = (
              (updated_ema_count_res + self.hparams.epsilon) /
              (n + 2**self.hparams.z_size * self.hparams.epsilon) * n)
          updated_ema_means_res = updated_ema_means_res/tf.expand_dims(
              updated_ema_count_res, axis=-1)

          with tf.control_dependencies([e_loss_res]):
            update_means_res = tf.assign(self.hparams.means[i],
                                         updated_ema_means_res)
            with tf.control_dependencies([update_means_res]):
              loss += self.hparams.beta * e_loss_res
        else:
          loss += q_loss_res + self.hparams.beta * e_loss_res

        # Update the residuals
        x_res -= x_means_res
        x_means += x_means_res
        x_means_hot.append(x_means_hot_res)

      # Get the discrete latent representation
      x_means_hot = tf.stack(x_means_hot, axis=1)
      x_means_idx = tf.argmax(x_means_hot, axis=-1)

      # Get the binary representation
      num_bits = int(self.hparams.z_size //
                     (self.hparams.num_blocks * self.hparams.num_residuals))
      x_means_bits = self.int_to_bit(x_means_idx, num_bits=num_bits, base=2)
      shape = common_layers.shape_list(x_means_bits)
      new_shape = shape[:-2]
      new_shape[0] = -1
      new_shape[-1] = self.hparams.z_size
      x_means_bits = tf.reshape(x_means_bits, new_shape)
      x_discrete = self.bit_to_int(
          tf.to_int32(x_means_bits), num_bits=self.hparams.z_size, base=2)

      # Reshape x_discrete
      shape_x = common_layers.shape_list(x)
      shape_discrete = shape_x[:-1]
      x_discrete = tf.reshape(x_discrete, shape_discrete)
      x_means = tf.reshape(x_means, shape=shape_x)
      h1 = x + tf.stop_gradient(x_means - x)

      h2 = tf.layers.dense(
          tf.nn.relu(h1), self.hparams.filter_size, name="vch2")
      res = tf.layers.dense(
          tf.nn.relu(h2), self.hparams.hidden_size, name="vcfin")
      embed_fn = partial(self.embed, scope=scope)
      return {
          "dense": res,
          "discrete": x_discrete,
          "loss": loss,
          "embed": embed_fn
      }
