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
    print ("self.hparams.z_size", self.hparams.z_size)
    # Set the discretization bottleneck specific things here
    self.hparams.z_size_per_residual = self.hparams.z_size // \
                                       self.hparams.num_residuals
    print ("self.hparams.num_residuals", self.hparams.num_residuals)
    self.hparams.block_dim = int(
        self.hparams.hidden_size // self.hparams.num_blocks)
    self.hparams.block_v_size = 2**(
        self.hparams.z_size_per_residual / self.hparams.num_blocks)
    self.hparams.block_v_size = int(self.hparams.block_v_size)
    self.means = tf.get_variable(
        name="means",
        shape=[
            self.hparams.num_blocks, self.hparams.block_v_size,
            self.hparams.block_dim
        ],
        initializer=tf.initializers.variance_scaling(distribution="uniform"))

    # Create the shadow variables if we are using EMA
    if self.hparams.ema:
      self.ema_count = tf.get_variable(
          "ema_count", [self.hparams.num_blocks, self.hparams.block_v_size],
          initializer=tf.constant_initializer(0),
          trainable=False)
      with tf.colocate_with(self.means):
        self.ema_means = tf.get_variable(
            "ema_means",
            initializer=self.means.initialized_value(),
            trainable=False)

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
    q_loss = tf.reduce_mean(
        tf.squared_difference(tf.stop_gradient(x), x_means))
    e_loss = tf.reduce_mean(
        tf.squared_difference(x, tf.stop_gradient(x_means)))
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
    # pylint: disable=g-complex-comprehension
    x_labels = [
        x_l[:, i] * tf.to_int32(base)**tf.to_int32(i) for i in range(num_bits)]
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
    # pylint: disable=g-complex-comprehension
    x_labels = [
        tf.floormod(
            tf.floordiv(tf.to_int32(x_l),
                        tf.to_int32(base)**i), tf.to_int32(base))
        for i in range(num_bits)]
    res = tf.concat(x_labels, axis=-1)
    return tf.to_float(res)

  def embed(self, x):
    """Embedding function that takes discrete latent and returns embedding.

    Args:
        x: Input to the discretization bottleneck.
    Returns:
        Continuous embedding to be passed on to the decoder.

    Raises:
        ValueError: For unknown or missing arguments.
    """
    shape_x = common_layers.shape_list(x)
    x_flat = tf.reshape(x, [-1, 1])
    c = self.int_to_bit(x_flat, num_bits=self.hparams.z_size, base=2)
    shape = common_layers.shape_list(c)
    new_shape = shape
    new_shape.append(self.hparams.num_blocks)
    new_shape.append(int(self.hparams.z_size / self.hparams.num_blocks))
    c = tf.to_int32(tf.reshape(c, shape=new_shape))
    h1_shape = shape_x
    h1_shape.append(self.hparams.hidden_size)
    h1 = tf.zeros(dtype=tf.float32, shape=h1_shape)
    c_int = self.bit_to_int(
        c, num_bits=int(self.hparams.z_size / self.hparams.num_blocks), base=2)
    c_hot = tf.one_hot(c_int, depth=self.hparams.block_v_size, axis=-1)
    c_hot_flat = tf.reshape(
        c_hot, shape=[-1, self.hparams.num_blocks, self.hparams.block_v_size])
    h1 = tf.matmul(tf.transpose(c_hot_flat, perm=[1, 0, 2]), self.means)
    h1 = tf.transpose(h1, perm=[1, 0, 2])
    h1 = tf.reshape(h1, shape=h1_shape)
    h1_shape[0] = self.hparams.batch_size
    h2 = tf.layers.dense(tf.nn.relu(h1), self.hparams.filter_size, name="vch2")
    res = tf.layers.dense(
        tf.nn.relu(h2), self.hparams.hidden_size, name="vcfin")
    return res

  def discrete_bottleneck(self, x):
    """Discretization bottleneck for latent variables.

    Args:
        x: Input to the discretization bottleneck.

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
    x_reshaped = self.slice_hidden(x)
    x_means_hot = []
    x_means = 0
    loss = 0
    x_means_hot, x_means, q_loss, e_loss = self.embedding_lookup(
        x_reshaped, self.means)

    if self.hparams.ema:
      tf.logging.info("Using EMA with beta = {}".format(self.hparams.beta))
      updated_ema_count = \
          moving_averages.assign_moving_average(
              self.ema_count,
              tf.reduce_sum(
                  tf.reshape(
                      x_means_hot,
                      shape=[-1, self.hparams.num_blocks,
                             self.hparams.block_v_size]),
                  axis=0),
              self.hparams.decay,
              zero_debias=False)

      dw = tf.matmul(
          tf.transpose(x_means_hot, perm=[1, 2, 0]),
          tf.transpose(x_reshaped, perm=[1, 0, 2]))

      updated_ema_means = \
          moving_averages.assign_moving_average(
              self.ema_means, dw, self.hparams.decay,
              zero_debias=False)
      n = tf.reduce_sum(updated_ema_count, axis=-1, keep_dims=True)
      updated_ema_count = ((updated_ema_count + self.hparams.epsilon) / (
          n + 2**self.hparams.z_size * self.hparams.epsilon) * n)
      updated_ema_means = updated_ema_means / tf.expand_dims(
          updated_ema_count, axis=-1)

      with tf.control_dependencies([e_loss]):
        update_means = tf.assign(self.means, updated_ema_means)
        with tf.control_dependencies([update_means]):
          loss += self.hparams.beta * e_loss
    else:
      # Use a gradient based loss for learning the cluster centers
      loss += q_loss + self.hparams.beta * e_loss

    # Get the discrete latent representation
    x_means_idx = tf.argmax(x_means_hot, axis=-1)

    # Get the binary representation
    num_bits = int(self.hparams.z_size // self.hparams.num_blocks)
    x_means_bits = self.int_to_bit(x_means_idx, num_bits=num_bits, base=2)
    x_discrete = self.bit_to_int(
        tf.to_int32(x_means_bits), num_bits=self.hparams.z_size, base=2)

    # Reshape x_discrete
    shape_x = common_layers.shape_list(x)
    shape_discrete = shape_x[:-1]
    x_discrete = tf.reshape(x_discrete, shape_discrete)
    x_means = tf.reshape(x_means, shape=shape_x)
    h1 = x + tf.stop_gradient(x_means - x)

    h2 = tf.layers.dense(tf.nn.relu(h1), self.hparams.filter_size, name="vch2")
    res = tf.layers.dense(
        tf.nn.relu(h2), self.hparams.hidden_size, name="vcfin")
    embed_fn = partial(self.embed)
    return {
        "dense": res,
        "discrete": x_discrete,
        "loss": loss,
        "embed": embed_fn
    }
