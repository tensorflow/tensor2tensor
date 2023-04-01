# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

"""Additional operations for transformer_glow_layers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models.transformer import transformer_decoder_layer
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def dense(name, x, n_out, dtype=tf.float32, init_w=0.05):
  """Dense layer."""
  n_in = common_layers.shape_list(x)[2]
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    w = tf.get_variable(
        "w", [n_in, n_out], dtype,
        initializer=tf.random_normal_initializer(0.0, init_w), trainable=True)
    b = tf.get_variable(
        "b", [n_out,], dtype, initializer=tf.zeros_initializer, trainable=True)
    x = tf.matmul(x, w) + b
    return x


def dense_weightnorm(
    name, x, n_out, x_mask, init_scale, init, dtype=tf.float32):
  """Dense layer with weight normalization."""
  n_in = common_layers.shape_list(x)[2]
  eps = tf.keras.backend.epsilon()
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    v = tf.get_variable(
        "v", [n_in, n_out], dtype,
        initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
    v = v / tf.norm(v, axis=0, keepdims=True)
    t = tf.matmul(x, v)  # [B, L, n_out]
    mean, var = moments_over_bl(t, x_mask)
    g_init = init_scale / (tf.sqrt(var) + eps)
    g = get_variable_ddi(
        "g", [n_out], g_init, init,
        initializer=tf.zeros_initializer, dtype=dtype, trainable=True)
    b = get_variable_ddi(
        "b", [n_out], -mean*g_init, init,
        initializer=tf.zeros_initializer, dtype=dtype, trainable=True)
    w = g * v
    y = tf.matmul(x, w) + b
    tf.summary.histogram("_g", g)
    return y


def transformer_decoder_block(name,
                              n_layers,
                              x,
                              x_mask,
                              output_size,
                              init,
                              **kwargs):
  """A transformation block composed of transformer decoder layers.

  Args:
    name: variable scope.
    n_layers: number of transformer layers.
    x: input to transformation.
    x_mask: mask.
    output_size: output dimensionality.
    init: data-dependent init for weightnorm parameters.
    **kwargs: Constains hparams, encoder_output,
      encoder_decoder_attention_bias and decoder_self_attention_bias

  Returns:
    outputs: Tensor of shape [batch_size, length, output_size].
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    hparams = kwargs.pop("hparams")
    disable_dropout = kwargs.pop("disable_dropout")
    if disable_dropout:
      hparams = copy.deepcopy(hparams)
      hparams.attention_dropout = 0.0
      hparams.layer_prepostprocess_dropout = 0.0
      hparams.relu_dropout = 0.0
    n_channels = common_layers.shape_list(x)[-1]
    if n_channels != hparams.hidden_size:
      hparams = copy.deepcopy(hparams)
      hparams.hidden_size = n_channels

    outputs = common_attention.add_timing_signal_1d(x)
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
      for layer_idx in range(n_layers):
        outputs = transformer_decoder_layer(
            decoder_input=outputs,
            layer_idx=layer_idx,
            hparams=hparams,
            **kwargs)
    outputs = common_layers.layer_preprocess(outputs, hparams)
    outputs = dense_weightnorm(
        "h2o", outputs, output_size, x_mask, init_scale=0.0, init=init)
    return outputs


def reduce_sum_over_lc(x, x_mask):
  """Returns sum of x (over L and C) given the actual length and pad.

  Args:
    x: input. (B,L,C)
    x_mask: binary padding mask. (B,L)

  Returns:
    sum of x. (B)
  """

  if x.shape.rank == 3 and x_mask.shape.rank == 2:
    x_mask = x_mask[..., tf.newaxis]
  else:
    tf.logging.info("x: {}, x_mask: {}".format(x.shape.rank, x_mask.shape.rank))
    raise ValueError("Dimension not supported.")

  mean = x * x_mask
  return tf.reduce_sum(mean, axis=[1, 2])  # sum over L, C


def reduce_sum_over_l(x, x_mask):
  """Returns sum of x (over L) given the actual length and pad.

  Args:
    x: input. (B,L,C)
    x_mask: binary padding mask. (B,L)

  Returns:
    sum of x. (B,C)
  """

  if x.shape.rank == 3 and x_mask.shape.rank == 2:
    x_mask = x_mask[..., tf.newaxis]
  else:
    tf.logging.info("x: {}, x_mask: {}".format(x.shape.rank, x_mask.shape.rank))
    raise ValueError("Dimension not supported.")

  mean = x * x_mask
  return tf.reduce_sum(mean, axis=-2)  # sum over L


def reduce_mean_over_l(x, x_mask):
  """Returns mean of x (over L) given the actual length and pad."""
  return reduce_sum_over_l(x, x_mask) / tf.reduce_sum(x_mask, 1, keepdims=True)


def reduce_mean_over_bl(x, x_mask):
  """Returns average of x (over B and L) given the actual length and pad.

  Args:
    x: input. (B,L,C)
    x_mask: binary padding mask. (B,L)

  Returns:
    mean of x. (C)
  """

  if x.shape.rank == 3 and x_mask.shape.rank == 2:
    x_mask = x_mask[..., tf.newaxis]
  else:
    tf.logging.info("x: {}, x_mask: {}".format(x.shape.rank, x_mask.shape.rank))
    raise ValueError("Dimension not supported.")

  mean = x * x_mask
  mean = tf.reduce_sum(mean, axis=[0, 1])  # sum over B, L
  return mean / tf.reduce_sum(x_mask)


def reduce_mean_over_l_sum_over_c(x, x_mask):
  """Returns mean of x over L and sum over C."""
  mean = reduce_sum_over_lc(x, x_mask)
  return mean / tf.reduce_sum(x_mask, 1)


def reduce_mean_over_bl_sum_over_c(x, x_mask):
  """Returns mean of x over B and L and sum over C."""
  mean = reduce_mean_over_bl(x, x_mask)
  return tf.reduce_sum(mean)


def moments_over_bl(x, x_mask):
  """Returns mean and var of x over B and L."""
  mean = reduce_mean_over_bl(x, x_mask)
  var = reduce_mean_over_bl((x-mean)**2, x_mask)
  return mean, var


def standard_normal_density(x, x_mask, reduce_sum=False):
  """Return standard normal distribution with same shape as x."""
  log_probs = -0.5 * (x**2 + math.log(math.pi * 2.0))
  if reduce_sum:
    log_probs = reduce_mean_over_bl_sum_over_c(log_probs, x_mask)
  else:
    log_probs = reduce_sum_over_lc(log_probs, x_mask)
  return log_probs


def standard_normal(x, name="normal"):
  """Return standard normal distribution with same shape as x."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    dist = tfp.distributions.Normal(
        loc=tf.zeros_like(x),
        scale=tf.ones_like(x),
        allow_nan_stats=False)
    return dist


def diagonal_normal(outputs, name="normal"):
  """Split outputs into mu and log_sigma and return z."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    loc, log_scale = tf.split(outputs, 2, axis=-1)
    scale = tf.exp(log_scale)
    dist = tfp.distributions.Normal(
        loc=loc,
        scale=scale + tf.keras.backend.epsilon(),
        allow_nan_stats=False)
    return dist


def split_coupling(
    x, x_mask, split_dim, identity_first, decoder_self_attention_bias):
  """Split function used in coupling flows."""
  n_channels = common_layers.shape_list(x)[-1]
  if split_dim == "c":
    n_transform = n_identity = n_channels // 2
    x_id = x[..., :n_identity] if identity_first else x[..., n_transform:]
    x_tr = x[..., n_identity:] if identity_first else x[..., :n_transform]
    bias, mask = decoder_self_attention_bias, x_mask

  elif split_dim == "a":
    n_transform = n_identity = n_channels // 2
    x_id = x[..., 0::2] if identity_first else x[..., 1::2]
    x_tr = x[..., 1::2] if identity_first else x[..., 0::2]
    bias, mask = decoder_self_attention_bias, x_mask

  elif split_dim == "t":
    n_transform = n_identity = n_channels
    x_id = x[:, 0::2, :] if identity_first else x[:, 1::2, :]
    x_tr = x[:, 1::2, :] if identity_first else x[:, 0::2, :]
    bias, mask = decoder_self_attention_bias[..., 0::2], x_mask[..., 0::2]

  return x_id, x_tr, n_identity, n_transform, bias, mask


def join_coupling(z_id, z_tr, split_dim, identity_first):
  """Reverse split function used in coupling flows."""
  assert z_id.shape.rank == 3 and z_tr.shape.rank == 3
  result = [z_id, z_tr] if identity_first else [z_tr, z_id]
  if split_dim == "c":
    result = tf.concat(result, axis=2)  # concat in the channel dimension
  elif split_dim == "a":
    result = tf.stack(result, axis=3)  # stack in the channel dimension
  elif split_dim == "t":
    result = tf.stack(result, axis=2)  # stack in the time dimension
  return result


def assign(w, initial_value):
  w = w.assign(initial_value)
  with tf.control_dependencies([w]):
    return w


def get_variable_ddi(
    name, shape, value, init, initializer=None, dtype=tf.float32,
    regularizer=None, trainable=True):
  """Wrapper for data-dependent initialization."""
  kwargs = {"trainable": trainable}
  if initializer:
    kwargs["initializer"] = initializer
  if regularizer:
    kwargs["regularizer"] = regularizer
  w = tf.get_variable(name, shape, dtype, **kwargs)
  if isinstance(init, bool):
    if init:
      return assign(w, value)
    return w
  else:
    return tf.cond(init, lambda: assign(w, value), lambda: w)
