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

"""Residual Shuffle-Exchange Network.

Implementation of
"Residual Shuffle-Exchange Networks for Fast Processing of Long Sequences"
paper by A.Draguns, E.Ozolins, A.Sostaks, M.Apinis, K.Freivalds.

Paper: https://arxiv.org/abs/2004.04662
Original code: https://github.com/LUMII-Syslab/RSE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.layers.common_layers import gelu
from tensor2tensor.models.research.shuffle_network import reverse_shuffle_layer
from tensor2tensor.models.research.shuffle_network import shuffle_layer
from tensor2tensor.models.research.shuffle_network import ShuffleNetwork
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator


class LayerNormalization(tf.keras.layers.Layer):
  """Layer Normalization (LayerNorm) without output bias and gain."""

  def __init__(self, axis=1, epsilon=1e-10, **kwargs):
    """Initialize Layer Normalization layer.

    Args:
      axis: Tuple or number of axis for calculating mean and variance
      epsilon: Small epsilon to avoid division by zero
      **kwargs: keyword args passed to super.
    """
    self.axis = axis
    self.epsilon = epsilon
    self.bias = None
    super(LayerNormalization, self).__init__(**kwargs)

  def build(self, input_shape):
    """Initialize bias weights for layer normalization.

    Args:
      input_shape: shape of input tensor
    """
    num_units = input_shape.as_list()[-1]
    self.bias = self.add_weight(
        "bias", [1, 1, num_units], initializer=tf.zeros_initializer)
    super(LayerNormalization, self).build(input_shape)

  def call(self, inputs, **kwargs):
    """Apply Layer Normalization without output bias and gain.

    Args:
      inputs: tensor to be normalized. Axis should be smaller than input tensor
        dimensions.
      **kwargs: more arguments (unused)

    Returns:
      tensor output.
    """
    inputs -= tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
    inputs += self.bias
    variance = tf.reduce_mean(tf.square(inputs), self.axis, keepdims=True)
    return inputs * tf.math.rsqrt(variance + self.epsilon)


def inv_sigmoid(y):
  """Inverse sigmoid function.

  Args:
    y: float in range 0 to 1

  Returns:
    the inverse sigmoid.
  """
  return np.log(y / (1 - y))


class RSU(tf.keras.layers.Layer):
  """Residual Switch Unit of Residual Shuffle-Exchange network."""

  def __init__(self, prefix, dropout, mode, **kwargs):
    """Initialize Switch Layer.

    Args:
      prefix: Name prefix for switch layer
      dropout: Dropout rate
      mode: Training mode
      **kwargs: more arguments (unused)
    """
    super().__init__(**kwargs)
    self.prefix = prefix
    self.dropout = dropout
    self.mode = mode
    self.first_linear = None
    self.second_linear = None
    self.layer_norm = None
    self.residual_scale = None

    residual_weight = 0.9
    self.candidate_weight = np.sqrt(1 - residual_weight**2) * 0.25
    self.init_value = inv_sigmoid(residual_weight)

  def build(self, input_shape):
    """Initialize layer weights and sublayers.

    Args:
      input_shape: shape of inputs
    """
    in_units = input_shape[-1]
    middle_units = in_units * 4
    out_units = in_units * 2
    init = tf.variance_scaling_initializer(
        scale=1.0, mode="fan_avg", distribution="uniform")

    self.first_linear = tf.keras.layers.Dense(
        middle_units,
        use_bias=False,
        kernel_initializer=init,
        name=self.prefix + "/cand1")

    self.second_linear = tf.keras.layers.Dense(
        out_units, kernel_initializer=init, name=self.prefix + "/cand2")
    self.layer_norm = LayerNormalization()

    init = tf.constant_initializer(self.init_value)
    self.residual_scale = self.add_weight(
        self.prefix + "/residual", [out_units], initializer=init)
    super(RSU, self).build(input_shape)

  def call(self, inputs, **kwargs):
    """Apply Residual Switch Layer to inputs.

    Args:
      inputs: Input tensor.
      **kwargs: unused kwargs.

    Returns:
      tf.Tensor: New candidate value
    """
    del kwargs
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    length = input_shape[1]
    num_units = inputs.shape.as_list()[2]

    n_bits = tf.log(tf.cast(length - 1, tf.float32)) / tf.log(2.0)
    n_bits = tf.floor(n_bits) + 1

    reshape_shape = [batch_size, length // 2, num_units * 2]
    reshaped_inputs = tf.reshape(inputs, reshape_shape)

    first_linear = self.first_linear(reshaped_inputs)
    first_linear = self.layer_norm(first_linear)
    first_linear = gelu(first_linear)
    candidate = self.second_linear(first_linear)

    residual = tf.sigmoid(self.residual_scale) * reshaped_inputs
    candidate = residual + candidate * self.candidate_weight
    candidate = tf.reshape(candidate, input_shape)

    if self.dropout > 0:
      candidate = tf.nn.dropout(candidate, rate=self.dropout / n_bits)
    if self.dropout != 0.0 and self.mode == tf_estimator.ModeKeys.TRAIN:
      noise = tf.random_normal(tf.shape(candidate), mean=1.0, stddev=0.001)
      candidate = candidate * noise

    return candidate


def residual_shuffle_network(inputs, hparams):
  """Residual Shuffle-Exchange network with weight sharing.

  Args:
    inputs: inputs to the Shuffle-Exchange network. Should be in length of power
      of 2.
    hparams: Model configuration

  Returns:
    tf.Tensor: Outputs of the Shuffle-Exchange last layer
  """
  input_shape = tf.shape(inputs)
  n_bits = tf.log(tf.cast(input_shape[1] - 1, tf.float32)) / tf.log(2.0)
  n_bits = tf.cast(n_bits, tf.int32) + 1

  block_out = inputs

  for k in range(hparams.num_hidden_layers):
    with tf.variable_scope("benes_block_" + str(k), reuse=tf.AUTO_REUSE):
      forward_output = forward_part(block_out, hparams, n_bits)
      block_out = reverse_part(forward_output, hparams, n_bits)

  return RSU("last_layer", hparams.dropout, hparams.mode)(block_out)


def reverse_part(inputs, hparams, n_bits):
  """Reverse part of Benes block.

  Repeatably applies interleaved Residual Switch layer and Reverse Shuffle
  Layer. One set of weights used for all Switch layers.

  Args:
    inputs: inputs for reverse part. Should be outputs from forward part.
    hparams: params of the network.
    n_bits: count of repeated layer applications.

  Returns:
    tf.Tensor: output of reverse part.
  """
  reverse_rsu = RSU("reverse_switch", hparams.dropout, hparams.mode)

  def reverse_step(state, _):
    with tf.variable_scope("reverse"):
      new_state = reverse_rsu(state)
      return reverse_shuffle_layer(new_state)

  reverse_outputs = tf.scan(
      reverse_step,
      tf.range(n_bits, n_bits * 2),
      initializer=inputs,
      parallel_iterations=1,
      swap_memory=True)

  return reverse_outputs[-1, :, :, :]


def forward_part(block_out, hparams, n_bits):
  """Forward part of Benes block.

  Repeatably applies interleaved Residual Switch layer and Shuffle
  Layer. One set of weights used for all Switch layers.

  Args:
    block_out: TODO(authors) document.
    hparams: params of the network.
    n_bits: count of repeated layer applications.

  Returns:
    tf.Tensor: output of forward part.
  """
  forward_rsu = RSU("switch", hparams.dropout, hparams.mode)

  def forward_step(state, _):
    with tf.variable_scope("forward"):
      new_state = forward_rsu(state)
      return shuffle_layer(new_state)

  forward_outputs = tf.scan(
      forward_step,
      tf.range(0, n_bits),
      initializer=block_out,
      parallel_iterations=1,
      swap_memory=True)

  return forward_outputs[-1, :, :, :]


@registry.register_model
class ResidualShuffleExchange(ShuffleNetwork):
  """T2T implementation of Residual Shuffle-Exchange network."""

  def body(self, features):
    """Body of Residual Shuffle-Exchange network.

    Args:
      features: dictionary of inputs and targets

    Returns:
      the network output.
    """

    inputs = tf.squeeze(features["inputs"], axis=2)
    logits = residual_shuffle_network(inputs, self._hparams)
    return tf.expand_dims(logits, axis=2)
