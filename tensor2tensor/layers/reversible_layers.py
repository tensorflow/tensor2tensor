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

"""Reversible layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Reverse(tf.keras.layers.Layer):
  """Swaps the forward and reverse transformations of a layer."""

  def __init__(self, reversible_layer, **kwargs):
    super(Reverse, self).__init__(**kwargs)
    if not hasattr(reversible_layer, 'reverse'):
      raise ValueError('Layer passed-in has not implemented "reverse" method: '
                       '{}'.format(reversible_layer))
    self.call = reversible_layer.reverse
    self.reverse = reversible_layer.call


class MADE(tf.keras.Model):
  """Masked autoencoder for distribution estimation (Germain et al., 2015).

  MADE takes as input a real Tensor of shape [..., length] and returns a
  Tensor of shape [..., num_heads * length] and same dtype. It masks layer
  weights to respect autoregressive constraints: for a given ordering, each
  input dimension can be reconstructed from previous input dimensions. The
  output dimensions represent multiple heads for, e.g., location and scale
  transforms in a flow.
  """

  def __init__(self,
               hidden_dims,
               num_heads=2,
               input_order='left-to-right',
               hidden_order='left-to-right',
               activation=None,
               use_bias=True,
               **kwargs):
    """Constructs network.

    Args:
      hidden_dims: list with the number of hidden units per layer. It does not
        include the output layer; those number of units will always be set to
        the input dimension multiplied by `num_heads`.
      num_heads: The number of output heads. The default is 2 representing
        the location and scale transform of an autoregressive flow.
      input_order: Order of degrees to the input units: 'random',
        'left-to-right', 'right-to-left', or an array of an explicit order.
        For example, 'left-to-right' builds an autoregressive model
        p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
      hidden_order: Order of degrees to the hidden units: 'random',
        'left-to-right'.
      activation: Activation function.
      use_bias: Whether to use a bias.
      **kwargs: Keyword arguments of parent class.
    """
    super(MADE, self).__init__(**kwargs)
    self.hidden_dims = hidden_dims
    self.num_heads = num_heads
    self.input_order = input_order
    self.hidden_order = hidden_order
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.network = tf.keras.Sequential([])

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = input_shape[-1]
    if isinstance(last_dim, tf.Dimension):
      last_dim = last_dim.value
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to '
                       '`MADE` should be defined. Found `None`.')
    masks = create_masks(input_dim=last_dim,
                         hidden_dims=self.hidden_dims,
                         input_order=self.input_order,
                         hidden_order=self.hidden_order)
    for l in range(len(self.hidden_dims)):
      layer = tf.keras.layers.Dense(
          self.hidden_dims[l],
          kernel_initializer=make_masked_initializer(masks[l]),
          kernel_constraint=make_masked_constraint(masks[l]),
          activation=self.activation,
          use_bias=self.use_bias)
      self.network.add(layer)

    mask = tf.tile(masks[-1], [1, self.num_heads])
    layer = tf.keras.layers.Dense(
        last_dim * self.num_heads,
        kernel_initializer=make_masked_initializer(mask),
        kernel_constraint=make_masked_constraint(mask),
        activation=None,
        use_bias=self.use_bias)
    self.network.add(layer)
    self.built = True

  def call(self, inputs):
    return self.network(inputs)


def create_degrees(input_dim,
                   hidden_dims,
                   input_order='left-to-right',
                   hidden_order='left-to-right'):
  """Returns a list of degree vectors, one for each input and hidden layer.

  A unit with degree d can only receive input from units with degree < d. Output
  units always have the same degree as their associated input unit.

  Args:
    input_dim: Number of inputs.
    hidden_dims: list with the number of hidden units per layer. It does not
      include the output layer; those number of units will always be set to
      input_dim downstream.
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_order: Order of degrees to the hidden units: 'random',
      'left-to-right'.
  """
  if (isinstance(input_order, str) and
      input_order not in ('random', 'left-to-right', 'right-to-left')):
    raise ValueError('Input order is not valid.')
  if hidden_order not in ('random', 'left-to-right'):
    raise ValueError('Hidden order is not valid.')

  degrees = []
  if isinstance(input_order, str):
    input_degrees = np.arange(1, input_dim + 1)
    if input_order == 'right-to-left':
      input_degrees = np.flip(input_degrees, 0)
    elif input_order == 'random':
      np.random.shuffle(input_degrees)
  else:
    input_order = np.array(input_order)
    if np.all(np.sort(input_order) != np.arange(1, input_dim + 1)):
      raise ValueError('invalid input order')
    input_degrees = input_order
  degrees.append(input_degrees)

  for units in hidden_dims:
    if hidden_order == 'random':
      min_prev_degree = min(np.min(degrees[-1]), input_dim - 1)
      hidden_degrees = np.random.randint(
          low=min_prev_degree, high=input_dim, size=units)
    elif hidden_order == 'left-to-right':
      hidden_degrees = (np.arange(units) % max(1, input_dim - 1) +
                        min(1, input_dim - 1))
    degrees.append(hidden_degrees)
  return degrees


def create_masks(input_dim,
                 hidden_dims,
                 input_order='left-to-right',
                 hidden_order='left-to-right'):
  """Returns a list of binary mask matrices respecting autoregressive ordering.

  Args:
    input_dim: Number of inputs.
    hidden_dims: list with the number of hidden units per layer. It does not
      include the output layer; those number of units will always be set to
      input_dim downstream.
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_order: Order of degrees to the hidden units: 'random',
      'left-to-right'.
  """
  degrees = create_degrees(input_dim, hidden_dims, input_order, hidden_order)
  masks = []
  # Create input-to-hidden and hidden-to-hidden masks.
  for input_degrees, output_degrees in zip(degrees[:-1], degrees[1:]):
    mask = tf.cast(input_degrees[:, np.newaxis] <= output_degrees, tf.float32)
    masks.append(mask)

  # Create hidden-to-output mask.
  mask = tf.cast(degrees[-1][:, np.newaxis] < degrees[0], tf.float32)
  masks.append(mask)
  return masks


def make_masked_initializer(mask):
  initializer = tf.keras.initializers.glorot_uniform()
  def masked_initializer(shape, dtype=None, partition_info=None):
    return mask * initializer(shape, dtype, partition_info)
  return masked_initializer


def make_masked_constraint(mask):
  constraint = tf.identity
  def masked_constraint(x):
    return mask * constraint(x)
  return masked_constraint
