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

"""Reversible layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import edward2 as ed


class Reverse(tf.keras.layers.Layer):
  """Swaps the forward and reverse transformations of a layer."""

  def __init__(self, reversible_layer, **kwargs):
    super(Reverse, self).__init__(**kwargs)
    if not hasattr(reversible_layer, 'reverse'):
      raise ValueError('Layer passed-in has not implemented "reverse" method: '
                       '{}'.format(reversible_layer))
    self.call = reversible_layer.reverse
    self.reverse = reversible_layer.call


class ActNorm(tf.keras.layers.Layer):
  """Actnorm, an affine reversible layer (Prafulla and Kingma, 2018).

  Weights use data-dependent initialization in which outputs have zero mean
  and unit variance per channel (last dimension). The mean/variance statistics
  are computed from the first batch of inputs.
  """

  def __init__(self, epsilon=tf.keras.backend.epsilon(), **kwargs):
    super(ActNorm, self).__init__(**kwargs)
    self.epsilon = epsilon

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = input_shape[-1]
    if isinstance(last_dim, tf.Dimension):
      last_dim = last_dim.value
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `ActNorm` '
                       'should be defined. Found `None`.')
    bias = self.add_weight('bias', [last_dim], dtype=self.dtype)
    log_scale = self.add_weight('log_scale', [last_dim], dtype=self.dtype)
    # Set data-dependent initializers.
    bias = bias.assign(self.bias_initial_value)
    with tf.control_dependencies([bias]):
      self.bias = bias
    log_scale = log_scale.assign(self.log_scale_initial_value)
    with tf.control_dependencies([log_scale]):
      self.log_scale = log_scale
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not self.built:
      mean, variance = tf.nn.moments(
          inputs, axes=[i for i in range(inputs.shape.ndims - 1)])
      self.bias_initial_value = -mean
      # TODO(trandustin): Optionally, actnorm multiplies log_scale by a fixed
      # log_scale factor (e.g., 3.) and initializes by
      # initial_value / log_scale_factor.
      self.log_scale_initial_value = tf.log(
          1. / (tf.sqrt(variance) + self.epsilon))

    if not isinstance(inputs, ed.RandomVariable):
      return super(ActNorm, self).__call__(inputs, *args, **kwargs)
    return TransformedRandomVariable(inputs, self)

  def call(self, inputs):
    return (inputs + self.bias) * tf.exp(self.log_scale)

  def reverse(self, inputs):
    return inputs * tf.exp(-self.log_scale) - self.bias

  def log_det_jacobian(self, inputs):
    """Returns log det | dx / dy | = num_events * sum log | scale |."""
    del inputs  # unused
    # Number of events is number of all elements excluding the batch and
    # channel dimensions.
    num_events = tf.reduce_prod(tf.shape(inputs)[1:-1])
    log_det_jacobian = num_events * tf.reduce_sum(self.log_scale)
    return log_det_jacobian


class MADE(tf.keras.Model):
  """Masked autoencoder for distribution estimation (Germain et al., 2015).

  MADE takes as input a real Tensor of shape [..., length, channels] and returns
  a Tensor of shape [..., length, units] and same dtype. It masks layer weights
  to satisfy autoregressive constraints with respect to the length dimension. In
  particular, for a given ordering, each input dimension of length can be
  reconstructed from previous dimensions.

  The output's units dimension captures per-time-step representations. For
  example, setting units to 2 can parameterize the location and log-scale of an
  autoregressive Gaussian distribution.
  """

  def __init__(self,
               units,
               hidden_dims,
               input_order='left-to-right',
               hidden_order='left-to-right',
               activation=None,
               use_bias=True,
               **kwargs):
    """Constructs network.

    Args:
      units: Positive integer, dimensionality of the output space.
      hidden_dims: list with the number of hidden units per layer. It does not
        include the output layer; those number of units will always be set to
        the input dimension multiplied by `num_heads`. Each hidden unit size
        must be at least the size of length (otherwise autoregressivity is not
        possible).
      input_order: Order of degrees to the input units: 'random',
        'left-to-right', 'right-to-left', or an array of an explicit order.
        For example, 'left-to-right' builds an autoregressive model
        p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
      hidden_order: Order of degrees to the hidden units: 'random',
        'left-to-right'. If 'left-to-right', hidden units are allocated equally
        (up to a remainder term) to each degree.
      activation: Activation function.
      use_bias: Whether to use a bias.
      **kwargs: Keyword arguments of parent class.
    """
    super(MADE, self).__init__(**kwargs)
    self.units = int(units)
    self.hidden_dims = hidden_dims
    self.input_order = input_order
    self.hidden_order = hidden_order
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.network = tf.keras.Sequential([])

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    length = input_shape[-2]
    channels = input_shape[-1]
    if isinstance(length, tf.Dimension):
      length = length.value
    if isinstance(channels, tf.Dimension):
      channels = channels.value
    if length is None or channels is None:
      raise ValueError('The two last dimensions of the inputs to '
                       '`MADE` should be defined. Found `None`.')
    masks = create_masks(input_dim=length,
                         hidden_dims=self.hidden_dims,
                         input_order=self.input_order,
                         hidden_order=self.hidden_order)

    # Input-to-hidden layer: [..., length, channels] -> [..., hidden_dims[0]].
    self.network.add(tf.keras.layers.Reshape([length * channels]))
    # Tile the mask so each element repeats contiguously; this is compatible
    # with the autoregressive contraints unlike naive tiling.
    mask = masks[0]
    mask = tf.tile(mask[:, tf.newaxis, :], [1, channels, 1])
    mask = tf.reshape(mask, [mask.shape[0] * channels, mask.shape[-1]])
    if self.hidden_dims:
      layer = tf.keras.layers.Dense(
          self.hidden_dims[0],
          kernel_initializer=make_masked_initializer(mask),
          kernel_constraint=make_masked_constraint(mask),
          activation=self.activation,
          use_bias=self.use_bias)
      self.network.add(layer)

    # Hidden-to-hidden layers: [..., hidden_dims[l-1]] -> [..., hidden_dims[l]].
    for l in range(1, len(self.hidden_dims)):
      layer = tf.keras.layers.Dense(
          self.hidden_dims[l],
          kernel_initializer=make_masked_initializer(masks[l]),
          kernel_constraint=make_masked_constraint(masks[l]),
          activation=self.activation,
          use_bias=self.use_bias)
      self.network.add(layer)

    # Hidden-to-output layer: [..., hidden_dims[-1]] -> [..., length, units].
    # Tile the mask so each element repeats contiguously; this is compatible
    # with the autoregressive contraints unlike naive tiling.
    if self.hidden_dims:
      mask = masks[-1]
    mask = tf.tile(mask[..., tf.newaxis], [1, 1, self.units])
    mask = tf.reshape(mask, [mask.shape[0], mask.shape[1] * self.units])
    layer = tf.keras.layers.Dense(
        length * self.units,
        kernel_initializer=make_masked_initializer(mask),
        kernel_constraint=make_masked_constraint(mask),
        activation=None,
        use_bias=self.use_bias)
    self.network.add(layer)
    self.network.add(tf.keras.layers.Reshape([length, self.units]))
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
      include the output layer. Each hidden unit size must be at least the size
      of length (otherwise autoregressivity is not possible).
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_order: Order of degrees to the hidden units: 'random',
      'left-to-right'. If 'left-to-right', hidden units are allocated equally
      (up to a remainder term) to each degree.
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
      input_dim downstream. Each hidden unit size must be at least the size of
      length (otherwise autoregressivity is not possible).
    input_order: Order of degrees to the input units: 'random', 'left-to-right',
      'right-to-left', or an array of an explicit order. For example,
      'left-to-right' builds an autoregressive model
      p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
    hidden_order: Order of degrees to the hidden units: 'random',
      'left-to-right'. If 'left-to-right', hidden units are allocated equally
      (up to a remainder term) to each degree.
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


def sinkhorn(inputs, n_iters=20):
  """Performs incomplete Sinkhorn normalization to inputs.

  By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
  with positive entries can be turned into a doubly-stochastic matrix
  (i.e. its rows and columns add up to one) via the succesive row and column
  normalization.
  -To ensure positivity, the effective input to sinkhorn has to be
  exp(inputs) (elementwise).
  -However, for stability, sinkhorn works in the log-space. It is only at
   return time that entries are exponentiated.

  Code is adapted from Mena et al. [2].

  [1] Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and
  doubly stochastic matrices. Pacific Journal of Mathematics, 1967.

  [2] Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek.
  Learning latent permutations with Gumbel-Sinkhorn networks. International
  Conference on Learning Representations, 2018.

  Args:
    inputs: A `Tensor` with shape `[..., vocab_size, vocab_size]`.
    n_iters: Number of sinkhorn iterations (in practice, as little as 20
      iterations are needed to achieve decent convergence for `vocab_size` ~100)

  Returns:
    outputs: A `Tensor` of close-to-doubly-stochastic matrices with shape
      `[:, vocab_size, vocab_size]`.
  """
  vocab_size = tf.shape(inputs)[-1]
  log_alpha = tf.reshape(inputs, [-1, vocab_size, vocab_size])

  for _ in range(n_iters):
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=2),
                            [-1, vocab_size, 1])
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=1),
                            [-1, 1, vocab_size])
  outputs = tf.exp(log_alpha)
  return outputs


class TransformedDistribution(tfp.distributions.Distribution):
  """Distribution of f(x), where x ~ p(x) and f is reversible."""

  def __init__(self, base, reversible_layer, name=None):
    """Constructs a transformed distribution.

    Args:
      base: Base distribution.
      reversible_layer: Callable with methods `reverse` and `log_det_jacobian`.
      name: Name for scoping operations in the class.
    """
    self.base = base
    self.reversible_layer = reversible_layer
    if name is None:
      name = reversible_layer.name + base.name
    super(TransformedDistribution, self).__init__(
        base.dtype,
        base.reparameterization_type,
        base.validate_args,
        base.allow_nan_stats,
        parameters=dict(locals()),
        name=name)

  def _event_shape_tensor(self):
    return self.base.event_shape_tensor()

  def _event_shape(self):
    return self.base.event_shape

  def _batch_shape_tensor(self):
    return self.base.batch_shape_tensor()

  def _batch_shape(self):
    return self.base.batch_shape

  def __getitem__(self, slices):
    overrides = {'base': self.base[slices]}
    return self.copy(**overrides)

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    x = self.base.sample(sample_shape, seed, **kwargs)
    y = self.reversible_layer(x)
    return y

  def _log_prob(self, value):
    x = self.reversible_layer.reverse(value)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(value)
    return self.base.log_prob(x) + log_det_jacobian

  def _prob(self, value):
    if not hasattr(self.base, '_prob'):
      return tf.exp(self.log_prob(value))
    x = self.reversible_layer.reverse(value)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(value)
    return self.base.prob(x) * tf.exp(log_det_jacobian)

  def _log_cdf(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.log_cdf(x)

  def _cdf(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.cdf(x)

  def _log_survival_function(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.log_survival_function(x)

  def _survival_function(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.survival_function(x)

  def _quantile(self, value):
    inverse_cdf = self.base.quantile(value)
    return self.reversible_layer(inverse_cdf)

  def _entropy(self):
    dummy = tf.zeros(
        tf.concat([self.batch_shape_tensor(), self.event_shape_tensor()], 0),
        dtype=self.dtype)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(dummy)
    entropy = self.base.entropy() - log_det_jacobian
    return entropy


@ed.interceptable
def TransformedRandomVariable(random_variable,  # pylint: disable=invalid-name
                              reversible_layer,
                              name=None,
                              sample_shape=(),
                              value=None):
  """Random variable for f(x), where x ~ p(x) and f is reversible."""
  return ed.RandomVariable(
      distribution=TransformedDistribution(random_variable.distribution,
                                           reversible_layer,
                                           name=name),
      sample_shape=sample_shape,
      value=value)
