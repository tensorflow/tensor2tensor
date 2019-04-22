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

"""Trax layers library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import operator as op

from jax import lax

import numpy as onp
from six.moves import reduce
from tensor2tensor.trax import backend
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.stax import base

# Following the convention used in Keras and tf.layers, we use CamelCase for the
# names of layer constructors, like Conv and Relu, while using snake_case for
# other functions, like lax.conv and relu. To allow this, we disable below.
# pylint: disable=invalid-name


# Initializers.


def randn(stddev=1e-2):
  """An initializer function for random normal coefficients."""
  def init(rng, shape):
    return (stddev * backend.random.normal(rng, shape)).astype('float32')
  return init


def glorot(out_dim=0, in_dim=1, scale=onp.sqrt(2)):
  """An initializer function for random Glorot-scaled coefficients."""
  def init(rng, shape):
    fan_in, fan_out = shape[in_dim], shape[out_dim]
    size = onp.prod(onp.delete(shape, [in_dim, out_dim]))
    std = scale / np.sqrt((fan_in + fan_out) / 2. * size)
    return (std * backend.random.normal(rng, shape)).astype('float32')
  return init


def xavier_uniform(out_dim=0, in_dim=1):
  """An initializer function for random uniform xavier-scaled coefficients."""
  def init(rng, shape):
    fan_in, fan_out = shape[in_dim], shape[out_dim]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    a = np.sqrt(3.0) * std
    return backend.random.uniform(rng, shape, minval=-a, maxval=a)
  return init


def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)


# Layers.


@base.layer()
def Relu(params, x, **kwargs):
  del params, kwargs
  return np.maximum(x, 0.)


@base.layer()
def Tanh(params, x, **kwargs):
  del params, kwargs
  return np.tanh(x)


@base.layer()
def Exp(params, x, **kwargs):
  del params, kwargs
  return np.exp(x)


@base.layer()
def LogSoftmax(params, x, axis=-1, **kwargs):
  """Apply log softmax to x: log-normalize along the given axis."""
  del params, kwargs
  return x - backend.logsumexp(x, axis, keepdims=True)


@base.layer()
def Softmax(params, x, axis=-1, **kwargs):
  """Apply softmax to x: exponentiate and normalize along the given axis."""
  del params, kwargs
  return np.exp(x - backend.logsumexp(x, axis, keepdims=True))


@base.layer()
def Softplus(params, x, **kwargs):
  del params, kwargs
  return np.logaddexp(x, 0.)


class Dense(base.Layer):
  """Layer constructor function for a dense (fully-connected) layer."""

  def __init__(self, out_dim, W_init=glorot(), b_init=randn()):
    super(Dense, self).__init__()
    self._out_dim = out_dim
    self._W_init = W_init
    self._b_init = b_init

  def call(self, params, inputs, **kwargs):
    del kwargs
    w, b = params
    return np.dot(inputs, w) + b

  def output_shape(self, input_shape):
    return tuple(input_shape[:-1]) + (self._out_dim,)

  def new_parameters(self, input_shape, rng):
    w = self._W_init(rng, (input_shape[-1], self._out_dim))
    b = self._b_init(rng, (self._out_dim,))
    return (w, b)


class Embedding(base.Layer):
  """Layer constructor function for an embedding layer."""

  def __init__(self, feature_depth, vocab_size, W_init=xavier_uniform()):
    super(Embedding, self).__init__()
    self._feature_depth = feature_depth
    self._vocab_size = vocab_size
    self._W_init = W_init

  def call(self, params, inputs, **kwargs):
    del kwargs
    return np.take(params, inputs, axis=0)

  def output_shape(self, input_shape):
    return tuple(input_shape) + (self._feature_depth,)

  def new_parameters(self, input_shape, rng):
    return self._W_init(rng, (self._vocab_size, self._feature_depth))


def padtype_to_pads(in_shape, window_shape, window_strides, padding):
  """Convert padding string to list of pairs of pad values."""
  padding = padding.upper()
  if padding == 'SAME':
    out_shape = onp.ceil(
        onp.true_divide(in_shape, window_strides)).astype(int)
    pad_sizes = [max((out_size - 1) * stride + window_shape - in_size, 0)
                 for out_size, stride, window_shape, in_size
                 in zip(out_shape, window_strides, window_shape, in_shape)]
    return [(pad_size // 2, pad_size - pad_size // 2)
            for pad_size in pad_sizes]
  elif padding == 'VALID':
    return [(0, 0)] * len(in_shape)
  else:
    msg = 'Unknown padding type: {}.'
    raise TypeError(msg.format(padding))


class Conv(base.Layer):
  """Layer constructor function for a general convolution layer."""

  def __init__(self, out_chan, filter_shape, strides=None, padding='VALID',
               dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
               W_init=None, b_init=randn(1e-6)):
    super(Conv, self).__init__()
    self._out_chan = out_chan
    self._filter_shape = filter_shape
    self._padding = padding
    self._dimension_numbers = dimension_numbers
    self._lhs_spec, self._rhs_spec, self._out_spec = dimension_numbers
    self._one = (1,) * len(filter_shape)
    self._strides = strides or self._one
    self._b_init = b_init
    rhs_spec = self._rhs_spec
    self._W_init = W_init or glorot(rhs_spec.index('O'), rhs_spec.index('I'))

  def call(self, params, inputs, **kwargs):
    del kwargs
    w, b = params
    return lax.conv_general_dilated(
        inputs, w, self._strides, self._padding, self._one, self._one,
        self._dimension_numbers) + b

  def _kernel_shape(self, input_shape):
    """Helper to calculate the kernel shape."""
    filter_shape_iter = iter(self._filter_shape)
    return [self._out_chan if c == 'O' else
            input_shape[self._lhs_spec.index('C')] if c == 'I' else
            next(filter_shape_iter) for c in self._rhs_spec]

  def _conv_shape_tuple(self, lhs_shape, rhs_shape, strides, pads):
    """Compute the shape of a conv given input shapes in canonical order."""
    if isinstance(pads, str):
      pads = padtype_to_pads(lhs_shape[2:], rhs_shape[2:], strides, pads)
    if len(pads) != len(lhs_shape) - 2:
      msg = 'Wrong number of explicit pads for conv: expected {}, got {}.'
      raise TypeError(msg.format(len(lhs_shape) - 2, len(pads)))
    lhs_padded = onp.add(lhs_shape[2:], onp.add(*zip(*pads)))
    out_space = onp.floor_divide(
        onp.subtract(lhs_padded, rhs_shape[2:]), strides) + 1
    out_space = onp.maximum(0, out_space)
    out_shape = (lhs_shape[0], rhs_shape[0]) + tuple(out_space)
    return tuple(out_shape)

  def _conv_general_permutations(self, dimension_numbers):
    """Utility for convolution dimension permutations relative to Conv HLO."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    lhs_char, rhs_char, out_char = ('N', 'C'), ('O', 'I'), ('N', 'C')
    charpairs = (lhs_char, rhs_char, out_char)
    for i, (a, b) in enumerate(charpairs):
      if not (dimension_numbers[i].count(a) == 1 and
              dimension_numbers[i].count(b) == 1):
        msg = ('convolution dimension_numbers[{}] must contain the characters '
               '"{}" and "{}" exatly once, got {}.')
        raise TypeError(msg.format(i, a, b, dimension_numbers[i]))
      if len(dimension_numbers[i]) != len(set(dimension_numbers[i])):
        msg = ('convolution dimension_numbers[{}] cannot have duplicate '
               'characters, got {}.')
        raise TypeError(msg.format(i, dimension_numbers[i]))
    if not (set(lhs_spec) - set(lhs_char) == set(rhs_spec) - set(rhs_char) ==
            set(out_spec) - set(out_char)):
      msg = ('convolution dimension_numbers elements must each have the same '
             'set of spatial characters, got {}.')
      raise TypeError(msg.format(dimension_numbers))

    def getperm(spec, charpair):
      spatial = (i for i, c in enumerate(spec) if c not in charpair)
      if spec is not rhs_spec:
        spatial = sorted(spatial, key=lambda i: rhs_spec.index(spec[i]))
      return (spec.index(charpair[0]), spec.index(charpair[1])) + tuple(spatial)

    lhs_perm, rhs_perm, out_perm = map(getperm, dimension_numbers, charpairs)
    return lhs_perm, rhs_perm, out_perm

  def _conv_general_shape_tuple(self, lhs_shape, rhs_shape, window_strides,
                                padding, dimension_numbers):
    """Generalized computation of conv shape."""
    lhs_perm, rhs_perm, out_perm = self._conv_general_permutations(
        dimension_numbers)
    lhs_trans = onp.take(lhs_shape, lhs_perm)
    rhs_trans = onp.take(rhs_shape, rhs_perm)
    out_trans = self._conv_shape_tuple(
        lhs_trans, rhs_trans, window_strides, padding)
    return tuple(onp.take(out_trans, onp.argsort(out_perm)))

  def output_shape(self, input_shape):
    kernel_shape = self._kernel_shape(input_shape)
    return self._conv_general_shape_tuple(
        input_shape, kernel_shape,
        self._strides, self._padding, self._dimension_numbers)

  def new_parameters(self, input_shape, rng):
    kernel_shape = self._kernel_shape(input_shape)
    bias_shape = [self._out_chan if c == 'C' else 1 for c in self._out_spec]
    bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
    w = self._W_init(rng, kernel_shape)
    b = self._b_init(rng, bias_shape)
    return (w, b)


# Flatten.
def _flatten_output_shape(input_shape, num_axis_to_keep=1):
  """Output shape of a flatten layer."""
  if num_axis_to_keep >= len(input_shape):
    raise ValueError(
        "num_axis_to_keep[%d] should be less than input's rank[%d]" %
        (num_axis_to_keep, len(input_shape)))
  return tuple(input_shape[:num_axis_to_keep]) + (
      reduce(op.mul, input_shape[num_axis_to_keep:], 1),)


@base.layer(output_shape=_flatten_output_shape)
def Flatten(params, inputs, num_axis_to_keep=1, **kwargs):
  del params, kwargs
  return np.reshape(inputs, (inputs.shape[:num_axis_to_keep] + (-1,)))


# Batch normalization.
def _batch_norm_new_params(input_shape, rng, axis=(0, 1, 2),
                           center=True, scale=True, **kwargs):
  """Helper to initialize batch norm params."""
  del rng, kwargs
  axis = (axis,) if np.isscalar(axis) else axis
  shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
  beta = np.zeros(shape, dtype='float32') if center else ()
  gamma = np.ones(shape, dtype='float32') if scale else ()
  return (beta, gamma)


@base.layer(new_parameters=_batch_norm_new_params)
def BatchNorm(params, x, axis=(0, 1, 2), epsilon=1e-5,
              center=True, scale=True, **unused_kwargs):
  """Layer construction function for a batch normalization layer."""
  mean = np.mean(x, axis, keepdims=True)
  # Fast but less numerically-stable variance calculation than np.var.
  m1 = np.mean(x**2, axis, keepdims=True)
  var = m1 - mean**2
  z = (x - mean) / np.sqrt(var + epsilon)

  # Expand the parameters to have the right axes.
  beta, gamma = params
  # TODO(phawkins): np.expand_dims should accept an axis tuple.
  # (https://github.com/numpy/numpy/issues/12290)
  ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x)))
  beta = beta[ed]
  gamma = gamma[ed]

  # Return the z rescaled by the parameters if requested.
  if center and scale:
    return gamma * z + beta
  if center:
    return z + beta
  if scale:
    return gamma * z
  return z


# Pooling.
def _pooling_output_shape(input_shape, pool_size=(2, 2),
                          strides=None, padding='VALID'):
  """Helper: compute the output shape for the pooling layer."""
  dims = (1,) + pool_size + (1,)  # NHWC
  spatial_strides = strides or (1,) * len(pool_size)
  strides = (1,) + spatial_strides + (1,)
  pads = padtype_to_pads(input_shape, dims, strides, padding)
  operand_padded = onp.add(input_shape, onp.add(*zip(*pads)))
  t = onp.floor_divide(onp.subtract(operand_padded, dims), strides) + 1
  return tuple(t)


def _pooling_general(inputs, reducer, init_val, rescaler=None,
                     pool_size=(2, 2), strides=None, padding='VALID'):
  """Helper: general pooling computation used in pooling layers later."""
  spatial_strides = strides or (1,) * len(pool_size)
  rescale = rescaler(pool_size, spatial_strides, padding) if rescaler else None
  dims = (1,) + pool_size + (1,)  # NHWC
  strides = (1,) + spatial_strides + (1,)
  out = lax.reduce_window(inputs, init_val, reducer, dims, strides, padding)
  return rescale(out, inputs) if rescale else out


@base.layer(output_shape=_pooling_output_shape)
def MaxPool(params, x, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return _pooling_general(x, lax.max, -np.inf, pool_size=pool_size,
                          strides=strides, padding=padding)


@base.layer(output_shape=_pooling_output_shape)
def SumPool(params, x, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return _pooling_general(x, lax.add, 0., pool_size=pool_size,
                          strides=strides, padding=padding)


def _normalize_by_window_size(dims, spatial_strides, padding):
  def rescale(outputs, inputs):
    one = np.ones(inputs.shape[1:-1], dtype=inputs.dtype)
    window_sizes = lax.reduce_window(
        one, 0., lax.add, dims, spatial_strides, padding)
    return outputs / window_sizes[..., np.newaxis]
  return rescale


@base.layer(output_shape=_pooling_output_shape)
def AvgPool(params, x, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return _pooling_general(x, lax.add, 0., _normalize_by_window_size,
                          pool_size, strides=strides, padding=padding)


@base.layer()
def Dropout(params, x, rate=0.0, mode='train', rng=None, **kwargs):
  """Layer construction function for a dropout layer with given rate."""
  del params, kwargs
  if rng is None:
    msg = ('Dropout layer requires apply_fun to be called with a rng keyword '
           'argument. That is, instead of `Dropout(params, inputs)`, call '
           'it like `Dropout(params, inputs, rng=key)`.')
    raise ValueError(msg)
  if rate >= 1.0:
    raise ValueError('Dropout rate (%f) must be lower than 1.' % rate)
  if mode == 'train' and rate > 0.0:
    keep = backend.random.bernoulli(rng, 1.0 - rate, x.shape)
    return np.where(keep, x / (1.0 - rate), 0)
  else:
    return x


@base.layer()
def Div(params, x, divisor=1.0, **kwargs):
  del params, kwargs
  return x / divisor


@base.layer()
def ShiftRight(params, inputs, **kwargs):
  """Layer to shift the tensor to the right by padding on axis 1."""
  del params, kwargs
  pad_widths = [(0, 0), (1, 0)]
  pad_widths += [(0, 0) for _ in range(len(inputs.shape) - 2)]
  padded = np.pad(inputs, pad_widths, mode='constant')
  return padded[:, :-1, ...]
