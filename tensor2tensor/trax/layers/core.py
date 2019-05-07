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

import operator as op
from six.moves import reduce
from tensor2tensor.trax import backend
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import initializers as init


@base.layer()
def Relu(x, **unused_kwargs):
  return np.maximum(x, np.array(0, dtype=x.dtype))


@base.layer()
def Sigmoid(x, **unused_kwargs):
  return 1. / (1. + np.exp(-x))


@base.layer()
def Tanh(x, **unused_kwargs):
  return np.tanh(x)


@base.layer()
def HardSigmoid(x, **unused_kwargs):
  """Linear approximation to sigmoid."""
  return np.maximum(0, np.minimum(1, (1 + x)))


@base.layer()
def HardTanh(x, **unused_kwargs):
  """Linear approximation to tanh."""
  return np.maximum(-1, np.minimum(1, x))


@base.layer()
def Exp(x, **unused_kwargs):
  return np.exp(x)


@base.layer()
def LogSoftmax(x, params, axis=-1, **kwargs):
  """Apply log softmax to x: log-normalize along the given axis."""
  del params, kwargs
  return x - backend.logsumexp(x, axis, keepdims=True)


@base.layer()
def Softmax(x, params, axis=-1, **kwargs):
  """Apply softmax to x: exponentiate and normalize along the given axis."""
  del params, kwargs
  return np.exp(x - backend.logsumexp(x, axis, keepdims=True))


@base.layer()
def Softplus(x, **unused_kwargs):
  return np.logaddexp(x, 0.)


class Dense(base.Layer):
  """Layer constructor function for a dense (fully-connected) layer."""

  def __init__(self, units,
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    super(Dense, self).__init__()
    self._units = units
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def call(self, x, params, **kwargs):
    del kwargs
    w, b = params
    return np.dot(x, w) + b

  def output_shape(self, input_shape):
    return tuple(input_shape[:-1]) + (self._units,)

  def new_parameters(self, input_shape, rng):
    rng1, rng2 = backend.random.split(rng, 2)
    w = self._kernel_initializer((input_shape[-1], self._units), rng1)
    b = self._bias_initializer((self._units,), rng2)
    return (w, b)


class Embedding(base.Layer):
  """Layer constructor function for an embedding layer."""

  def __init__(self, feature_depth, vocab_size,
               kernel_initializer=init.GlorotUniformInitializer()):
    super(Embedding, self).__init__()
    self._feature_depth = feature_depth
    self._vocab_size = vocab_size
    self._kernel_initializer = kernel_initializer

  def call(self, x, params, **kwargs):
    del kwargs
    return np.take(params, x, axis=0)

  def output_shape(self, input_shape):
    return tuple(input_shape) + (self._feature_depth,)

  def new_parameters(self, input_shape, rng):
    return self._kernel_initializer(
        (self._vocab_size, self._feature_depth), rng)


# Flatten.
def _flatten_output_shape(input_shape, num_axis_to_keep=1):  # pylint: disable=invalid-name
  """Output shape of a flatten layer."""
  if num_axis_to_keep >= len(input_shape):
    raise ValueError(
        "num_axis_to_keep[%d] should be less than input's rank[%d]" %
        (num_axis_to_keep, len(input_shape)))
  return tuple(input_shape[:num_axis_to_keep]) + (
      reduce(op.mul, input_shape[num_axis_to_keep:], 1),)


@base.layer(output_shape=_flatten_output_shape)
def Flatten(x, params, num_axis_to_keep=1, **kwargs):
  del params, kwargs
  return np.reshape(x, (x.shape[:num_axis_to_keep] + (-1,)))


@base.layer()
def Dropout(x, params, rate=0.0, mode='train', rng=None, **kwargs):
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
def Div(x, params, divisor=1.0, **kwargs):
  del params, kwargs
  return x / divisor


@base.layer()
def AddConstant(x, params, constant=0.0, **unused_kwargs):
  del params
  return x + constant


def one_hot(x, size, dtype=np.float32):  # pylint: disable=invalid-name
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)


# Mean.
def _mean_output_shape(input_shape, axis=-1, keepdims=False):  # pylint: disable=invalid-name
  shape1 = list(input_shape)[:axis]  # Shape before axis.
  shape2 = list(input_shape)[axis:][1:]  # Shape after axis.
  mid_shape = [1] if keepdims else []
  return tuple(shape1 + mid_shape + shape2)


@base.layer(output_shape=_mean_output_shape)
def Mean(x, params, axis=-1, keepdims=False, **kwargs):
  del params, kwargs
  return np.mean(x, axis=axis, keepdims=keepdims)
