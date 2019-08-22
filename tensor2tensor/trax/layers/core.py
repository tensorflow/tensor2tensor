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

import jax
import numpy as onp

from tensor2tensor.trax import backend
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import initializers as init


@base.layer()
def Relu(x, **unused_kwargs):
  return np.maximum(x, np.zeros_like(x))


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


@base.layer()
def ToFloat(x, **unused_kwargs):
  return x.astype(onp.float32)


class Dense(base.Layer):
  """Layer constructor function for a dense (fully-connected) layer."""

  def __init__(self, n_units,
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    super(Dense, self).__init__()
    self._n_units = n_units
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def call(self, x, params, state, **kwargs):
    del kwargs
    w, b = params
    return np.dot(x, w) + b, state

  def new_parameters(self, input_shape, input_dtype, rng):
    del input_dtype
    rng1, rng2 = backend.random.split(rng, 2)
    w = self._kernel_initializer((input_shape[-1], self._n_units), rng1)
    b = self._bias_initializer((self._n_units,), rng2)
    return (w, b), ()


class Embedding(base.Layer):
  """Layer constructor function for an embedding layer."""

  def __init__(self, d_feature, vocab_size,
               kernel_initializer=init.GlorotUniformInitializer()):
    super(Embedding, self).__init__()
    self._d_feature = d_feature  # feature dimensionality
    self._vocab_size = vocab_size
    self._kernel_initializer = kernel_initializer

  def call(self, x, params, state, **kwargs):
    del kwargs
    return np.take(params, x, axis=0), state

  def new_parameters(self, input_shape, input_dtype, rng):
    del input_dtype
    return self._kernel_initializer(
        (self._vocab_size, self._d_feature), rng), ()


# Flatten.
@base.layer()
def Flatten(x, params, n_axes_to_keep=1, **kwargs):
  del params, kwargs
  if n_axes_to_keep >= len(x.shape):
    raise ValueError(
        "n_axes_to_keep[%d] should be less than input's rank[%d]" %
        (n_axes_to_keep, len(x.shape)))
  return np.reshape(x, (x.shape[:n_axes_to_keep] + (-1,)))


@base.layer()
def Dropout(x, params, rate=0.0, mode='train', rng=None, **kwargs):
  """Layer construction function for a dropout layer with given rate."""
  del params, kwargs
  if rng is None:
    msg = ('Dropout layer requires apply_fn to be called with a rng keyword '
           'argument. That is, instead of `Dropout(params, inputs)`, call '
           'it like `Dropout(params, inputs, rng=key)`.')
    raise ValueError(msg)
  if rate >= 1.0:
    raise ValueError('Dropout rate (%f) must be lower than 1.' % rate)
  if mode == 'train' and rate > 0.0:
    keep = backend.random.bernoulli(rng, 1.0 - rate, x.shape)
    return np.where(keep, x / (1.0 - rate), np.zeros_like(x))
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
  arange_size = np.arange(size)
  if backend.get_name() == 'jax':
    # Work around a jax broadcasting issue.
    arange_size = jax.lax.tie_in(x, arange_size)
  return np.array(x[..., np.newaxis] == arange_size, dtype)


# Mean.
@base.layer()
def Mean(x, params, axis=-1, keepdims=False, **kwargs):
  del params, kwargs
  return np.mean(x, axis=axis, keepdims=keepdims)
