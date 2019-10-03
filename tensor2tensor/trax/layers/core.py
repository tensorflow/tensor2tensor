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
def ParametricRelu(x, a=1., **unused_kwargs):
  return np.maximum(a * x, np.zeros_like(x))


@base.layer()
def LeakyRelu(x, a=0.01, **unused_kwargs):
  return np.where(x >= 0, x, a * x)


@base.layer()
def Elu(x, a=1., **unused_kwargs):
  return np.where(x > 0, x, a * np.expm1(x))


@base.layer()
def Selu(x,
         alpha=1.6732632423543772848170429916717,
         lmbda=1.0507009873554804934193349852946):
  return lmbda * np.where(x > 0, x, alpha * np.expm1(x))


@base.layer()
def Gelu(x, **unused_kwargs):
  return x * backend.erf(x)


@base.layer()
def Sigmoid(x, **unused_kwargs):
  return backend.expit(x)


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
def LogSoftmax(x, axis=-1, **unused_kwargs):
  """Apply log softmax to x: log-normalize along the given axis."""
  return x - backend.logsumexp(x, axis, keepdims=True)


@base.layer()
def Softmax(x, axis=-1, **unused_kwargs):
  """Apply softmax to x: exponentiate and normalize along the given axis."""
  return np.exp(x - backend.logsumexp(x, axis, keepdims=True))


@base.layer()
def Softplus(x, **unused_kwargs):
  return np.logaddexp(x, 0.)


@base.layer()
def ToFloat(x, **unused_kwargs):
  return x.astype(onp.float32)


class Dense(base.Layer):
  """A dense (a.k.a. fully-connected, affine) layer."""

  def __init__(self,
               n_units,
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    super(Dense, self).__init__()
    self._n_units = n_units
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def forward(self, x, params=(), state=(), **kwargs):
    del kwargs
    w, b = params
    return np.dot(x, w) + b, state

  def new_params_and_state(self, input_shape, input_dtype, rng):
    del input_dtype
    rng1, rng2 = backend.random.split(rng, 2)
    w = self._kernel_initializer((input_shape[-1], self._n_units), rng1)
    b = self._bias_initializer((self._n_units,), rng2)
    return (w, b), ()


class Embedding(base.Layer):
  """Layer constructor function for an embedding layer."""

  def __init__(self,
               d_feature,
               vocab_size,
               kernel_initializer=init.GlorotUniformInitializer()):
    super(Embedding, self).__init__()
    self._d_feature = d_feature  # feature dimensionality
    self._vocab_size = vocab_size
    self._kernel_initializer = kernel_initializer

  def forward(self, x, params=(), state=(), **kwargs):
    del kwargs
    return np.take(params, x, axis=0), state

  def new_params_and_state(self, input_shape, input_dtype, rng):
    del input_shape, input_dtype
    out_dim = (self._vocab_size, self._d_feature)
    params = self._kernel_initializer(out_dim, rng)
    return params, ()


# Flatten.
@base.layer()
def Flatten(x, n_axes_to_keep=1, **unused_kwargs):
  if n_axes_to_keep >= len(x.shape):
    raise ValueError("n_axes_to_keep[%d] should be less than input's rank[%d]" %
                     (n_axes_to_keep, len(x.shape)))
  return np.reshape(x, (x.shape[:n_axes_to_keep] + (-1,)))


class Dropout(base.Layer):
  """Dropout."""

  def __init__(self, rate=0.0, name='dropout', mode='train'):
    super(Dropout, self).__init__()
    self._initial_rate = rate
    # TODO(lukaszkaiser): remove the name property by the end of September'19.
    # It's only needed for a specific purpose in the short term, will go.
    self._name = 'dropout_' + name
    self._mode = mode

  def new_params_and_state(self, input_shape, input_dtype, rng):
    del input_shape, input_dtype, rng
    params = ()
    state = {self._name: np.array(self._initial_rate)}
    return params, state

  def forward(self, x, params=(), state=(), rng=None, **kwargs):
    """Execute dropout."""
    del kwargs
    rate = self._initial_rate
    if isinstance(state, dict) and self._name in state:
      rate = state[self._name]
    if rng is None:
      msg = ('Dropout layer requires apply_fn to be called with a rng keyword '
             'argument. That is, instead of `Dropout(params, inputs)`, call '
             'it like `Dropout(params, inputs, rng=key)`.')
      raise ValueError(msg)
    if self._mode != 'train':
      return x, state
    keep = backend.random.bernoulli(rng, 1.0 - rate, x.shape)
    return np.where(keep, x / (1.0 - rate), np.zeros_like(x)), state


@base.layer()
def Div(x, divisor=1.0, **unused_kwargs):
  return x / divisor


@base.layer()
def AddConstant(x, constant=0.0, **unused_kwargs):
  return x + constant


@base.layer()
def MulConstant(x, constant=1.0, **unused_kwargs):
  return x * constant


def one_hot(x, size, dtype=np.float32):  # pylint: disable=invalid-name
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  arange_size = np.arange(size)
  if backend.get_name() == 'jax':
    # Work around a jax broadcasting issue.
    arange_size = jax.lax.tie_in(x, arange_size)
  return np.array(x[..., np.newaxis] == arange_size, dtype)


# Mean.
@base.layer()
def Mean(x, axis=-1, keepdims=False, **unused_kwargs):
  return np.mean(x, axis=axis, keepdims=keepdims)


def log_gaussian_pdf(x, mu, sigma):  # pylint: disable=invalid-name
  """Compute log N(x | mu, sigma)."""
  a = mu.shape[-1] * np.log(2 * np.pi)
  _, b = np.linalg.slogdet(sigma)
  y = np.linalg.solve(sigma, x - mu)
  y = np.expand_dims(y, axis=-1)
  xm = np.expand_dims(x - mu, axis=-2)
  c = np.matmul(xm, y)
  c = np.squeeze(np.squeeze(c, axis=-1), axis=-1)
  return -0.5 * (a + b + c)


def log_gaussian_diag_pdf(x, mu, diag_sigma):  # pylint: disable=invalid-name
  """Compute log N(x | mu, eye(diag_sigma))."""
  a = mu.shape[-1] * np.log(2 * np.pi)
  b = np.sum(np.log(diag_sigma), axis=-1)
  y = x - mu / diag_sigma
  y = np.expand_dims(y, axis=-1)
  xm = np.expand_dims(x - mu, axis=-2)
  c = np.matmul(xm, y)
  c = np.squeeze(np.squeeze(c, axis=-1), axis=-1)
  return -0.5 * (a + b + c)


def multigaussian_loss(preds, targets, ngauss=1):  # pylint: disable=invalid-name
  """Compute mixture of gaussians loss."""
  ndims = targets.shape[-1]
  logits = preds[:, :ngauss]
  mus = preds[:, ngauss:ngauss*(ndims + 1)]
  sigmas = preds[:, ngauss(ndims + 1):]
  sigmas = sigmas * sigmas + 1e-6  # Make positive.
  loglogits = logits - backend.logsumexp(logits, axis=-1, keepdims=True)
  mus = np.reshape(mus, [-1, ngauss, ndims])
  sigmas = np.reshape(sigmas, [-1, ngauss, ndims])
  targets = np.reshape(targets, [-1, 1, ndims])
  glogprobs = log_gaussian_diag_pdf(targets, mus, sigmas)
  return backend.logsumexp(loglogits + glogprobs, axis=-1)
