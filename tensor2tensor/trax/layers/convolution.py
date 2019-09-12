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

"""Trax convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import operator

import six

from tensor2tensor.trax import backend
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import initializers as init


class Conv(base.Layer):
  """Layer constructor function for a general convolution layer."""

  def __init__(self, filters, kernel_size, strides=None, padding='VALID',
               dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
               kernel_initializer=None,
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    super(Conv, self).__init__()
    self._filters = filters
    self._kernel_size = kernel_size
    self._padding = padding
    self._dimension_numbers = dimension_numbers
    self._lhs_spec, self._rhs_spec, self._out_spec = dimension_numbers
    self._one = (1,) * len(kernel_size)
    self._strides = strides or self._one
    self._bias_initializer = bias_initializer
    rhs_spec = self._rhs_spec
    self._kernel_initializer = kernel_initializer
    if kernel_initializer is None:
      self._kernel_initializer = init.GlorotNormalInitializer(
          rhs_spec.index('O'), rhs_spec.index('I'))

  def _check_nhwc(self):
    msg = 'Convolutions on more than 4 dimensions only supported in NHWC.'
    assert self._lhs_spec == self._out_spec == 'NHWC', msg

  def call(self, x, params=(), state=(), **kwargs):
    del kwargs
    w, b = params
    x_shape = list(x.shape)
    if len(x_shape) > 4:
      self._check_nhwc()
      new_batch_dim = six.moves.reduce(operator.mul, x_shape[:-3])
      x = np.reshape(x, [new_batch_dim] + x_shape[-3:])
    res = backend.conv(
        x, w, self._strides, self._padding, self._dimension_numbers,
        self._one) + b
    if len(x_shape) > 4:
      res = np.reshape(res, x_shape[:-3] + list(res.shape[-3:]))
    return res, state

  def _kernel_shape(self, input_shape):
    """Helper to calculate the kernel shape."""
    kernel_size_iter = iter(self._kernel_size)
    return [self._filters if c == 'O' else
            input_shape[self._lhs_spec.index('C')] if c == 'I' else
            next(kernel_size_iter) for c in self._rhs_spec]

  def new_parameters(self, input_shape, input_dtype, rng):
    del input_dtype
    if len(input_shape) > 4:
      self._check_nhwc()
      new_batch_dim = six.moves.reduce(operator.mul, input_shape[:-3])
      input_shape = [new_batch_dim] + list(input_shape[-3:])
    kernel_shape = self._kernel_shape(input_shape)
    bias_shape = [self._filters if c == 'C' else 1 for c in self._out_spec]
    bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
    w = self._kernel_initializer(kernel_shape, rng)
    b = self._bias_initializer(bias_shape, rng)
    return (w, b), ()


class CausalConv(Conv):
  """Causal (masked) convolution for [batch x time x depth] sequences.

  Maintains causality along time axis. Used in language modeling tasks.
  """

  def __init__(self,
               filters,
               kernel_width=3,
               kernel_initializer=None,
               bias_initializer=init.RandomNormalInitializer(1e-6)):
    super(CausalConv, self).__init__(
        filters=filters,
        kernel_size=(kernel_width,),
        strides=None,
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)

  def call(self, x, params=(), **kwargs):
    assert self._padding == 'VALID'
    # Left pad with 0s. Applying an unmasked valid convolution on top of this
    # yields a causal convolution.
    # TODO(ddohan): Support strided and dilated convolutions.
    rate = 1
    effective_kernel_size = int((self._kernel_size[0] - 1) * rate + 1)
    pad = effective_kernel_size - 1
    x_leftpad = np.pad(x, pad_width=[[0, 0], [pad, 0], [0, 0]], mode='constant')

    res = super(CausalConv, self).call(x_leftpad, params)
    return res
