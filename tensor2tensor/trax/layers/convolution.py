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

from jax import lax

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

  def stack_items_to_pass(self):
    return 1

  def call(self, x, params=(), **kwargs):
    del kwargs
    w, b = params
    return lax.conv_general_dilated(
        x, w, self._strides, self._padding, self._one, self._one,
        self._dimension_numbers) + b

  def _kernel_shape(self, input_shape):
    """Helper to calculate the kernel shape."""
    kernel_size_iter = iter(self._kernel_size)
    return [self._filters if c == 'O' else
            input_shape[self._lhs_spec.index('C')] if c == 'I' else
            next(kernel_size_iter) for c in self._rhs_spec]

  def new_parameters(self, input_shape, rng):
    kernel_shape = self._kernel_shape(input_shape)
    bias_shape = [self._filters if c == 'C' else 1 for c in self._out_spec]
    bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
    w = self._kernel_initializer(kernel_shape, rng)
    b = self._bias_initializer(bias_shape, rng)
    return (w, b)
