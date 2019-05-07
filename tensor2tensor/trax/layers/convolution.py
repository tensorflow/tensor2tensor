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

import numpy as onp
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import initializers as init


def PadtypeToPads(in_shape, window_shape, window_strides, padding):
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

  def _conv_shape_tuple(self, lhs_shape, rhs_shape, strides, pads):
    """Compute the shape of a conv given input shapes in canonical order."""
    if isinstance(pads, str):
      pads = PadtypeToPads(lhs_shape[2:], rhs_shape[2:], strides, pads)
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

    def GetPerm(spec, charpair):
      spatial = (i for i, c in enumerate(spec) if c not in charpair)
      if spec is not rhs_spec:
        spatial = sorted(spatial, key=lambda i: rhs_spec.index(spec[i]))
      return (spec.index(charpair[0]), spec.index(charpair[1])) + tuple(spatial)

    lhs_perm, rhs_perm, out_perm = map(GetPerm, dimension_numbers, charpairs)
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
    bias_shape = [self._filters if c == 'C' else 1 for c in self._out_spec]
    bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
    w = self._kernel_initializer(kernel_shape, rng)
    b = self._bias_initializer(bias_shape, rng)
    return (w, b)
