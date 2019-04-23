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

"""Combinators for composing layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax import backend
from tensor2tensor.trax.stax import base


class Serial(base.Layer):
  """Layer composing a number of sub-layers in a serial way.."""

  def __init__(self, *layers):
    super(Serial, self).__init__()
    self._nlayers = len(layers)
    self._layers = layers

  def call(self, x, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._nlayers
    if rng is not None:
      rngs = backend.random.split(rng, self._nlayers)
    for layer, p, rng in zip(self._layers, params, rngs):
      x = layer(x, p, rng=rng, **kwargs)
    return x

  def output_shape(self, input_shape):
    cur_shape = input_shape
    for layer in self._layers:
      cur_shape = layer.output_shape(cur_shape)
    return cur_shape

  def new_parameters(self, input_shape, rng):
    params = []
    cur_shape = input_shape
    for layer in self._layers:
      rng, layer_rng = backend.random.split(rng)
      param = layer.initialize(cur_shape, layer_rng)
      cur_shape = layer.output_shape(cur_shape)
      params.append(param)
    return params


@base.layer()
def Identity(x, **unused_kwargs):
  return x


@base.layer(output_shape=lambda input_shape, size=2: [input_shape] * size)
def FanOut(x, params, size=2, **kwargs):
  del params, kwargs
  return [x] * size


@base.layer(output_shape=lambda input_shape_list: input_shape_list[0])
def FanInSum(x, **unused_kwargs):
  return sum(x)  # Here x is a list of tensors of the same shape, we add them.


def _fan_in_concat_shape(input_shape, axis=-1):  # pylint: disable=invalid-name
  """Helper to determine the shape of FanInConcat output."""
  ax = axis % len(input_shape[0])
  concat_size = sum(shape[ax] for shape in input_shape)
  out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax+1:]
  return out_shape


@base.layer(output_shape=_fan_in_concat_shape)
def FanInConcat(x, params, axis=-1, **kwargs):
  del params, kwargs
  return backend.numpy.concatenate(x, axis)


class Parallel(base.Layer):
  """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the FanOut and
  FanInSum layers.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the
    parallel composition of the given sequence of layers. In particular, the
    returned layer takes a sequence of inputs and returns a sequence of outputs
    with the same length as the argument `layers`.
  """

  def __init__(self, *layers):
    super(Parallel, self).__init__()
    self._nlayers = len(layers)
    self._layers = layers

  def call(self, inputs, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._nlayers
    if rng is not None:
      rngs = backend.random.split(rng, self._nlayers)
    return [layer(x, params=p, rng=r, **kwargs)
            for layer, x, p, r in zip(self._layers, inputs, params, rngs)]

  def output_shape(self, input_shapes):
    return tuple([layer.output_shape(shape)
                  for layer, shape in zip(self._layers, input_shapes)])

  def new_parameters(self, input_shape, rng):
    rngs = backend.random.split(rng, self._nlayers)
    return [layer.initialize(shape, rng) for layer, shape, rng
            in zip(self._layers, input_shape, rngs)]


def Residual(*layers, **kwargs):
  """Constructs a residual version of layers, summing input to layers output."""
  res = kwargs.get('res', Identity())  # pylint: disable=no-value-for-parameter
  if len(layers) > 1:
    return Serial(
        FanOut(),  # pylint: disable=no-value-for-parameter
        Parallel(Serial(*layers), res),
        FanInSum()  # pylint: disable=no-value-for-parameter
    )
  elif len(layers) == 1:
    return Serial(
        FanOut(),  # pylint: disable=no-value-for-parameter
        Parallel(layers[0], res),
        FanInSum()  # pylint: disable=no-value-for-parameter
    )
  else:
    raise ValueError('Empty residual combinator.')
