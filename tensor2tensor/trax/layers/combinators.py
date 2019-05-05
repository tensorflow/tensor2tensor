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

import operator
import six

from tensor2tensor.trax import backend
from tensor2tensor.trax.layers import base


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
      cur_shape = layer.output_shape_catch_errors(cur_shape)
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
def Copy(x, **unused_kwargs):
  """Copy layer, return the inputs."""
  return x


def Unnest(x):
  """Helper: remove nesting in x, return a flat tuple."""
  if not isinstance(x, (list, tuple)):
    return (x,)
  return tuple([z for y in x for z in Unnest(y)])  # pylint: disable=g-complex-comprehension


def UnnestShape(shape):
  """Unnest a nested structure of shapes."""

  class Shape(object):
    """Since shapes are tuples, make them a class to not unnest too far."""

    def __init__(self, shape):
      self.shape = shape

  def MakeShape(nested_shape):
    """Make all shape-tuples in the nested object shape-classes."""
    if isinstance(nested_shape[0], int):  # Not nested.
      return Shape(nested_shape)
    return [MakeShape(shape) for shape in nested_shape]

  # Unnest on the level of shape-classes and bring back shape-tuples.
  return tuple([y.shape for y in Unnest(MakeShape(shape))])


@base.layer(output_shape=UnnestShape)
def UnnestBranches(x, **unused_kwargs):
  return Unnest(x)


# Re-ordering layer.
class Select(base.Layer):
  """Select elements from a tuple or create another tuple from them.

  For example, we can re-order (x, y) into (y, x) or even (y, (x, y), y).
  The output argument specifies how to re-order, using integers that refer
  to indices in the input tuple. For example, if

    input = (x, y, z)

  then

    Select(0)                = x
    Select((1, 0, 2))        = (y, x, z)
    Select((0, 0))           = (x, x)
    Select((0, (1, 1)))      = (x, (y, y))
    Select(((2, 0), (1, 1))) = ((z, x), (y, y))

  By default (if no output is given) Select does nothing (Copy).

  Args:
    x: the input tuple to re-order.
    params: layer parameters (unused).
    output: the specification of the output tuple: a nested tuple of ints.
    **kwargs: other arguments (unused).

  Returns:
    The re-ordered tuple with the same shape as output.
  """

  def __init__(self, output=None):
    super(Select, self).__init__()
    self._output = output

  def call(self, x, params=(), **kwargs):
    del params, kwargs
    if self._output is None:
      return x
    return base.nested_map(self._output, lambda i: x[i])

  def output_shape(self, input_shape):
    if self._output is None:
      return input_shape
    return base.nested_map(self._output, lambda i: input_shape[i])

  def new_parameters(self, input_shape, rng):
    return ()


class Branch(base.Layer):
  """Combinator for applying layers to copies of the input.

  This layer is often used to create parallel towers in neural networks:
  * Branch(Copy(), Copy()) -- creates a pair with copied input
  * Branch(main, shortcut) -- start a residual tower (see Residual below)

  Args:
    *layers: a sequence of layers.
    **kwlayers: a dictionary of layers.

  Returns:
    A new layer in which each of the given layers has been applied to
    a copy of the input independently.
  """

  def __init__(self, *layers, **kwlayers):
    super(Branch, self).__init__()
    if layers and kwlayers:
      raise ValueError('Cannot specify a Branch with both a list and dict.')
    layers = layers or kwlayers
    self._nlayers = len(layers)
    self._layers = layers

  def call(self, x, params=(), **kwargs):
    # Split the random number generators.
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._nlayers
    if rng is not None:
      rngs = backend.random.split(rng, self._nlayers)
    # If layers are a list or a tuple, just apply them.
    if isinstance(self._layers, (list, tuple)):
      res = [layer(x, params=p, rng=r, **kwargs)
             for layer, p, r in zip(self._layers, params, rngs)]
      return tuple(res)
    # If layers are a dictionary, apply to matching keys.
    assert isinstance(self._layers, dict)
    result, counter = {}, 0
    for k in self._layers:
      result[k] = self._layers[k](
          x, params=params[k], rng=rngs[counter], **kwargs)
      counter += 1
    return result

  def output_shape(self, input_shape):
    output_shapes = []
    # If the argument layers are a sequence, apply each to calculate shape.
    if not isinstance(self._layers, dict):
      for layer in self._layers:
        output_shapes.append(layer.output_shape_catch_errors(input_shape))
      return tuple(output_shapes)
    # If layers are a dictionary, apply to the input shape.
    result = {}
    for k in self._layers:
      result[k] = self._layers[k].output_shape_catch_errors(input_shape)
    return result

  def new_parameters(self, input_shape, rng):
    rngs = backend.random.split(rng, self._nlayers)
    # If the argument layers are a sequence, create parameters for each one.
    if not isinstance(self._layers, dict):
      return [layer.initialize(input_shape, rng) for layer, rng
              in zip(self._layers, rngs)]
    # If the argument layers are a dictionary, create a dictionary too.
    result, counter = {}, 0
    for k in self._layers:
      result[k] = self._layers[k].initialize(input_shape, rngs[counter])
      counter += 1
    return result


def _nested_op(inputs, op):  # pylint: disable=invalid-name
  """Helper: apply op over a list of arrays or nested arrays."""
  # If input is a dictionary, apply to the values (ignore keys).
  if isinstance(inputs, dict):
    return _nested_op(list(inputs.values()), op)
  # First the simple non-nested case.
  if not isinstance(inputs[0], (list, tuple)):
    return op(inputs)
  # In the nested case, sum on each axis separately.
  result_list = []
  for i in range(len(inputs[0])):
    result_list.append(_nested_op([x[i] for x in inputs], op=op))
  if isinstance(inputs[0], list):
    return result_list
  return tuple(result_list)


def _nested_sum(inputs):  # pylint: disable=invalid-name
  return _nested_op(inputs=inputs, op=sum)


def _nested_product(inputs):  # pylint: disable=invalid-name
  return _nested_op(
      inputs=inputs, op=lambda xs: six.moves.reduce(operator.mul, xs))


def _first_from_tuple_or_dict(tuple_or_dict):  # pylint: disable=invalid-name
  """Helper: return the first element from a tuple or dict."""
  for x in tuple_or_dict:
    return x


@base.layer(output_shape=_first_from_tuple_or_dict)
def Add(x, **unused_kwargs):
  """Add branches elementwise."""
  # Here x is a list of tensors of the same shape, or nested structures.
  return _nested_sum(x)


@base.layer(output_shape=_first_from_tuple_or_dict)
def Multiply(x, **unused_kwargs):
  """Multiply branches elementwise."""
  return _nested_product(x)


@base.layer(output_shape=_first_from_tuple_or_dict)
def Gate(x, **unused_kwargs):
  """Implements a gating function on a (memory, gate, candidate) tuple.

  Final update is memory * gate + (1-gate) * candidate

  This gating equation may also be referred to as Highway Network.
  Highway Networks: https://arxiv.org/abs/1505.00387

  Args:
    x: A tuple of (memory, gate, candidate)

  Returns:
    The result of applying gating.
  """
  assert len(x) == 3, x
  state, gate, candidate = x
  return gate * state + (1.0 - gate) * candidate


def _concatenate_shape(input_shape, axis=-1):  # pylint: disable=invalid-name
  """Helper to determine the shape of Concatenate output."""
  if isinstance(input_shape, dict):  # For named tuples, just use the values.
    input_shape = list(input_shape.values())
  ax = axis % len(input_shape[0])
  concat_size = sum(shape[ax] for shape in input_shape)
  out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax+1:]
  return out_shape


@base.layer(output_shape=_concatenate_shape)
def Concatenate(x, params, axis=-1, **kwargs):
  del params, kwargs
  if isinstance(x, dict):  # For dictionaries, just use the values.
    x = list(x.values())
  return backend.numpy.concatenate(x, axis)


class Parallel(base.Layer):
  """Combinator for applying layers to parts of a tuple.

  This layer is often used with the Branch and Add layers.

  Args:
    *layers: a sequence of layers.
    **kwlayers: a dictionary of layers.

  Returns:
    A new layer in which each of the given layers has been applied to
    its corresponding argument in the input tuple or dictionary.
  """

  def __init__(self, *layers, **kwlayers):
    super(Parallel, self).__init__()
    if layers and kwlayers:
      raise ValueError('Cannot specify a Parallel with both a list and dict.')
    layers = layers or kwlayers
    self._nlayers = len(layers)
    self._layers = layers

  def call(self, inputs, params=(), **kwargs):
    # Split the random number generators.
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._nlayers
    if rng is not None:
      rngs = backend.random.split(rng, self._nlayers)
    # If layers are a list or a tuple, just apply them.
    if not isinstance(self._layers, dict):
      res = [layer(x, params=p, rng=r, **kwargs)
             for layer, x, p, r in zip(self._layers, inputs, params, rngs)]
      # Return a list if inputs are a list and a tuple if inputs are a tuple.
      if isinstance(inputs, list):
        return res
      return tuple(res)
    # If layers are a dictionary, apply to matching keys.
    result, counter = {}, 0
    for k in inputs:
      if k in self._layers:
        result[k] = self._layers[k](
            inputs[k], params=params[k], rng=rngs[counter], **kwargs)
        counter += 1
      else:
        result[k] = inputs[k]
    return result

  def output_shape(self, input_shape):
    output_shapes = []
    # If the argument layers are a sequence, apply each to calculate shape.
    if not isinstance(self._layers, dict):
      for i, layer in enumerate(self._layers):
        output_shapes.append(layer.output_shape_catch_errors(input_shape[i]))
      return tuple(output_shapes)
    # If layers are a dictionary, apply to matching keys in the input shape.
    result = {}
    for k in input_shape:
      if k in self._layers:
        result[k] = self._layers[k].output_shape_catch_errors(input_shape[k])
      else:
        result[k] = input_shape[k]
    return result

  def new_parameters(self, input_shape, rng):
    rngs = backend.random.split(rng, self._nlayers)
    # If the argument layers are a sequence, create parameters for each one.
    if not isinstance(self._layers, dict):
      return [layer.initialize(shape, rng) for layer, shape, rng
              in zip(self._layers, input_shape, rngs)]
    # If the argument layers are a dictionary, create a dictionary too.
    result, counter = {}, 0
    for k in self._layers:
      result[k] = self._layers[k].initialize(input_shape[k], rngs[counter])
      counter += 1
    return result


def Residual(*layers, **kwargs):
  """Constructs a residual version of layers, summing input to layers output."""
  shortcut = kwargs.get('shortcut', Copy())  # pylint: disable=no-value-for-parameter
  if len(layers) > 1:
    return Serial(
        Branch(Serial(*layers), shortcut),
        Add()  # pylint: disable=no-value-for-parameter
    )
  elif len(layers) == 1:
    return Serial(
        Branch(layers[0], shortcut),
        Add()  # pylint: disable=no-value-for-parameter
    )
  else:
    raise ValueError('Empty residual combinator.')


class Map(base.Layer):
  """Combinator for applying a layer to a list or tuple.

  Args:
    layer: a layer to apply to each element.

  Returns:
    A new layer representing mapping layer to all elements of the input.
  """

  def __init__(self, layer, check_shapes=True):
    super(Map, self).__init__()
    self._layer = layer
    # Generally a Map should be applied to lists where all elements have
    # the same shape -- because self._layer will only be initialized once
    # and it could have different parameters for different shapes. But there
    # are valid cases -- e.g., when self._layer has no parameters -- where we
    # can apply Map to different shapes -- set check_shapes=False in such cases.
    self._check_shapes = check_shapes

  def call(self, inputs, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * len(inputs)
    if rng is not None:
      rngs = backend.random.split(rng, len(inputs))
    result = [self._layer(x, params=params, rng=r, **kwargs)
              for x, r in zip(inputs, rngs)]
    if isinstance(inputs, list):
      return result
    return tuple(result)

  def output_shape(self, input_shapes):
    return tuple([self._layer.output_shape(shape) for shape in input_shapes])

  def new_parameters(self, input_shape, rng):
    first_shape = input_shape[0]
    if self._check_shapes:
      for shape in input_shape:
        if shape != first_shape:
          raise ValueError('Map layer can only be applied to list of elements '
                           'with the same shapes. Shapes: %s' % str(shape))
    return self._layer.initialize(first_shape, rng)
