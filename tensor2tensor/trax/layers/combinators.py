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
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base


def _DeepFlatten(xs):  # pylint: disable=invalid-name
  for x in xs:
    if isinstance(x, (list, tuple)):
      for y in _DeepFlatten(x):
        yield y
    else:
      yield x


def _EnsureSublayers(layers):
  # TODO(jonni): Implement for dict if dicts remain important.
  if isinstance(layers, dict):
    return layers
  sublayers_not_lists = []
  for layer in layers:
    sublayers_not_lists.append(
        Serial(layer) if isinstance(layer, list) else layer)
  return sublayers_not_lists


class Serial(base.Layer):
  """Layer composing a number of sub-layers in a serial way.."""

  def __init__(self, *layers):
    super(Serial, self).__init__()
    layers = list(_DeepFlatten(layers))
    # TODO(jonni): Consider flattening (unpacking) also embedded Serial layers.
    self._layers = layers
    self._nlayers = len(layers)

  def call(self, x, params=(), **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = (None,) * self._nlayers
    if rng is not None:
      rngs = backend.random.split(rng, self._nlayers)
    for layer, p, rng in zip(self._layers, params, rngs):
      x = layer(x, p, rng=rng, **kwargs)
    return x

  def output_shape_fn(self, input_shape):
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
def NoOp(x, **unused_kwargs):
  """NoOp layer, return the inputs."""
  return x


def _print_shape(x, message='PrintShape'):  # pylint: disable=invalid-name
  print(message + ' ; stack shape = ' + str(x))
  return x


@base.layer(output_shape=_print_shape, stack_items_to_pass=0)
def PrintShape(x, message='PrintShape', **unused_kwargs):
  """NoOp layer that prints the shape of the stack."""
  _print_shape(base.shapes(x), message=message)
  return x


def _dup(x):  # pylint: disable=invalid-name
  """Helper: copy the top element of a list or a tuple."""
  if isinstance(x, list):
    return [x[0]] + x
  assert isinstance(x, tuple)
  return tuple([x[0]] + list(x))


@base.layer(output_shape=_dup, stack_items_to_pass=0)
def Dup(x, **unused_kwargs):
  """Duplicate (copy) the first element on the stack."""
  return _dup(x)


def _swap(x):  # pylint: disable=invalid-name
  """Helper: swap the top two elements of a list or a tuple."""
  if isinstance(x, list):
    return [x[1], x[0]] + x[2:]
  assert isinstance(x, tuple)
  return tuple([x[1], x[0]] + list(x[2:]))


@base.layer(output_shape=_swap, stack_items_to_pass=0)
def Swap(x, **unused_kwargs):
  """Swap the first two element on the stack."""
  return _swap(x)


def _top_shape(x_shape):  # pylint: disable=invalid-name
  """Helper: shape of top element of a stack."""
  if isinstance(x_shape[0], (list, tuple)):
    return x_shape[0]
  return x_shape


@base.layer(output_shape=_top_shape, stack_items_to_pass=0)
def _Top(x, **unused_kwargs):
  """Top element from the stack."""
  if isinstance(x, (list, tuple)):
    return x[0]
  return x


def _drop(x):  # pylint: disable=invalid-name
  """Helper: pop top element of a stack (make it a non-list if length is 1)."""
  result = x[1:]
  if len(result) == 1:
    return result[0]
  return result


@base.layer(output_shape=_drop, stack_items_to_pass=0)
def Drop(x, **unused_kwargs):
  """Drop first element from the stack."""
  return _drop(x)


def _flatten_shape(x_shape):  # pylint: disable=invalid-name
  """Helper: shape of the flatten operation."""
  shapes = []
  for shape in x_shape:
    if isinstance(shape[0], (list, tuple)):
      shapes.extend(shape)
    else:
      shapes.append(shape)
  return tuple(shapes)


@base.layer(output_shape=_flatten_shape, stack_items_to_pass=0)
def Flatten(xs, **unused_kwargs):
  """Flatten lists."""
  return tuple(_DeepFlatten(xs))


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

  By default (if no output is given) Select does nothing (NoOp).
  It is also possible to name the inputs to access tuple elements, e.g.:

  Select(inputs=('encoder', ('decoder', 'mask')), output='decoder')

  will transform a tuple (x, (y, x)) into y.

  Args:
    x: the input tuple to re-order.
    params: layer parameters (unused).
    output: the specification of the output tuple: a nested tuple of ints.
    input: the specification of the input tuple if we need to disassemble it.
    **kwargs: other arguments (unused).

  Returns:
    The re-ordered tuple with the same shape as output.
  """

  def __init__(self, output=None, inputs=None):
    super(Select, self).__init__()
    self._output = output
    if inputs is None:
      self._map = lambda x, i: x[i]
    else:
      self._input_map = {}
      self._build_input_map(inputs, [])
      def InputMapping(x, i):
        cur = x
        for idx in self._input_map[i]:
          cur = cur[idx]
        return cur
      self._map = InputMapping

  def _build_input_map(self, inputs, prefix):
    for i, e in enumerate(inputs):
      if isinstance(e, (list, tuple)):
        self._build_input_map(e, prefix + [i])
      else:
        self._input_map[e] = prefix + [i]

  def call(self, x, params=(), **kwargs):
    del params, kwargs
    if self._output is None:
      return x
    return base.nested_map(self._output, lambda i: self._map(x, i))

  def output_shape_fn(self, input_shape):
    if self._output is None:
      return input_shape
    return base.nested_map(self._output, lambda i: self._map(input_shape, i))

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
    layers = _EnsureSublayers(layers)
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

  def output_shape_fn(self, input_shape):
    output_shapes = []
    # If the argument layers are a sequence, apply each to calculate shape.
    if not isinstance(self._layers, dict):
      for layer in self._layers:
        output_shapes.append(layer.output_shape(input_shape))
      return tuple(output_shapes)
    # If layers are a dictionary, apply to the input shape.
    result = {}
    for k in self._layers:
      result[k] = self._layers[k].output_shape(input_shape)
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


def _binary_op(inputs, op):  # pylint: disable=invalid-name
  """Helper: apply op to the first 2 elements."""
  xs, rest = inputs[:2], inputs[2:]
  s = _nested_op(xs, op)
  if not rest:
    return s
  if not isinstance(s, (list, tuple)):
    s = [s]
  res = list(s) + list(rest)
  # TODO(lukaszkaiser): should we drop this tuple/list distinction?
  if isinstance(s, tuple):
    res = tuple(res)
  return res


def _binary_op_shape(stack_shape):  # pylint: disable=invalid-name
  """Helper: shape for the top-two operation above (shape-preserving op)."""
  if len(stack_shape) == 2:
    return stack_shape[0]
  return tuple([stack_shape[0]] + list(stack_shape[2:]))


@base.layer(output_shape=_binary_op_shape, stack_items_to_pass=0)
def Add(x, **unused_kwargs):
  """Add first and second element on the stack."""
  # Here x is a list of tensors of the same shape, or nested structures.
  return _binary_op(x, op=sum)


@base.layer(output_shape=_binary_op_shape, stack_items_to_pass=0)
def Multiply(x, **unused_kwargs):
  """Multiply first and second element on the stack."""
  return _binary_op(x, op=lambda xs: six.moves.reduce(operator.mul, xs))


def _nested_sum(inputs):  # pylint: disable=invalid-name
  return _nested_op(inputs=inputs, op=sum)


def _first_from_tuple_or_dict(tuple_or_dict):  # pylint: disable=invalid-name
  """Helper: return the first element from a tuple or dict."""
  for x in tuple_or_dict:
    return x


@base.layer(output_shape=_first_from_tuple_or_dict, stack_items_to_pass=0)
def AddAll(x, **unused_kwargs):
  """Add branches elementwise."""
  # Here x is a list of tensors of the same shape, or nested structures.
  return _nested_sum(x)


@base.layer(output_shape=_first_from_tuple_or_dict, stack_items_to_pass=0)
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


@base.layer(output_shape=_concatenate_shape, stack_items_to_pass=0)
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
    layers = _EnsureSublayers(layers)
    self._nlayers = len(layers)
    self._layers = layers

  def stack_items_to_pass(self):
    return self._nlayers

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

  def output_shape_fn(self, input_shape):
    output_shapes = []
    # If the argument layers are a sequence, apply each to calculate shape.
    if not isinstance(self._layers, dict):
      for i, layer in enumerate(self._layers):
        output_shapes.append(layer.output_shape(input_shape[i]))
      return tuple(output_shapes)
    # If layers are a dictionary, apply to matching keys in the input shape.
    result = {}
    for k in input_shape:
      if k in self._layers:
        result[k] = self._layers[k].output_shape(input_shape[k])
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
  shortcut = kwargs.get('shortcut', _Top())  # pylint: disable=no-value-for-parameter
  return [
      Branch(shortcut, Serial(layers)),  # Use Serial here to flatten layers.
      Flatten(),  # pylint: disable=no-value-for-parameter
      Add(),  # pylint: disable=no-value-for-parameter
  ]


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

  def output_shape_fn(self, input_shapes):
    return tuple([self._layer.output_shape(shape) for shape in input_shapes])

  def new_parameters(self, input_shape, rng):
    first_shape = input_shape[0]
    if self._check_shapes:
      for shape in input_shape:
        if shape != first_shape:
          raise ValueError('Map layer can only be applied to list of elements '
                           'with the same shapes. Shapes: %s' % str(shape))
    return self._layer.initialize(first_shape, rng)


class Rebatch(base.Layer):
  """Combinator for treating the first `n` dims as batch.

  Args:
    layer: subclass of base.Layer, a layer to apply to the input.
    n_batch_dims: int, the number of leading dimensions to consider as batch.

  Returns:
    A new layer that will reshape the input into a virtual batch, apply the
    layer and unbatch the virtual batch.
  """

  def __init__(self, layer, n_batch_dims=1):
    super(Rebatch, self).__init__()
    self._layer = layer
    self._n_batch_dims = n_batch_dims

  def _modify_shape(self, input_shape):
    input_shape = tuple(input_shape)
    batch_dims, non_batch_dims = (input_shape[:self._n_batch_dims],
                                  input_shape[self._n_batch_dims:])
    new_batch_dim = six.moves.reduce(operator.mul, batch_dims)
    return (new_batch_dim,) + non_batch_dims, batch_dims

  def _unmodify_shape(self, input_shape, batch_dims):
    return batch_dims + tuple(input_shape[1:])

  def _modify(self, inp):
    modified_shape, batch_dims = self._modify_shape(inp.shape)
    return np.reshape(inp, modified_shape), batch_dims

  def _unmodify(self, inp, batch_dims):
    return np.reshape(inp, self._unmodify_shape(inp.shape, batch_dims))

  def call(self, inp, params=(), **kwargs):
    if isinstance(inp, (tuple, list)):
      # TODO(afrozm): This should be easy to do though.
      # Tip from Lukasz - base.nested_map(self._modify, inp)
      raise ValueError("Rebatch doesn't support list/tuple inputs now.")
    inp, batch_dims = self._modify(inp)
    out = self._layer(inp, params=params, **kwargs)
    return self._unmodify(out, batch_dims)

  def output_shape_fn(self, input_shape):
    modified_shape, batch_dims = self._modify_shape(input_shape)
    out = self._layer.output_shape(modified_shape)
    return self._unmodify_shape(out, batch_dims)

  def new_parameters(self, input_shape, rng):
    modified_shape, _ = self._modify_shape(input_shape)
    return self._layer.initialize(modified_shape, rng)
