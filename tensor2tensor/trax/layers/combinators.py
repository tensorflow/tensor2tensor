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


def Model(*layers):
  """Ensures that a layer or list of layers can be treated as a model.

  Currently, any subclass of base.Layer can be treated as a model.

  Args:
    *layers: One or more layer objects. In fuller detail, the list may contain
        nested sublists, and the top-level list can also be a tuple.

  Returns:
    A single object that treated as a model, e.g., trained or evaluated.
  """
  return Serial(*layers)


def _deep_flatten(xs):  # pylint: disable=invalid-name
  for x in xs:
    if isinstance(x, (list, tuple)):
      for y in _deep_flatten(x):
        yield y
    else:
      yield x


def _ensure_sublayers(layers):  # pylint: disable=invalid-name
  """Ensures that elements in a layer list are layers.

  Args:
    layers: A tuple or list whose elements can each be a layer, tuple, or list,
        and so on recursively.

  Returns:
    An analogous collection of layers in which embedded layer lists are
    wrapped in Serial layer instances.
  """
  if not layers:  # None or an empty list can signal a no-op.
    return Serial([])  # no-op, but still handles shapes and initialization
  elif isinstance(layers, (list, tuple)):
    sublayers_not_lists = []
    for layer in layers:
      sublayers_not_lists.append(
          Serial(layer) if isinstance(layer, (list, tuple)) else layer)
    return sublayers_not_lists
  else:
    raise TypeError(type(layers))


def _pop_rng_and_split(args_dict, n_copies):  # pylint: disable=invalid-name
  rng = args_dict.pop('rng', None)
  if rng is None:
    return (None,) * n_copies
  return backend.random.split(rng, n_copies)


class Serial(base.Layer):
  """Layer composing a number of sub-layers in a serial way.."""

  def __init__(self, *layers):
    super(Serial, self).__init__()
    layers = list(_deep_flatten(layers))
    # TODO(jonni): Consider flattening (unpacking) also embedded Serial layers.
    self._layers = layers
    self._nlayers = len(layers)

  def call(self, x, params=(), **kwargs):
    rngs = _pop_rng_and_split(kwargs, self._nlayers)
    for layer, p, rng in zip(self._layers, params, rngs):
      x = layer(x, p, rng=rng, **kwargs)
    return x

  def new_parameters(self, input_shape, input_dtype, rng):
    def MakeShapeType(shape, dtype):
      if isinstance(dtype, (list, tuple)):
        return tuple(MakeShapeType(s, t) for s, t in zip(shape, dtype))
      return base.ShapeType(shape=shape, dtype=dtype)
    params = []
    pseudo_data = MakeShapeType(input_shape, input_dtype)
    for layer in self._layers:
      rng, layer_rng = backend.random.split(rng)
      cur_shape = base.nested_map(pseudo_data, lambda x: x.shape)
      cur_dtype = base.nested_map(pseudo_data, lambda x: x.dtype)
      param = layer.initialize(cur_shape, cur_dtype, layer_rng)
      pparam = layer._params   # pylint: disable=protected-access
      pseudo_data = layer.pseudo_call(pseudo_data, pparam)
      params.append(param)
    return params


@base.layer(stack_items_to_pass=0)
def PrintShape(x, message='PrintShape', **unused_kwargs):
  """No-op layer that prints the shape of the stack."""
  print(message + ' ; stack shape = ' + str(base.shapes(x)))
  return x


@base.layer(stack_items_to_pass=0)
def Dup(x, **unused_kwargs):
  """Duplicate (copy) the first element on the stack."""
  if isinstance(x, list):
    return [x[0]] + x
  if isinstance(x, tuple):
    return tuple([x[0]] + list(x))
  return [x, x]


@base.layer(stack_items_to_pass=0)
def Swap(x, **unused_kwargs):
  """Swap the first two element on the stack."""
  if isinstance(x, list):
    return [x[1], x[0]] + x[2:]
  assert isinstance(x, tuple)
  return tuple([x[1], x[0]] + list(x[2:]))


@base.layer(stack_items_to_pass=0)
def _Top(x, **unused_kwargs):
  """Top element from the stack."""
  if isinstance(x, (list, tuple)):
    return x[0]
  return x


@base.layer(stack_items_to_pass=0)
def Drop(x, **unused_kwargs):
  """Drop first element from the stack."""
  result = x[1:]
  if len(result) == 1:
    return result[0]
  return result


@base.layer(stack_items_to_pass=0)
def FlattenList(xs, **unused_kwargs):
  """Flatten lists."""
  return tuple(_deep_flatten(xs))


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

  By default (if no output is given) Select does nothing. It is also possible
  to name the inputs to access tuple elements, e.g.:

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

  def new_parameters(self, input_shape, input_dtype, rng):
    return ()


class Branch(base.Layer):
  """Combinator for applying layers to copies of the input.

  This layer is often used to create parallel towers in neural networks:
  * Branch(main, shortcut) -- start a residual tower (see Residual below)

  Args:
    *layers: a sequence of layers.

  Returns:
    A new layer in which each of the given layers has been applied to
    a copy of the input independently.
  """

  def __init__(self, *layers):
    super(Branch, self).__init__()
    layers = _ensure_sublayers(layers)
    self._nlayers = len(layers)
    self._layers = layers

  def call(self, x, params=(), **kwargs):
    rngs = _pop_rng_and_split(kwargs, self._nlayers)
    if isinstance(self._layers, (list, tuple)):
      res = [layer(x, params=p, rng=r, **kwargs)
             for layer, p, r in zip(self._layers, params, rngs)]
      return tuple(res)

  def new_parameters(self, input_shape, input_dtype, rng):
    rngs = backend.random.split(rng, self._nlayers)
    return [layer.initialize(input_shape, input_dtype, rng)
            for layer, rng in zip(self._layers, rngs)]


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


@base.layer(stack_items_to_pass=0)
def Add(x, **unused_kwargs):
  """Add first and second element on the stack."""
  # Here x is a list of tensors of the same shape, or nested structures.
  return _binary_op(x, op=sum)


@base.layer(stack_items_to_pass=0)
def SubtractTop(x, **unused_kwargs):
  """Subtract the first element on the stack from the second element."""
  # Here x is a list of tensors of the same shape, or nested structures.
  return _binary_op(x, op=lambda xs: xs[1] - xs[0])


@base.layer(stack_items_to_pass=0)
def Multiply(x, **unused_kwargs):
  """Multiply first and second element on the stack."""
  return _binary_op(x, op=lambda xs: six.moves.reduce(operator.mul, xs))


@base.layer(stack_items_to_pass=0)
def AddAll(x, **unused_kwargs):
  """Add branches elementwise."""
  # Here x is a list of tensors of the same shape, or nested structures.
  return _nested_op(x, op=sum)


@base.layer(stack_items_to_pass=0)
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


@base.layer(stack_items_to_pass=0)
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
    layers = _ensure_sublayers(layers)
    self._nlayers = len(layers)
    self._layers = layers

  def stack_items_to_pass(self):
    return self._nlayers

  def call(self, inputs, params=(), **kwargs):
    rngs = _pop_rng_and_split(kwargs, self._nlayers)
    # Note that zip silently truncates its result if lengths don't match.
    assert len(inputs) == self._nlayers
    assert len(params) == self._nlayers
    assert len(rngs) == self._nlayers
    return tuple(layer(x, params=p, rng=r, **kwargs)
                 for layer, x, p, r in zip(self._layers, inputs, params, rngs))

  def new_parameters(self, input_shape, input_dtype, rng):
    rngs = backend.random.split(rng, self._nlayers)
    return [layer.initialize(shape, dtype, rng) for layer, shape, dtype, rng
            in zip(self._layers, input_shape, input_dtype, rngs)]


def Residual(*layers, **kwargs):
  """Constructs a residual version of layers, summing input to layers output."""
  shortcut = kwargs.get('shortcut', _Top())  # pylint: disable=no-value-for-parameter
  return [
      Branch(shortcut, Serial(layers)),  # Use Serial here to flatten layers.
      FlattenList(),  # pylint: disable=no-value-for-parameter
      Add(),  # pylint: disable=no-value-for-parameter
  ]
