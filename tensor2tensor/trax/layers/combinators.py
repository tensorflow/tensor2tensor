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
from tensor2tensor.trax.backend import numpy as np
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


def _deep_flatten(items):  # pylint: disable=invalid-name
  """Returns a list of objects, flattening sublists/subtuples along the way.

  Example: _deep_flatten([1, (2, 3, (4, 5), [6, 7]), [[[8]]]]) would return
  the list [1, 2, 3, 4, 5, 6, 7, 8].

  Args:
    items: An iterable. If elements of this iterable are lists or tuples, they
        will be (recursively) flattened until non-list non-tuple objects are
        reached.

  Returns:
    A list of non-list, non-tuple objects.
  """
  def _flat_gen(xs):  # pylint: disable=invalid-name
    for x in xs:
      if isinstance(x, (list, tuple)):
        for y in _flat_gen(x):
          yield y
      else:
        yield x
  return list(_flat_gen(items))


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
    return Serial(None)  # no-op, but still handles shapes and initialization
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


def _count_items(xs):  # pylint: disable=invalid-name
  return len(xs) if isinstance(xs, (list, tuple)) else 1


class Serial(base.Layer):
  """Combinator that applies layers serially (by function composition).

  A Serial combinator uses stack semantics to manage data for its sublayers.
  Each sublayer sees only the inputs it needs and returns only the outputs it
  has generated. The sublayers interact via the data stack. For instance, a
  sublayer k, following sublayer j, gets called with the data stack in the
  state left after layer j has applied. The Serial combinator then:

    - takes N_in items off the top of the stack (N_in = k.n_inputs) and calls
      layer k, passing those items as arguments; and

    - takes layer k's N_out return values (N_out = k.n_outputs) and pushes
      them onto the data stack.
  """

  def __init__(self, *layers):
    super(Serial, self).__init__()

    layers = self._ensure_flat(layers)
    self._sublayers = layers
    self._n_layers = len(layers)

    if not layers:
      self._n_inputs = 1
      self._n_outputs = 1
    else:
      self._n_inputs, self._n_outputs = self._n_inputs_n_outputs(layers)

  def _ensure_flat(self, layers):
    """Ensures that layers is a single flat list of Layer instances."""
    del self
    if len(layers) == 1 and layers[0] is None:
      layers = []
    else:
      layers = _deep_flatten(layers)
    for obj in layers:
      if not isinstance(obj, base.Layer):
        raise ValueError(
            'Found nonlayer object ({}) in layers: {}.'.format(obj, layers))
    return layers

  def _n_inputs_n_outputs(self, layers):
    del self
    running_max = 0
    running_total = 0
    for layer in layers:
      running_total += layer.n_inputs
      running_max = max(running_max, running_total)
      running_total -= layer.n_outputs
    return running_max, (running_max - running_total)

  def _validate_call_inputs(self, xs):
    if not isinstance(xs, tuple) and self._n_inputs != 1:
      raise TypeError(
          'Serial.call input must be a tuple; instead got {}'.format(xs))
    len_xs = 1 if isinstance(xs, np.ndarray) else len(xs)
    if len_xs < self.n_inputs:
      raise ValueError(
          'number of inputs ({}) to Serial.call less than n_inputs'
          ' ({})'.format(len(xs), self.n_inputs))

  def call(self, xs, params=(), state=(), **kwargs):
    self._validate_call_inputs(xs)
    rngs = _pop_rng_and_split(kwargs, self._n_layers)
    if not self.sublayers:  # No-op: leave args unchanged.
      return (xs, state)

    stack = xs
    new_state = []
    n_layers = self._n_layers
    if n_layers != 1 and len(params) != n_layers:
      raise ValueError('number of params ({}) not equal to number of layers '
                       '({})'.format(len(params), n_layers))
    if n_layers != 1 and len(state) != n_layers:
      raise ValueError('length of state ({}) not equal to number of layers '
                       '({})'.format(len(state), n_layers))
    for layer, p, s, rng in zip(self.sublayers, params, state, rngs):
      is_stack_just_one_item = (_count_items(stack) == 1)

      # Give layer its args from the stack; treat 1-arg layer specially.
      n_in = layer.n_inputs
      if n_in == 1 and is_stack_just_one_item:
        inputs = stack
      elif n_in == 1:
        inputs = stack[0]
      else:
        inputs = stack[:n_in]
      outputs, s = layer(inputs, p, state=s, rng=rng, **kwargs)
      new_state.append(s)

      # Push outputs onto remaining stack (if any).
      if n_in < _count_items(stack):
        if layer.n_outputs == 1:
          outputs = (outputs,)
        stack = outputs + stack[n_in:]
      else:
        stack = outputs  # NOTE: can be single value or tuple.

    return stack, new_state

  def new_parameters(self, input_shape, input_dtype, rng):
    def MakeShapeType(shape, dtype):
      if isinstance(dtype, (list, tuple)):
        return tuple(MakeShapeType(s, t) for s, t in zip(shape, dtype))
      return base.ShapeType(shape=shape, dtype=dtype)

    params = []
    states = []
    pseudo_xs = MakeShapeType(input_shape, input_dtype)
    for layer in self.sublayers:
      rng, layer_rng = backend.random.split(rng)

      # Give layer its args from pseudo_xs; treat 1-arg layer specially.
      is_stack_just_one_item = (_count_items(pseudo_xs) == 1)
      n_in = layer.n_inputs
      if n_in == 1 and is_stack_just_one_item:
        inputs = pseudo_xs
      elif n_in == 1:
        inputs = pseudo_xs[0]
      else:
        inputs = pseudo_xs[:n_in]

      in_shape = base.nested_map(inputs, lambda x: x.shape)
      in_dtype = base.nested_map(inputs, lambda x: x.dtype)
      param, state = layer.initialize(in_shape, in_dtype, layer_rng)
      pparam = layer._params   # pylint: disable=protected-access

      outputs, _ = layer.pseudo_call(inputs, pparam, state)

      # Push outputs onto remaining pseudo_xs (if any).
      if n_in < _count_items(pseudo_xs):
        if layer.n_outputs == 1:
          outputs = (outputs,)
        pseudo_xs = outputs + pseudo_xs[n_in:]
      else:
        pseudo_xs = outputs  # NOTE: can be single value or tuple.

      params.append(param)
      states.append(state)
    return params, states


@base.layer(n_outputs=2)
def Dup(x, **unused_kwargs):
  """Duplicates (copies) an element."""
  return (x, x)


@base.layer(n_inputs=2, n_outputs=2)
def Swap(xs, **unused_kwargs):
  """Swaps two elements."""
  return (xs[1], xs[0])


@base.layer(n_outputs=0)
def Drop(x, **unused_kwargs):
  """Drops one element."""
  del x  # Just for the compiler.
  return ()


@base.layer(n_inputs=0)
def FlattenList(xs, **unused_kwargs):
  """Flatten lists."""
  # TODO(jonni): Consider renaming layer to DeepFlatten.
  return tuple(_deep_flatten(xs))


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


@base.layer(n_inputs=2)
def Add(xs, **unused_kwargs):
  """Adds two tensors."""
  return xs[0] + xs[1]


@base.layer(n_inputs=2)
def SubtractTop(xs, **unused_kwargs):
  """Subtracts the first tensor from the second."""
  return xs[1] - xs[0]


@base.layer(n_inputs=2)
def Multiply(xs, **unused_kwargs):
  """Multiplies two tensors."""
  return xs[0] * xs[1]


@base.layer(n_inputs=3)
def Gate(xs, **unused_kwargs):
  """Implements a gating function on a (memory, gate, candidate) tuple.

  Final update is memory * gate + (1-gate) * candidate

  This gating equation may also be referred to as Highway Network.
  Highway Networks: https://arxiv.org/abs/1505.00387

  Args:
    xs: A tuple of memory, gate, candidate

  Returns:
    The result of applying gating.
  """
  state, gate, candidate = xs
  return gate * state + (1.0 - gate) * candidate


class Concatenate(base.Layer):
  """Concatenates n tensors into a single tensor."""

  def __init__(self, n_items=2, axis=-1):
    super(Concatenate, self).__init__(n_inputs=n_items)
    self._n_items = n_items
    self._axis = axis

  def new_parameters(self, input_shape, input_dtype, rng):
    return (), ()

  def call(self, xs, params=(), state=(), **kwargs):
    del params, kwargs
    return backend.numpy.concatenate(xs, self._axis), state


class Parallel(base.Layer):
  """Combinator that applies a list of layers in parallel to its inputs.

  Layers in the list apply to successive spans of inputs, where the spans are
  determined how many inputs each layer takes. The resulting output is the
  (flattened) concatenation of the resepective layer outputs.

  For example, suppose one has three layers:

    - F: 1 input, 1 output
    - G: 3 inputs, 1 output
    - H: 2 inputs, 2 outputs (h1, h2)

  Then Parallel(F, G, H) will take 6 inputs and give 4 outputs:

    - inputs: a, b, c, d, e, f
    - outputs: F(a), G(b, c, d), h1, h2

  As an important special case, a None argument to Parallel acts as if it takes
  one argument, which it leaves unchanged. (It acts as a one-arg no-op.) For
  example:

    Parallel(None, F)

  creates a layer that passes its first input unchanged and applies F to the
  following input(s).
  """

  def __init__(self, *layers):
    """The constructor.

    Args:
      *layers: A list of layers.

    Returns:
      A new layer in which each of the given layers applies to its corresponding
      span of elements in the dataflow stack.
    """
    super(Parallel, self).__init__()
    layers = self._validate(layers)
    self._n_layers = len(layers)
    self._sublayers = layers
    self._n_inputs = sum(x.n_inputs for x in layers)
    self._n_outputs = sum(x.n_outputs for x in layers)

  def _validate(self, layers):
    if not layers or len(layers) < 2:
      raise ValueError(
          'layers ({}) must be a list with at least two elements'.format(
              layers))
    layers = list(layers)  # Ensure we can modify layers.
    for i, obj in enumerate(layers):
      if obj is None or obj == []:  # pylint: disable=g-explicit-bool-comparison
        layers[i] = Serial(None)
      elif isinstance(obj, (list, tuple)):
        layers[i] = Serial(obj)
      else:
        if not isinstance(obj, base.Layer):
          raise ValueError(
              'Found nonlayer object ({}) in layers list: [{}].'.format(
                  obj, layers))
      if layers[i].n_inputs == 0:
        raise ValueError(
            'Sublayer with n_inputs = 0 not allowed in Parallel:'
            ' {}'.format(layers[i]))
    return layers

  def _allot_to_sublayers(self, inputs):
    """Divides Parallel's inputs for use by the sublayers.

    Args:
      inputs: Tuple of elements.

    Returns:
      A tuple that partitions this layer's inputs among its sublayers.
      Sublayers that take one argument get that argument directly. All other
      sublayers get a tuple of items.
    """
    start, end = 0, 0
    sub_inputs = []
    for layer in self.sublayers:
      n_in = layer.n_inputs
      end = start + n_in
      if n_in == 1:
        sub_inputs.append(inputs[start])
      else:
        sub_inputs.append(inputs[start:end])
      start = end
    return tuple(sub_inputs)

  def call(self, inputs, params=(), state=(), **kwargs):
    n_layers, layers = self._n_layers, self.sublayers
    sublayer_inputs = self._allot_to_sublayers(inputs)
    rngs = _pop_rng_and_split(kwargs, n_layers)
    assert len(sublayer_inputs) == n_layers
    assert len(params) == n_layers
    assert len(state) == n_layers
    assert len(rngs) == n_layers
    outputs = []
    new_state = []
    for layer, x, p, s, r in zip(layers, sublayer_inputs, params, state, rngs):
      # Note that zip silently truncates its result if lengths don't match.
      sub_outputs, s = layer(x, params=p, state=s, rng=r, **kwargs)
      if layer.n_outputs == 1:
        outputs.append(sub_outputs)
      else:
        outputs.extend(sub_outputs)
      new_state.append(s)
    output = outputs[0] if self.n_outputs == 1 else tuple(outputs)
    return output, new_state

  def new_parameters(self, input_shapes, input_dtypes, rng):
    sublayer_shapes = self._allot_to_sublayers(input_shapes)
    sublayer_dtypes = self._allot_to_sublayers(input_dtypes)
    rngs = backend.random.split(rng, self._n_layers)
    inits = [layer.initialize(shape, dtype, rng) for layer, shape, dtype, rng
             in zip(self.sublayers, sublayer_shapes, sublayer_dtypes, rngs)]
    if not inits:
      return (), ()
    else:
      return tuple(zip(*inits))


def Residual(*layers, **kwargs):
  """Constructs a residual version of layers, summing input to layers output."""
  shortcut = kwargs.get('shortcut')  # default None signals no-op
  return [
      Dup(),  # pylint: disable=no-value-for-parameter
      Parallel(shortcut, layers),
      Add(),  # pylint: disable=no-value-for-parameter
  ]
