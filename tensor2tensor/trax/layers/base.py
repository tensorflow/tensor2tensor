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

"""Base layer class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import traceback

import jax
from jax.interpreters import partial_eval as pe

import numpy as onp
from tensor2tensor.trax import backend


class Layer(object):
  """Layer object, base class. Handles parameter sharing."""

  def __init__(self, **kwargs):
    # We store kwargs by default, used below in creating a generic decorator.
    self._init_kwargs = kwargs
    # This field says if this layer's init has already been called or not.
    self._first_init = True
    # Cache parameters here, defaults empty params (we use () for that).
    self._params = ()  # cached parameters
    # Caller field storing info on where the caller class was created.
    self._caller = _find_frame(inspect.stack())

  def call(self, x, params=(), **kwargs):
    """Call this layer in input x using the given parameters."""
    raise NotImplementedError

  def new_parameters(self, input_shape, rng):
    """Create new parameters for the layer given an input shape and rng.

    Note that all arguments and return values can be tuples or dictionaries
    or arbitraty nested structures composed of tuples and dictionaries.

    Args:
      input_shape: a tuple representing the shape of the input.
      rng: random number generator.

    Returns:
      The newly created parameters for this layer.
    """
    raise NotImplementedError

  # TODO(lukaszkaiser): re-visit the 2 items below in the future.
  def stack_items_to_pass(self):
    """How many of the top stack items do we process."""
    return 0

  def default_input_is_int(self):
    """Whether the default inputs are ints or floats."""
    return False

  # End of subclassing interface, all functions below are internal.

  def output_shape(self, input_shape_and_type, params):
    """Output shape and type for this layer given input shape and type.

    Note that all arguments and return values can be tuples or dictionaries
    or arbitrary nested structures composed of tuples and dictionaries.

    Args:
      input_shape_and_type: a ShapeType with shape and type of the input.
      params: parameters for this layer.

    Returns:
      The shape and type of the output.
    """
    try:
      with backend.use_backend('jax'):
        rng = backend.random.get_prng(0)
        def call_on_input(x, params):
          f = lambda y: self.call(y, params=params, rng=rng)
          n = self.stack_items_to_pass() if isinstance(x, (list, tuple)) else 0
          return _apply_to_first_n(f, x, n)
        params_shapes = nested_map(
            params, lambda x: ShapeType(shape=x.shape, tp=x.dtype))
        s = _eval_on_shapes(call_on_input, input_shape_and_type, params_shapes)
      return s
    except Exception:
      name, trace = self.__class__.__name__, _short_traceback(skip=3)
      raise LayerError(name, 'output_shape', self._caller,
                       input_shape_and_type, trace)

  def initialize(self, input_shape, rng):
    """Initialize the layer given an input shape and rng.

    Returns new_parameters(input_shape, rng) on the first call and () on any
    subsequent call, as the layer is already initialized. This is used for
    networks that share parameters, so the layer only produces them once.

    Note that all arguments and return values can be tuples or dictionaries
    or arbitraty nested structures composed of tuples and dictionaries.

    Args:
      input_shape: a tuple representing the shape of the input.
      rng: random number generator.

    Returns:
      Newly created parameters on the first call and () on all subsequent calls.
    """
    try:
      # Re-using this layer, no new parameters.
      if not self._first_init:
        return ()

      # First call of this layer, create parameters.
      self._first_init = False
      is_list = isinstance(input_shape, (list, tuple))
      is_list = is_list and isinstance(input_shape[0], (list, tuple))
      if is_list and self.stack_items_to_pass() > 0:
        input_shape = input_shape[:self.stack_items_to_pass()]
        if len(input_shape) == 1:
          input_shape = input_shape[0]
      self._params = self.new_parameters(input_shape, rng)
      return self._params
    except Exception:
      name, trace = self.__class__.__name__, _short_traceback(skip=3)
      raise LayerError(name, 'initialize', self._caller, input_shape, trace)

  def __call__(self, x, params=(), **kwargs):
    try:
      # If params are nothing, we may be reusing this layer.
      # Use the cached parameters to calculate the value.
      # Note: to make sure jit tracers can decide this branch in python we
      #   use "params is ()" instead of, e.g., "not params" or "params == ()".
      if params is ():  # pylint: disable=literal-comparison
        params = self._params
      # In this case, we're called for the first time: cache parameters.
      self._params = params
      f = lambda y: self.call(y, params=params, **kwargs)
      n = self.stack_items_to_pass() if isinstance(x, (list, tuple)) else 0
      return _apply_to_first_n(f, x, n)
    except Exception:
      name, trace = self.__class__.__name__, _short_traceback()
      raise LayerError(name, 'call', self._caller, shapes(x), trace)


class ShapeType(object):
  """Store shape and type."""

  def __init__(self, shape, tp):
    self.shape = shape
    self.tp = tp

  def __repr__(self):
    return '[shape:' + str(self.shape) + ', type:' + str(self.tp) + ']'


class LayerError(Exception):
  """Exception raised in the layer stack.

  Attributes:
    message: the message corresponding to this exception.
  """

  def __init__(self, layer_name, function_name, caller,
               input_shapes, traceback_string):
    self._layer_name = layer_name
    self._function_name = function_name  # Is it call or initialize?
    self._caller = caller  # Python inspect object with init caller info.
    self._traceback = traceback_string
    self._input_shapes = input_shapes
    super(LayerError, self).__init__(self.message)

  @property
  def message(self):
    prefix = 'Exception passing through layer '
    prefix += '%s (in %s):\n' % (self._layer_name, self._function_name)
    short_path = '[...]/' + '/'.join(self._caller.filename.split('/')[-3:])
    caller = '  layer created in file %s, line %d\n' % (short_path,
                                                        self._caller.lineno)
    shapes_str = '  layer input shapes: %s\n\n' % str(self._input_shapes)
    return prefix + caller + shapes_str + self._traceback


# TODO(lukaszkaiser): remove this function once JAX has an analogue.
def _eval_on_shapes(f, *args):
  """Evaluate f given only shapes and types."""
  def abstractify(x):
    return jax.abstract_arrays.raise_to_shaped(jax.core.get_aval(x))

  def make_array(arg):
    return backend.numpy.zeros(shape=arg.shape, dtype=arg.tp)

  def turn_back_into_pytree(x):
    if isinstance(x, jax.core.JaxTuple):
      return tuple([turn_back_into_pytree(y) for y in x])
    return x

  def get_shapes_and_types(x):
    if isinstance(x, jax.core.AbstractTuple):
      return tuple([get_shapes_and_types(y) for y in x])
    return ShapeType(x.shape, x.dtype)

  def f_jaxtuple(*jaxtuple_args):
    args = map(turn_back_into_pytree, jaxtuple_args)
    out = f(*args)
    res, _ = jax.api_util.pytree_to_jaxtupletree(out)
    return res

  args_arrays = nested_map(args, make_array)
  jaxtuple_args, _ = jax.util.unzip2(
      map(jax.api_util.pytree_to_jaxtupletree, args_arrays))
  res = pe.abstract_eval_fun(f_jaxtuple, *map(abstractify, jaxtuple_args))

  return get_shapes_and_types(res)


def _apply_to_first_n(f, x, n):
  """Helper: apply f to first n elements on the stack x if n > 0."""
  if n < 1:
    return f(x)
  argument, rest = x[:n], x[n:]
  if n == 1:
    argument = argument[0]
  result = f(argument)
  if not rest:
    return result
  if n == 1:
    result = [result]
  result = list(result) + list(rest)
  if isinstance(x, tuple):
    result = tuple(result)
  return result


def nested_map(x, f):
  """Map the function f to the nested structure x (dicts, tuples, lists)."""
  if isinstance(x, list):
    return [nested_map(y, f) for y in x]
  if isinstance(x, tuple):
    return tuple([nested_map(y, f) for y in x])
  if isinstance(x, dict):
    return {k: nested_map(x[k], f) for k in x}
  return f(x)


def nested_reduce(x, f):
  """Fold the function f to the nested structure x (dicts, tuples, lists)."""
  if isinstance(x, list):
    return f([nested_reduce(y, f) for y in x])
  if isinstance(x, tuple):
    return f([nested_reduce(y, f) for y in x])
  if isinstance(x, dict):  # We apply f only to values in the dicts.
    return f([nested_reduce(v, f) for v in x.values()])
  return x


def shapes(x):
  """Get a structure of shapes for a structure of nested arrays."""
  def shape(x):
    try:
      return tuple([int(i) for i in x.shape])
    except Exception:  # pylint: disable=broad-except
      return []
  return nested_map(x, shape)


def sizes(x):
  """Get a structure of sizes for a structure of nested arrays."""
  def size(x):
    try:
      return x.size
    except Exception:  # pylint: disable=broad-except
      return 0
  return nested_map(x, size)


def _find_frame(stack, start=0):
  """Find the frame with the caller on the stack."""
  # We want to find the first place where the layer was called
  # that is *not* an __init__ function of an inheriting layer.
  frame = inspect.getframeinfo(stack[start][0])
  # If we are in an init, move on.
  if frame.function == '__init__':
    return _find_frame(stack, start + 1)
  return frame


def _shorten_file_path(line):
  """Shorten file path in error lines for more readable tracebacks."""
  start = line.lower().find('file')
  if start < 0:
    return line
  first_quote = line.find('"', start)
  if first_quote < 0:
    return line
  second_quote = line.find('"', first_quote + 1)
  if second_quote < 0:
    return line
  path = line[first_quote + 1:second_quote]
  new_path = '/'.join(path.split('/')[-3:])
  return line[:first_quote] + '[...]/' + new_path + line[second_quote + 1:]


def _short_traceback(skip=7):
  """Cleaned-up form of traceback."""
  counter, res = 0, []
  # Skipping 3 lines by default: the top (useless) and self-call.
  lines = traceback.format_exc().splitlines()[skip:]
  for l in lines:
    res.append(_shorten_file_path(l))
    if counter % 2 == 1:
      res.append('')
    counter += 1
    # If we see a LayerError, the traceback has already been processed.
    if l.startswith('LayerError'):
      # Skip 4 back except last as these are internal base-layer calls.
      res = res[:-4] + [res[-1]]
      res += lines[counter:]
      break
  return '\n'.join(res)


# Decorator for making layers from functions.


def layer(new_parameters=None, stack_items_to_pass=1, input_is_int=False):
  """Create a layer class from a function."""
  def layer_decorator(call):
    """Decorating the call function."""

    def stack_items_to_pass_fn(self):
      del self
      return stack_items_to_pass

    def default_input_is_int_fn(self):
      del self
      return input_is_int

    def new_parameters_fn(self, input_shape, rng):
      if new_parameters is None:
        return ()
      kwargs = self._init_kwargs  # pylint: disable=protected-access
      return new_parameters(input_shape, rng, **kwargs)

    def call_fn(self, x, params=(), **kwargs):
      """The call function of the created class, derived from call."""
      # Merge on-call kwargs with class-kwargs.
      call_kwargs = kwargs.copy()
      call_kwargs.update(self._init_kwargs)  # pylint: disable=protected-access
      # Call with the merged kwargs.
      return call(x, params=params, **call_kwargs)

    # Set doc for python help.
    call_fn.__doc__ = call.__doc__
    if new_parameters is None:
      new_parameters_fn.__doc__ = new_parameters.__doc__

    # Create the class.
    cls = type(call.__name__, (Layer,),
               {'call': call_fn,
                'default_input_is_int': default_input_is_int_fn,
                'new_parameters': new_parameters_fn,
                'stack_items_to_pass': stack_items_to_pass_fn})

    return cls
  return layer_decorator


def _random_inputs(input_shape, rng, integer_inputs=False):
  """Create random floats of the given shape.

  Args:
    input_shape: Could be either:
        list/tuple of ints, ex: (210, 160, 3) or
        list/tuple of nested shapes, ex: [(210, 160, 3), (105, 80, 3)] or
        dictionary of nested shapes, ex: {"obs": [(28, 28, 1), (4,)],
                                          "sensors": [(3,4), (4, 9)]} or
        any other combination of these, ex: list of dictionaries of tuples etc.
    rng: random number generator.
    integer_inputs: boolean, True if we want arrays of integers, otherwise we
        produce float32s.

  Returns:
    Random values of the type and shape specified.
  """
  if not isinstance(input_shape, dict) and isinstance(input_shape[0], int):
    # Non-nested shape, create a random tuple.
    if not integer_inputs:
      return backend.random.uniform(rng, input_shape, minval=-1.0, maxval=1.0)
    return backend.random.bernoulli(rng, 0.5, input_shape).astype(onp.int32)
  elif isinstance(input_shape, list):  # Nested shape: list.
    return [_random_inputs(shape, rng, integer_inputs) for shape in input_shape]
  elif isinstance(input_shape, tuple):  # Nested shape: tuple.
    return tuple(_random_inputs(list(input_shape), rng, integer_inputs))
  elif isinstance(input_shape, dict):  # Nested shape: dict.
    return {k: _random_inputs(input_shape[k], rng, integer_inputs)
            for k in input_shape}
  else:
    raise TypeError(type(input_shape))


def to_shape_and_type(x_shapes, integers):
  """Make a shape-and-type tuple from shapes."""
  if isinstance(x_shapes, dict):  # Nested shape: dict.
    return {k: to_shape_and_type(x_shapes[k], integers) for k in x_shapes}
  if isinstance(x_shapes, onp.ndarray):  # Numpy array shape
    return ShapeType(shape=x_shapes.tolist(),
                     tp=onp.int32 if integers else onp.float32)
  if isinstance(x_shapes[0], (int, onp.int32, onp.int64)):
    return ShapeType(shape=x_shapes,
                     tp=onp.int32 if integers else onp.float32)
  if isinstance(x_shapes, list):  # Nested shape: list.
    return [to_shape_and_type(s, integers) for s in x_shapes]
  if isinstance(x_shapes, tuple):  # Nested shape: tuple.
    return tuple([to_shape_and_type(s, integers) for s in x_shapes])
  assert False  # Should never get here.


def check_shape_agreement(layer_instance, input_shape, integer_inputs=False):
  """Check if layer.output_shape agrees with the actual output shape."""
  rng1, rng2, rng3 = backend.random.split(backend.random.get_prng(0), 3)
  params = layer_instance.initialize(input_shape, rng1)
  input_shape_and_type = to_shape_and_type(input_shape, integer_inputs)
  output_shape_and_type = layer_instance.output_shape(
      input_shape_and_type, params)
  output_shape = nested_map(output_shape_and_type, lambda x: x.shape)
  output_shape = nested_map(output_shape, int)  # Make non-numpy.
  inputs = _random_inputs(input_shape, rng2, integer_inputs=integer_inputs)
  result = layer_instance(inputs, params, rng=rng3)
  result_shape = shapes(result)
  msg = 'output shape %s != real result shape %s' % (output_shape, result_shape)
  assert output_shape == result_shape, msg
  return output_shape
