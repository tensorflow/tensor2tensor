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

  def output_shape(self, input_shape):
    """The shape of the output of this layer given the shape of the input.

    Note that all arguments and return values can be tuples or dictionaries
    or arbitraty nested structures composed of tuples and dictionaries.

    Args:
      input_shape: a tuple representing the shape of the input.

    Returns:
      The shape of the output.
    """
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

  # End of subclassing interface, all functions below are internal.

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
    # Re-using this layer, no new parameters.
    if not self._first_init:
      return ()

    # First call of this layer, create parameters.
    self._first_init = False
    self._params = self.new_parameters(input_shape, rng)
    return self._params

  def __call__(self, x, params=(), **kwargs):
    try:
      # If params are nothing, we may be reusing this layer.
      # Use the cached parameters to calculate the value.
      # Note: to make sure jit tracers can decide this branch in python we
      #   use "params is ()" instead of, e.g., "not params" or "params == ()".
      if params is ():  # pylint: disable=literal-comparison
        return self.call(x, params=self._params, **kwargs)
      # In this case, we're called for the first time: cache parameters.
      self._params = params
      return self.call(x, params=params, **kwargs)
    except Exception:
      name, trace = self.__class__.__name__, _short_traceback()
      raise LayerError(name, self._caller, shapes(x), trace)


class LayerError(Exception):
  """Exception raised in the layer stack.

  Attributes:
    message: the message corresponding to this exception.
  """

  def __init__(self, layer_name, caller, input_shapes, traceback_string):
    self._layer_name = layer_name
    self._caller = caller  # Python inspect object with init caller info.
    self._traceback = traceback_string
    self._input_shapes = input_shapes
    super(LayerError, self).__init__(self.message)

  @property
  def message(self):
    prefix = 'Exception passing through layer %s:\n' % self._layer_name
    short_path = '[...]/' + '/'.join(self._caller.filename.split('/')[-3:])
    caller = '  layer created in file %s, line %d\n' % (short_path,
                                                        self._caller.lineno)
    shapes_str = '  layer input shapes: %s\n\n' % str(self._input_shapes)
    return prefix + caller + shapes_str + self._traceback


def nested_map(x, f):
  """Map the function f to the nested structure x (dicts, tuples, lists)."""
  if isinstance(x, list):
    return [nested_map(y, f) for y in x]
  if isinstance(x, tuple):
    return tuple([nested_map(y, f) for y in x])
  if isinstance(x, dict):
    return {k: nested_map(x[k], f) for k in x}
  return f(x)


def shapes(x):
  """Get a structure of shapes for a structure of nested arrays."""
  def shape(x):
    try:
      return x.shape
    except Exception:  # pylint: disable=broad-except
      return []
  return nested_map(x, shape)


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


def _short_traceback(skip=3):
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


def layer(output_shape=None, new_parameters=None):
  """Create a layer class from a function."""
  def layer_decorator(call):
    """Decorating the call function."""
    def output_shape_fun(self, input_shape):
      if output_shape is None:
        return input_shape
      kwargs = self._init_kwargs  # pylint: disable=protected-access
      return output_shape(input_shape, **kwargs)

    def new_parameters_fun(self, input_shape, rng):
      if new_parameters is None:
        return ()
      kwargs = self._init_kwargs  # pylint: disable=protected-access
      return new_parameters(input_shape, rng, **kwargs)

    def call_fun(self, x, params=(), **kwargs):
      """The call function of the created class, derived from call."""
      # Merge on-call kwargs with class-kwargs.
      call_kwargs = kwargs.copy()
      call_kwargs.update(self._init_kwargs)  # pylint: disable=protected-access
      # Call with the merged kwargs.
      return call(x, params=params, **call_kwargs)

    # Set doc for python help.
    call_fun.__doc__ = call.__doc__
    if output_shape is None:
      output_shape_fun.__doc__ = output_shape.__doc__
    if new_parameters is None:
      new_parameters_fun.__doc__ = new_parameters.__doc__

    # Create the class.
    cls = type(call.__name__, (Layer,),
               {'call': call_fun,
                'output_shape': output_shape_fun,
                'new_parameters': new_parameters_fun})

    return cls
  return layer_decorator
