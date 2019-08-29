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

import numpy as onp
from tensor2tensor.trax import backend
from tensor2tensor.trax.backend import nested_map
from tensor2tensor.trax.backend import ShapeType


class Layer(object):
  """Base class for composable layers in a deep learning network.

  A layer is a function from zero or more inputs to zero or more outputs,
  possibly with trainable parameters. A layer is either atomic or composed
  of sublayers. These aspects of a layer are set via a layer's constructor,
  and can be inspected via read-only properties:

    - n_inputs
    - n_outputs
    - sublayers

  The inputs to a layer are activation tensors, packaged according to how many
  there are:

    - n_inputs = 0: an empty tuple ()
    _ n_inputs = 1: the activation tensor (NOT wrapped in a tuple)
    _ n_inputs > 1: a tuple of activation tensors

  (The special treatment for the single-input case is intended as a
  simplification for layer writers; this design choice may be revisited in the
  future.)

  The outputs from a layer are also activations tensors, packaged the same as
  layer inputs:

    - n_outputs = 0: an empty tuple ()
    _ n_outputs = 1: the activation tensor (NOT wrapped in a tuple)
    _ n_outputs > 1: a tuple of activation tensors

  The runtime maintains a data stack with which layer calls are composed. One
  can therefore view each layer as a function from stack state to stack state,
  where the function's inputs are a slice from the stack, and the function's
  outputs are spliced back into the stack.
  """

  def __init__(self, n_inputs=1, n_outputs=1):
    self._n_inputs = n_inputs
    self._n_outputs = n_outputs
    self._sublayers = ()  # Default is no sublayers.
    self._params = ()  # cached parameters
    self._caller = _find_frame(inspect.stack())  # for custom error messages
    self._init_finished = False

  def __repr__(self):
    class_str = self.__class__.__name__
    fields_str = 'in={},out={}'.format(self.n_inputs, self.n_outputs)
    objs = self.sublayers
    if objs:
      objs_str = ', '.join(str(x) for x in objs)
      return '{}[{},layers=[{}]]'.format(class_str, fields_str, objs_str)
    else:
      return '{}[{}]'.format(class_str, fields_str)

  def call(self, inputs, params=(), state=(), **kwargs):
    """Applies this layer to given activation tensors, using trainable params.

    Args:
      inputs: Data tensors, matching the number (n_inputs) expected by this
          layer. Specifically:
            - n_inputs = 0: an empty tuple ()
            - n_inputs = 1: a data tensor (NOT wrapped in a tuple)
            - n_inputs > 1: a tuple of data tensors, with n_inputs items
      params: A tuple of trainable parameters, with one element for this layer
          and one for each of this layer's sublayers. If a layer (or sublayer)
          has no trainable parameters, the corresponding params element is an
          empty tuple.
      state: start state.
      **kwargs: Layer-specific keyword args.

    Returns:
      Data tensors, matching the number (n_outputs) promised by this layer.
      Specifically:
        - n_outputs = 0: an empty tuple
        - n_outputs = 1: a data tensor (NOT wrapped in a tuple)
        - n_outputs > 1: a tuple of data tensors, with n_outputs items
      A tuple of activation tensors, one for each output.
    """
    raise NotImplementedError

  # TODO(wangpeng): Should be called `new_parameters_and_state`.
  def new_parameters(self, input_shapes, input_dtype, rng):
    """Creates layer-specific parameters based on data shape, dtype and rng.

    Args:
      input_shapes: A tuple, depending on the number of inputs (n_inputs)
          expected by this layer:
            - n_inputs = 0: an empty tuple ()
            - n_inputs = 1: a tuple representing the shape of the input
            - n_inputs > 1: a tuple of shape tuples, one for each input
          For example:
            - 0 inputs: ()
            - 1 input: (210, 160, 3) [NOTE: no tuple wrapping the shape]
            - 2 inputs: ((210, 160, 3), (105, 80, 3))
      input_dtype: numpy dtype of the input.
      rng: A random number generator.

    Returns:
      The newly created parameters for this layer.
    """
    raise NotImplementedError

  @property
  def n_inputs(self):
    """Specifies how many data tensors this layer expects as input."""
    return self._n_inputs

  @property
  def n_outputs(self):
    """Specifies how many data tensors this layer promises as output."""
    return self._n_outputs

  @property
  def sublayers(self):
    """Returns the sublayers contained in / managed by this layer."""
    return self._sublayers

  @property
  def has_custom_grad(self):
    """Whether to use custom gradients (in which case, see below)."""
    return False

  def custom_grad(self, inputs, output, grad, params, state, **kwargs):
    """Custom backward pass to propagate gradients in a custom way.

    Args:
      inputs: Input activations; can be a (possibly nested) tuple.
      output: The result of running this layer on inputs.
      grad: gradient signal (called cotangent in jax) computed based on
        subsequent layers. The structure and shape must match output.
      params: layer parameters
      state: start state.
      **kwargs: kwargs for the layer

    Returns:
      The custom gradient signal for the input. Note that we need to return
      a gradient for each argument of call, so it will usually be a tuple
      of signals: the gradient for inputs and parameters.
    """
    raise NotImplementedError

  # End of subclassing interface, all functions below are internal.

  def pseudo_call(self, pseudo_inputs, params, state):
    """Computes shapes and types this layer would produce for the given inputs.

    Args:
      pseudo_inputs: A ShapeType instance (input data minus the actual values)
          or a tuple of ShapeType instances, following the same conventions as
          Layer.call's input arg.
      params: Parameters for this layer.
      state: start state.

    Returns:
      A ShapeType instance representing the shape and type of the output (if
      this layer has one output) or a tuple of ShapeType instances (if this
      layer has more than one output).
    """
    try:
      # Beware: using an actual RNG (as opposed to this ShapeType stub) would
      # cause a large number of dropout masks to be computed and permanently
      # stored in global memory.
      rng = ShapeType(shape=(2,), dtype=onp.uint32)
      def call_on_input(x, params, state, rng):
        return self.call(x, params=params, state=state, rng=rng)
      params_shapes = nested_map(
          params, lambda x: ShapeType(shape=x.shape, dtype=x.dtype))
      s = backend.eval_on_shapes(call_on_input)(pseudo_inputs,
                                                params_shapes, state, rng)
      return s
    except Exception:
      name, trace = self.__class__.__name__, _short_traceback(skip=3)
      raise LayerError(name, 'pseudo_call', self._caller, pseudo_inputs, trace)

  def initialize(self, input_shapes, input_dtype, rng):
    """Initialize the layer given an input shape, dtype and rng.

    Returns new_parameters(input_shapes, rng) on the first call and () on any
    subsequent call, as the layer is already initialized. This is used for
    networks that share parameters, so the layer only produces them once.

    Args:
      input_shapes: A tuple representing a shape (if this layer takes one input)
          or a tuple of shapes (if this layer takes more than one input).
          For example: (210, 160, 3) or ((210, 160, 3), (105, 80, 3)).
      input_dtype: numpy dtype of the input.
      rng: A random number generator.

    Returns:
      Newly created parameters on the first call and () on all subsequent calls.
    """
    try:
      # Initialize params once; store them for use when this layer is called.
      # Needs to call new_parameters regardless of _init_finished because state
      # also needs to be initialized. After jitting, graph pruning should be
      # able to remove unnecessary computation.
      # TODO(lukaszkaiser): Revisit this decision and see whether layers sharing
      #   params should also share states.
      params, state = self.new_parameters(input_shapes, input_dtype, rng)
      if not self._init_finished:
        self._init_finished = True
        self._params = params
      else:
        params = ()
      return (params, state)
    except Exception:
      name, trace = self.__class__.__name__, _short_traceback(skip=3)
      raise LayerError(name, 'initialize', self._caller, input_shapes, trace)

  def __call__(self, x, params=(), state=(), **kwargs):
    try:
      # If params are nothing, we may be reusing this layer.
      # Use the cached parameters to calculate the value.
      # Note: to make sure jit tracers can decide this branch in python we
      #   use "params is ()" instead of, e.g., "not params" or "params == ()".
      if params is ():  # pylint: disable=literal-comparison
        params = self._params
      else:
        # In this case, we're called for the first time: cache parameters.
        self._params = params

      if not self.has_custom_grad:
        return self.call(x, params=params, state=state, **kwargs)

      # Custom gradients part.
      assert backend.get_name() == 'jax', (
          'Custom gradients are only supported in JAX for now.')

      # TODO(wangpeng): JAX doesn't support custom grads for functions with
      #   auxiliary output yet (https://github.com/google/jax/issues/844). Will
      #   remove the constraints on state below when this feature is added to
      #   JAX.

      assert not jax.tree_util.tree_leaves(state), (
          'Custom gradients require trivial start state. Got %s' % str(state))

      def check_end_state(output_state):
        output, state = output_state
        assert not jax.tree_util.tree_leaves(state), (
            'Custom gradients require trivial end state. Got %s' % str(state))
        return output

      # See this link for how custom transformations are defined in JAX:
      # https://jax.readthedocs.io/en/latest/jax.html#jax.custom_transforms
      # Note that we capture the kwargs and don't calculate gradients wrt. them.
      @jax.custom_transforms
      def do_call(y, params):
        return check_end_state(self.call(y, params=params, state=state,
                                         **kwargs))

      # This is the custom gradient (vector-jacobian product in JAX) function.
      # For the exact specification of this custom transformation see this link:
      # https://jax.readthedocs.io/en/latest/jax.html#jax.defjvp_all
      def do_call_vjp(y, params):
        output = check_end_state(self.call(y, params=params, state=state,
                                           **kwargs))
        def vjpfun(grad):
          return self.custom_grad(y, output, grad, params, state, **kwargs)
        return output, vjpfun

      jax.defvjp_all(do_call, do_call_vjp)
      return do_call(x, params), state

    except Exception:
      name, trace = self.__class__.__name__, _short_traceback()
      raise LayerError(name, 'call', self._caller, shapes(x), trace)


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


def nested_reduce(x, f):
  """Fold the function f to the nested structure x (dicts, tuples, lists)."""
  if isinstance(x, list):
    return f([nested_reduce(y, f) for y in x])
  if isinstance(x, tuple):
    return f([nested_reduce(y, f) for y in x])
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


def _validate_call_input(x, n_inputs):
  if n_inputs != 1:
    if not isinstance(x, tuple):
      raise TypeError(
          'expected input to be a tuple; instead received {}'.format(type(x)))
    if len(x) != n_inputs:
      raise ValueError(
          'input tuple length ({}) does not equal required number of inputs'
          ' ({})'.format(len(x), n_inputs))


def layer(n_inputs=1, n_outputs=1, new_parameters=None):
  """Decorates a function to make it the call method of a new Layer class."""
  # TODO(jonni): Consider renaming new_parameters to new_parameters_fn.

  def _build_layer_class(raw_call_fn):
    """Returns a Layer class built around the given call function."""

    def _init(self, **kwargs):
      self._kwargs = kwargs  # pylint: disable=protected-access
      Layer.__init__(self, n_inputs=n_inputs, n_outputs=n_outputs)

    def _new_parameters(self, input_shapes, input_dtype, rng):
      if new_parameters is None:
        return (), ()
      kwargs = self._kwargs  # pylint: disable=protected-access
      return new_parameters(input_shapes, input_dtype, rng, **kwargs), ()

    def _is_empty(raw_output):
      return raw_output is None or (isinstance(raw_output, (list, tuple))
                                    and len(raw_output) == 0)  # pylint: disable=g-explicit-length-test

    def _call_with_context(self, x, params=(), state=(), **kwargs):
      """Calls raw_call_fn with extra keyword args from Layer.__init__."""
      merged_kwargs = kwargs.copy()
      merged_kwargs.update(self._kwargs)  # pylint: disable=protected-access

      _validate_call_input(x, n_inputs)
      raw_output = raw_call_fn(x, params=params, **merged_kwargs)
      output = () if _is_empty(raw_output) else raw_output
      return (output, state)

    # Set docstrings and create the class.
    _call_with_context.__doc__ = raw_call_fn.__doc__
    _new_parameters.__doc__ = new_parameters.__doc__  # None.__doc__ is None
    cls = type(raw_call_fn.__name__, (Layer,),
               {'__init__': _init,
                'call': _call_with_context,
                'new_parameters': _new_parameters})
    return cls

  return _build_layer_class


def _random_values(input_shapes, rng, integer_inputs=False):
  """Creates random floats or ints of the given shape.

  Args:
    input_shapes: A tuple representing a shape (if the layer takes one input)
        or a tuple of shapes (if this layer takes more than one input).
        For example: (210, 160, 3) or ((210, 160, 3), (105, 80, 3)).
    rng: A random number generator.
    integer_inputs: If True, use numpy int32 to produce the random data, else
        use float32.

  Returns:
    Random values with the shape and type specified.
  """
  if isinstance(input_shapes[0], int):
    # Non-nested shape, create a random tuple.
    if not integer_inputs:
      return backend.random.uniform(rng, input_shapes, minval=-1.0, maxval=1.0)
    return backend.random.bernoulli(rng, 0.5, input_shapes).astype(onp.int32)
  elif isinstance(input_shapes, tuple):  # Nested shape: tuple.
    return tuple(_random_values(x, rng, integer_inputs) for x in input_shapes)
  else:
    raise TypeError(type(input_shapes))


def _is_tuple_of_shapes(shape):
  # TODO(jonni): Find better way to distinguish a shape from a tuple of shapes.
  if not isinstance(shape, tuple):
    raise TypeError('shape must be a tuple or tuple of tuples, instead got:'
                    ' {}'.format(shape))
  return isinstance(shape, tuple) and isinstance(shape[0], tuple)


def check_shape_agreement(layer_fn, input_shapes, integer_inputs=False):
  """Checks if the layer's call output agrees its pseudo_call predictions.

  This function helps test layer mechanics and inter-layer connections that
  aren't dependent on specific data values.

  Args:
    layer_fn: A Layer instance, viewed as a function from input shapes to
        output shapes.
    input_shapes: A tuple representing a shape (if the layer takes one input)
        or a tuple of shapes (if this layer takes more than one input).
        For example: (210, 160, 3) or ((210, 160, 3), (105, 80, 3)).
    integer_inputs: If True, use numpy int32 as the type for the pseudo-data,
        else use float32.

  Returns:
    A tuple representing either a single shape (if the layer has one output) or
    a tuple of shape tuples (if the layer has more than one output).
  """
  rng1, rng2, rng3 = backend.random.split(backend.random.get_prng(0), 3)
  input_dtype = onp.int32 if integer_inputs else onp.float32
  if _is_tuple_of_shapes(input_shapes):
    pseudo_data = tuple(ShapeType(x, input_dtype) for x in input_shapes)
    input_dtype = tuple(input_dtype for _ in input_shapes)
  else:
    pseudo_data = ShapeType(input_shapes, input_dtype)
  params, state = layer_fn.initialize(input_shapes, input_dtype, rng1)
  pseudo_output, _ = layer_fn.pseudo_call(pseudo_data, params, state)
  if isinstance(pseudo_output, tuple):
    output_shape = tuple(x.shape for x in pseudo_output)
  else:
    output_shape = pseudo_output.shape

  random_input = _random_values(input_shapes, rng2, integer_inputs)
  real_output, _ = layer_fn(random_input, params, state=state, rng=rng3)
  result_shape = shapes(real_output)

  msg = 'output shape %s != real result shape %s' % (output_shape, result_shape)
  assert output_shape == result_shape, msg
  # TODO(jonni): Remove this assert? It makes test logs harder to read.
  return output_shape
