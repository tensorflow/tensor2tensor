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

  Layers are the basic building blocks for deep learning models. A Trax layer
  computes a function from zero or more inputs to zero or more outputs,
  optionally using trainable parameters (common) and non-parameter state (not
  common). Authors of new layer subclasses typically override at most two
  methods of the base `Layer` class:

    forward(inputs, params=(), state=(), **kwargs):
      Computes this layer's output as part of a forward pass through the model.

    new_params_and_state(self, input_shape, input_dtype, rng):
      Returns a (params, state) pair suitable for initializing this layer.

  A small subset of layer types are combinators -- they organize the computation
  of their sublayers, e.g., applying their sublayers in series or in parallel.

  All layers have the following properties, with default values implemented
  in the base `Layer` class:

    - n_inputs: int (default 1)
    - n_outputs: int (default 1)
    - params: tuple (default empty -- the layer has no parameters)
    - state: tuple (default empty -- the layer has no non-parameter state)
    - sublayers: tuple (default empty -- the layer has no sublayers)

  The inputs to a layer are tensors, packaged according to how many there are:

    - n_inputs = 0: an empty tuple ()
    - n_inputs = 1: one tensor (NOT wrapped in a tuple)
    - n_inputs > 1: a tuple of tensors

  (The special treatment of the single-input case is meant to simplify the
  work of layer writers; this design choice may be revisited in the future.)

  The outputs from a layer are also tensors, packaged the same as layer inputs:

    - n_outputs = 0: an empty tuple ()
    - n_outputs = 1: the tensor (NOT wrapped in a tuple)
    - n_outputs > 1: a tuple of tensors

  The Trax runtime maintains a data stack with which layer calls are composed.
  For more complex data network architectures, possibly involving multiple data
  flows, one can view each layer as a function from stack state to stack state,
  where the function's inputs are a slice from the stack, and the function's
  outputs are spliced back into the stack.
  """

  def __init__(self, n_inputs=1, n_outputs=1):
    """Creates a partially initialized, unconnected layer instance.

    Args:
      n_inputs: Number of inputs expected by this layer.
      n_outputs: Number of outputs promised by this layer.
    """
    self._n_inputs = n_inputs
    self._n_outputs = n_outputs
    self._sublayers = ()  # Default is no sublayers.
    self._params = ()  # cached parameters
    self._state = ()
    self._caller = _find_frame(inspect.stack())  # for custom error messages
    self._init_finished = False

  def __repr__(self):
    class_str = self.__class__.__name__
    fields_str = 'in={},out={}'.format(self.n_inputs, self.n_outputs)
    objs = self.sublayers
    if objs:
      objs_str = ', '.join(str(x) for x in objs)
      return '{}{{{},sublayers=[{}]}}'.format(class_str, fields_str, objs_str)
    else:
      return '{}{{{}}}'.format(class_str, fields_str)

  def forward(self, inputs, params=(), state=(), **kwargs):
    """Computes this layer's output as part of a forward pass through the model.

    Authors of new Layer subclasses should override this method to define the
    forward computation that their layer performs.

    Args:
      inputs: Input tensors, matching the number (n_inputs) expected by this
          layer. Specifically:
            - n_inputs = 0: an empty tuple ()
            - n_inputs = 1: a tensor (NOT wrapped in a tuple)
            - n_inputs > 1: a tuple of tensors, with n_inputs items
      params: A tuple of trainable parameters, with one element for this layer
          if this layer has no sublayers, or one for each sublayer if this
          layer has sublayers. If a layer (or sublayer) has no trainable
          parameters, the corresponding params element is an empty tuple.
      state: Layer-specific non-parameter state that can update between batches.
      **kwargs: Often empty; main current use is to carry a PRNG key for random
          number generation, using the keyword 'rng'.

    Returns:
      Tensors, matching the number (n_outputs) promised by this layer.
      Specifically:
        - n_outputs = 0: an empty tuple
        - n_outputs = 1: one tensor (NOT wrapped in a tuple)
        - n_outputs > 1: a tuple of tensors, with n_outputs items
    """
    raise NotImplementedError

  def new_params_and_state(self, input_shape, input_dtype, rng):
    """Returns a (params, state) pair suitable for initializing this layer.

    Authors of new Layer subclasses should override this method if their layer
    uses trainable parameters or has non-parameter state that gets updated
    between batches. The default implementation works for layers that have
    no parameters or state.

    Args:
      input_shape: A tuple representing a shape (if this layer takes one input)
          or a tuple of shapes (if this layer takes more than one input).
          For example: (210, 160, 3) or ((210, 160, 3), (105, 80, 3)).
      input_dtype: Numpy dtype(s) for each of the inputs.
      rng: A PRNG key for random number generation.
    """
    del input_shape, input_dtype, rng
    return (), ()

  @property
  def n_inputs(self):
    """Returns how many tensors this layer expects as input."""
    return self._n_inputs

  @property
  def n_outputs(self):
    """Returns how many tensors this layer promises as output."""
    return self._n_outputs

  @property
  def sublayers(self):
    """Returns a tuple containing this layer's sublayers; may be empty."""
    return self._sublayers

  @property
  def params(self):
    """Returns a tuple containing this layer's parameters; may be empty."""
    return self._params

  @params.setter
  def params(self, params):
    self._params = params

  @property
  def state(self):
    """Returns a tuple containing this layer's state; may be empty."""
    return self._state

  @state.setter
  def state(self, state):
    self._state = state

  @property
  def has_backward(self):
    """Returns True if this layer provides its own (custom) backward pass code.

    A layer subclass that provides custom backward pass code (for custom
    gradients) must override this method to return True.
    """
    return False

  def backward(self, inputs, output, grad, params, state, **kwargs):
    """Custom backward pass to propagate gradients in a custom way.

    Args:
      inputs: Input tensors; can be a (possibly nested) tuple.
      output: The result of running this layer on inputs.
      grad: gradient signal (called cotangent in jax) computed based on
        subsequent layers. The structure and shape must match output.
      params: layer parameters
      state: start state.
      **kwargs: kwargs for the layer

    Returns:
      The custom gradient signal for the input. Note that we need to return
      a gradient for each argument of forward, so it will usually be a tuple
      of signals: the gradient for inputs and parameters.
    """
    raise NotImplementedError

  # End of subclassing interface, all functions below are internal.

  def pseudo_forward(self, pseudo_inputs, params, state):
    """Computes shapes and types this layer would produce for the given inputs.

    Args:
      pseudo_inputs: A ShapeType instance (input data minus the actual values)
          or a tuple of ShapeType instances, following the same conventions as
          Layer.forward's input arg.
      params: Parameters for this layer.
      state: start state.

    Returns:
      A tuple of (output, state).

      The output part of the tuple is a ShapeType instance representing the
      shape and type of the output (if this layer has one output) or a tuple
      of ShapeType instances (if this layer has more than one output).
    """
    try:
      # Beware: using an actual RNG (as opposed to this ShapeType stub) would
      # cause a large number of dropout masks to be computed and permanently
      # stored in global memory.
      rng = ShapeType(shape=(2,), dtype=onp.uint32)
      def call_on_input(x, params, state, rng):
        return self.forward(x, params=params, state=state, rng=rng)
      params_shapes = nested_map(
          params, lambda x: ShapeType(shape=x.shape, dtype=x.dtype))
      s = backend.eval_on_shapes(call_on_input)(pseudo_inputs,
                                                params_shapes, state, rng)
      return s
    except Exception:
      name, trace = self.__class__.__name__, _short_traceback(skip=3)
      raise LayerError(name, 'pseudo_forward', self._caller, pseudo_inputs,
                       None, trace)

  def initialize_once(self, input_shapes, input_dtype, rng):
    """Initializes this layer and its sublayers recursively.

    This method is designed to initialize each layer instance once, even if the
    same layer instance occurs in multiple places in the network. This enables
    weight sharing to be implemented as layer sharing.

    Args:
      input_shapes: A tuple representing a shape (if this layer takes one input)
          or a tuple of shapes (if this layer takes more than one input).
          For example: (210, 160, 3) or ((210, 160, 3), (105, 80, 3)).
      input_dtype: Numpy dtype(s) for each of the inputs.
      rng: A PRNG key for random number generation.

    Returns:
      A (params, state) tuple, in which params contains newly created parameters
          on the first call and () on all subsequent calls.
    """
    try:
      # Initialize params once; store them for use when this layer is called.
      # Needs to call new_params_and_state regardless of _init_finished because
      # state also needs to be initialized. After jitting, graph pruning should
      # be able to remove unnecessary computation.
      # TODO(lukaszkaiser): Revisit this decision and see whether layers sharing
      #   params should also share states.
      params, state = self.new_params_and_state(input_shapes, input_dtype, rng)
      if not self._init_finished:
        self._init_finished = True
        self._params = params
        self._state = state
      else:
        params = ()
      return (params, state)
    except Exception:
      name, trace = self.__class__.__name__, _short_traceback(skip=3)
      raise LayerError(name, 'initialize_once', self._caller, input_shapes,
                       input_dtype, trace)

  # XXX(kitaev):
  _STASH_IN = None
  _STASH_OUT = None

  def __call__(self, x, **kwargs):
    """Makes Layer instances callable; for use in tests or interactive settings.

    This convenience method helps library users play with, test, or otherwise
    probe the behavior of layers outside of a full training environment. It
    presents the layer as callable function from inputs to outputs, with the
    option of manually specifying parameters and non-parameter state per
    individual call. For convenience, parameters and non-parameter state are
    cached per layer instance, starting from default values of () and (), and
    acquiring non-empty values either by initialization or from values
    explicitly provided via the params and state keyword arguments.

    Args:
      x: 0 or more input tensors, formatted the same as the inputs to
          Layer.forward.
      **kwargs: Additional keyword arguments if needed/desired for this layer.
          Three possible keyword arguments are especially relevant:
            - params=... will override any cached params values
            - state=... will override any cached state values
            - rng=... will supply a PRNG key for use by the layer

    Returns:
      0 or more output tensors, formatted the same as the outputs from
          Layer.forward.
    """
    params = kwargs.pop('params', self.params)
    state = kwargs.pop('state', self.state)
    outputs, _ = self.apply_forward(x, params=params, state=state, **kwargs)
    return outputs

  def apply_forward(self, x, params=(), state=(), **kwargs):
    """Applies this layer as part of a forward pass; an internal system method.

    This method is reserved for handling plumbing and other internal affairs
    as needed by the overall library. Trax library users should use or override
    the `forward` method instead.

    Args:
      x: See Layer.forward inputs.
      params: See Layer.forward.
      state: See Layer.forward.
      **kwargs: See Layer.forward.

    Returns:
      See Layer.forward.
    """
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

      if not self.has_backward or Layer._STASH_IN is not None:
        outputs, s = self.forward(x, params=params, state=state, **kwargs)
      else:
        outputs, s = self._do_custom_gradients(x, params, state, **kwargs)
      self._state = s
      return outputs, s

    except Exception:
      name, trace = self.__class__.__name__, _short_traceback()
      raise LayerError(name, 'apply_forward', self._caller,
                       shapes(x), None, trace)

  def _do_custom_gradients(self, x, params, state, **kwargs):
    """Calls this layer for a forward pass, but with custom gradients."""
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
    def _do_forward(y, params):
      return check_end_state(self.forward(y, params=params, state=state,
                                          **kwargs))

    # This is the custom gradient (vector-jacobian product in JAX) function.
    # For the exact specification of this custom transformation see this link:
    # https://jax.readthedocs.io/en/latest/jax.html#jax.defjvp_all
    def do_forward_vjp(y, params):
      """Custom gradient (vjp) function."""
      stash = None
      if Layer._STASH_IN is None:
        Layer._STASH_IN = stash = {}
      output = check_end_state(self.forward(y, params=params, state=state,
                                            **kwargs))
      if stash is not None:
        Layer._STASH_IN = None
      def vjpfun(grad):
        assert Layer._STASH_OUT is None
        Layer._STASH_OUT = stash
        res = self.backward(y, output, grad, params, state, **kwargs)
        Layer._STASH_OUT = None
        return res
      return output, vjpfun

    jax.defvjp_all(_do_forward, do_forward_vjp)
    return _do_forward(x, params), state


class LayerError(Exception):
  """Exception raised in the layer stack.

  Attributes:
    message: the message corresponding to this exception.
  """

  def __init__(self, layer_name, function_name, caller,
               input_shapes, input_types, traceback_string):
    self._layer_name = layer_name
    self._function_name = function_name
    self._caller = caller  # Python inspect object with init caller info.
    self._traceback = traceback_string
    self._input_shapes = input_shapes
    self._input_types = input_types
    super(LayerError, self).__init__(self.message)

  @property
  def message(self):
    """Create error message."""
    prefix = 'Exception passing through layer '
    prefix += '%s (in %s):\n' % (self._layer_name, self._function_name)
    short_path = '[...]/' + '/'.join(self._caller.filename.split('/')[-3:])
    caller = '  layer created in file %s, line %d\n' % (short_path,
                                                        self._caller.lineno)
    shapes_str = '  layer input shapes: %s\n\n' % str(self._input_shapes)
    if self._input_types is not None:
      types_str = '  layer input types: %s\n' % str(self._input_types)
      shapes_str = types_str + shapes_str
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


def _validate_forward_input(x, n_inputs):
  if n_inputs != 1:
    if not isinstance(x, tuple):
      raise TypeError(
          'expected input to be a tuple; instead received {}'.format(type(x)))
    if len(x) != n_inputs:
      raise ValueError(
          'input tuple length ({}) does not equal required number of inputs'
          ' ({})'.format(len(x), n_inputs))


def layer(n_inputs=1, n_outputs=1, new_params_and_state_fn=None):
  """Returns a decorator that converts a function into a Layer class builder."""

  def _build_layer_class(raw_fn):
    """Returns a Layer class whose callable instances execute the function."""

    def _init(self, **kwargs):
      self._kwargs = kwargs  # pylint: disable=protected-access
      Layer.__init__(self, n_inputs=n_inputs, n_outputs=n_outputs)

    def _new_params_and_state(self, input_shapes, input_dtype, rng):
      if new_params_and_state_fn is None:
        return (), ()
      kwargs = self._kwargs  # pylint: disable=protected-access
      return new_params_and_state_fn(input_shapes, input_dtype, rng, **kwargs)

    def _is_empty(raw_output):
      return raw_output is None or (isinstance(raw_output, (list, tuple))
                                    and len(raw_output) == 0)  # pylint: disable=g-explicit-length-test

    def _forward(self, x, params=(), state=(), **kwargs):
      """Uses this layer as part of a forward pass through the model."""
      merged_kwargs = kwargs.copy()
      merged_kwargs.update(self._kwargs)  # pylint: disable=protected-access

      _validate_forward_input(x, n_inputs)
      raw_output = raw_fn(x, params=params, **merged_kwargs)
      output = () if _is_empty(raw_output) else raw_output
      return (output, state)

    # Set docstrings and create the class.
    _forward.__doc__ = raw_fn.__doc__
    _new_params_and_state.__doc__ = new_params_and_state_fn.__doc__
    # Note: None.__doc__ is None
    cls = type(raw_fn.__name__, (Layer,),
               {'__init__': _init,
                'forward': _forward,
                'new_params_and_state': _new_params_and_state})
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


def check_shape_agreement(layer_obj, input_shapes, integer_inputs=False):
  """Checks if the layer's call output agrees its pseudo_forward predictions.

  This function helps test layer mechanics and inter-layer connections that
  aren't dependent on specific data values.

  Args:
    layer_obj: A Layer instance.
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
  params, state = layer_obj.initialize_once(input_shapes, input_dtype, rng1)
  pseudo_output, _ = layer_obj.pseudo_forward(pseudo_data, params, state)
  if isinstance(pseudo_output, tuple):
    output_shape = tuple(x.shape for x in pseudo_output)
  else:
    output_shape = pseudo_output.shape

  random_input = _random_values(input_shapes, rng2, integer_inputs)
  real_output = layer_obj(random_input, params=params, state=state, rng=rng3)
  result_shape = shapes(real_output)

  msg = 'output shape %s != real result shape %s' % (output_shape, result_shape)
  assert output_shape == result_shape, msg
  # TODO(jonni): Remove this assert? It makes test logs harder to read.
  return output_shape
