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

"""SLAX - Layer eXtensions to Stax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from absl import logging
from jax.tree_util import register_pytree_node as _register_pytree_node

from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.stax import stax_base as stax


# Utility functions
# ------------------------------------------------------------------------------
def one_hot(x, size, dtype=np.float32):
  """Make a n+1 dim one-hot array from n dim int-categorical array."""
  return np.array(x[..., np.newaxis] == np.arange(size), dtype)


def ShiftRight():  # pylint: disable=invalid-name
  """Layer to shift the tensor to the right by padding on axis 1."""
  init_fun = lambda input_shape: (input_shape, ())
  def apply_fun(params, inputs, **kwargs):
    del params, kwargs
    pad_widths = [(0, 0), (1, 0)]
    pad_widths += [(0, 0) for _ in range(len(inputs.shape) - 2)]
    padded = np.pad(inputs, pad_widths, mode='constant')
    return padded[:, :-1, ...]
  return init_fun, apply_fun


# Utility Combinators
# ------------------------------------------------------------------------------
def repeat(layer, num_repeats):
  """Repeats layers serially num_repeats times."""
  if num_repeats < 1:
    raise ValueError('Repeat combinator num_repeats must be >= 1.')
  layers = num_repeats * (layer,)
  return stax.serial(*layers)


def residual(*layers, **kwargs):
  """Constructs a residual version of layers, summing input to layers output."""
  res = kwargs.get('res', stax.Identity)
  if len(layers) > 1:
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(stax.serial(*layers), res),
        stax.FanInSum
    )
  elif len(layers) == 1:
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(layers[0], res),
        stax.FanInSum
    )
  else:
    raise ValueError('Empty residual combinator.')


# Utility Layers
# ------------------------------------------------------------------------------
def Take(*args):  # pylint: disable=invalid-name
  """Layer to pick subset of inputs from parallel input stream.

  Args:
    *args: a sequence of ints

  Returns:
    A new layer that selects inputs from an incoming parallel stream.
    In numpy notation: outputs = parallel_inputs[args]
    If the resulting output list has only one member, it is automatically
    unwrapped and the contents are passed on directly.
  """
  def init_fun(input_shape):
    output_shape = []
    for arg in args:
      output_shape.append(input_shape[arg])
    if len(output_shape) == 1:
      output_shape = output_shape[0]
    return (output_shape, ())
  def apply_fun(params, inputs, **kwargs):
    del params, kwargs
    outputs = []
    for arg in args:
      outputs.append(inputs[arg])
    if len(outputs) == 1:
      outputs = outputs[0]
    return outputs
  return init_fun, apply_fun


def LogInputs(prefix='', debug=True):  # pylint: disable=invalid-name
  """Logging side-effects layer, equivalent to Identity.

  Args:
    prefix: string: logging prefix
    debug: bool: if True this will print logs, otherwise not.

  Returns:
    An Identity layer with log-printing side-effects. This
  prints the types and shapes of the inputs.  NB: at the moment
  this doesn't handle printing nested tuple/list shapes!
  """
  def return_shapes(inputs):
    """Return shape information of inputs."""
    if isinstance(inputs, _PlaceholderTree):
      return []
    if isinstance(inputs, (list, tuple)):
      return [x.shape for x in inputs]
    elif isinstance(inputs, dict):
      return [inputs[k].shape for k in inputs.keys()]
    else:
      return inputs.shape
  def init_fun(input_shape):
    if debug:
      logging.info('%s [init]: %s', prefix, input_shape)
    return input_shape, ()
  def apply_fun(params, inputs, **kwargs):
    del params, kwargs
    if debug:
      logging.info('%s: %s %s', prefix, type(inputs), return_shapes(inputs))
    return inputs
  return init_fun, apply_fun


# Staxlayer binding to python variables
# ------------------------------------------------------------------------------
# Stax params-tree leaf type to mark bound subtrees references.
class _TreeMarker(dict):
  pass
# Add this leaf-type to JAX's tree-walker.
_register_pytree_node(_TreeMarker,
                      lambda xs: (tuple(), None),
                      lambda _, xs: _TreeMarker())


# TODO(levskaya, rsepassi): abstract away tuple-subclassing to StaxLayer?
class Share(tuple):
  """Layer parameter caching function to allow weight sharing.

  Args:
    A staxlayer: an (init_fun, apply_fun) pair.

  Returns:
    A 'parameter-bound' staxlayer that can be assigned to a python variable.
  Wherever this value is needed elsewhere in the stax tree, call this bound
  variable and all occurrences will share parameters that will automatically
  be updated by Stax optimizers.
  """

  def __init__(self, staxlayer):  # pylint: disable=super-init-not-called
    self._orig_init_fun, self._orig_apply_fun = staxlayer
    self._first_init = True
    self.params = None  # cached staxlayer params

  def _init_fun(self, input_shape):  # pylint: disable=missing-docstring
    if self._first_init:
      # point of first subgraph initialization call: sets params, output_shape
      self._first_init = False
      out_shape, self.params = self._orig_init_fun(input_shape)
      return out_shape, self.params
    else:
      # point of subgraph reuse:
      # params are just a marker to apply_funs signalling subgraph params reuse
      out_shape, _ = self._orig_init_fun(input_shape)
      return out_shape, _TreeMarker()

  def _apply_fun(self, params, inputs, **kwargs):
    if isinstance(params, _TreeMarker):
      # point of subgraph reuse: calculate new value with cached params
      return self._orig_apply_fun(self.params, inputs, **kwargs)
    else:
      # point of first subgraph application to params: cache params
      self.params = params
      return self._orig_apply_fun(params, inputs, **kwargs)

  # when unpacking this (init, apply) pair we return the wrapped funs
  def __iter__(self):
    return iter((self._init_fun, self._apply_fun))


class Bind(tuple):
  """Layer/variable caching function to allow name binding.

  Args:
    A staxlayer: an (init_fun, apply_fun) pair.

  Returns:
    A 'bound' staxlayer that can be assigned to a python variable.
  Wherever this value is needed elsewhere in the stax tree, call this bound
  variable and all occurrences will share output values.
  """

  def __init__(self, staxlayer):  # pylint: disable=super-init-not-called
    self._orig_init_fun, self._orig_apply_fun = staxlayer
    self._first_init = True
    self._out_shape = None  # cached staxlayer output shape
    self.params = None  # cached staxlayer params
    self.value = None  # cached staxlayer output value

  def _init_fun(self, input_shape):
    if self._first_init:
      # point of first subgraph initialization call: sets params, output_shape
      self._first_init = False
      self._out_shape, self.params = self._orig_init_fun(input_shape)
      return self._out_shape, self.params
    else:
      # point of subgraph reuse:
      # params are just a marker to apply_funs signalling subgraph value reuse
      return self._out_shape, _TreeMarker()

  def _apply_fun(self, params, inputs, **kwargs):
    if isinstance(params, _TreeMarker):
      # point of subgraph reuse: return cached value
      return self.value
    else:
      # point of first subgraph application to params: cache value
      self.params = params
      self.value = self._orig_apply_fun(params, inputs, **kwargs)
      return self.value

  # when unpacking this (init, apply) pair we return the wrapped funs
  def __iter__(self):
    return iter((self._init_fun, self._apply_fun))


# Convenience methods for common use-case of input variable capture and reuse.
Var = lambda: Bind(stax.Identity)  # pylint: disable=invalid-name,
Vars = lambda num_vars: tuple(Bind(stax.Identity) for _ in range(num_vars))  # pylint: disable=invalid-name,


def make_apply_fun(bound_layer):
  """Returns an apply function partially applied to bound params.

  Requires that the top-level model apply_fun be fed params with
  concrete values for these bound params to be numerically meaningful!
  (e.g. not JaxprTrace arrays from a JAX JIT pass!)

  Args:
    bound_layer: Share/Bind/Lambda-bound staxlayer

  Returns:
    An apply function for this subgraph.
  """
  if not isinstance(bound_layer, (Share, Bind)):
    raise ValueError('Can only create apply function from bound layer.')
  def partial_apply_fun(inputs, **kwargs):
    return bound_layer._orig_apply_fun(  # pylint: disable=protected-access
        bound_layer.params, inputs, **kwargs)
  return partial_apply_fun


# Lambda
# ------------------------------------------------------------------------------
# The below provide a nicer syntax for 'pointy' function definition than using
# raw bound variables.
class LambdaBind(Bind):
  """Layer/variable caching function to allow name binding for Lambda layers.

  Args:
    A staxlayer: an (init_fun, apply_fun) pair.

  Returns:
    A 'bound' staxlayer that can be assigned to a python variable.
  Wherever this value is needed elsewhere in the stax tree, call this bound
  variable and all occurrences will share output values.  Overloads __call__
  to provide syntactic sugar for Lambda-like invocation.
  """

  # Syntactic sugar for applying this Lambda to other staxlayers
  # NB: we do not bind the result by default here!
  def __call__(self, *args):
    if len(args) > 1:
      return stax.serial(stax.parallel(*args), self)
    elif len(args) == 1:
      return stax.serial(args[0], self)
    else:
      return self


class _PlaceholderTree(tuple):
  """Placeholder tree object for 'initializing' combinators inside Lambdas.

  When we create a Lambda, we're cutting off normal Stax data flow into
  the subgraph that Lambda wraps with its bound inputs.  This is a
  problem for any (potentially nested) parallel/serial combinators that
  are input-facing, as they'll try to unpack the input_shape, inputs, and
  params trees to feed their sub-layers.  We can't easily know what series
  of nested access patterns are in a function, so we instead provide
  recursive placeholder trees to placate the combinators. These placeholders
  should feed into Lambda input nodes that completely ignore their inputs
  anyway, but they'll break immediately if the user tries to use unbound
  inputs from the Stax chain, which is a useful way to force the semantics
  of Lambda.  This is aggressively tested for correctness in our unit tests.
  """

  def __init__(self):  # pylint: disable=super-init-not-called
    self.shape = 0
    # set generous safety limits for placeholder tree recursion and traversal
    self.iterator_limit = 1000
    self.recursion_limit = 30

  def __getitem__(self, _):
    if self.recursion_limit > 0:
      self.recursion_limit -= 1
      return self
    else:
      raise IndexError('_PlaceholderTree reached maximum depth')

  def __iter__(self):
    return self

  def __next__(self):  # PY3
    return self.next()

  def next(self):  # PY2
    if self.iterator_limit > 0:
      self.iterator_limit -= 1
      return self
    else:
      raise StopIteration
# Register this class with tree-walker to be ignored by optimizers' init fns.
_register_pytree_node(_PlaceholderTree,
                      lambda xs: (tuple(), None),
                      lambda _, xs: _PlaceholderTree())


def _PlaceholderInputs():  # pylint: disable=invalid-name
  """Feeds placeholders into input combinators of a Lambda-bound staxlayer."""
  init_fun = lambda input_shape: iter((_PlaceholderTree(), _PlaceholderTree()))
  apply_fun = lambda params, inputs, **kwargs: _PlaceholderTree()
  return init_fun, apply_fun
_PlaceholderInputs = _PlaceholderInputs()  # pylint: disable=invalid-name


def Lambda(fn):  # pylint: disable=invalid-name
  """Turn a normal function into a bound, callable Stax layer.

  Args:
    fn: a python function with _named_ args (i.e. no *args) and no kwargs.

  Returns:
    A callable, 'bound' staxlayer that can be assigned to a python variable and
    called like a function with other staxlayers as arguments.  Like Bind,
    wherever this value is placed in the stax tree, it will always output the
    same cached value.
  """
  # fn's args are just symbolic names that we fill with Vars.
  num_args = len(inspect.getargspec(fn).args)
  if num_args > 1:
    bound_args = Vars(num_args)
    return LambdaBind(stax.serial(
        stax.parallel(*bound_args),  # capture inputs
        _PlaceholderInputs,  # placeholders for input combinators inside fn
        fn(*bound_args)  # feed captured inputs into fn's args
    ))
  elif num_args == 1:
    bound_arg = Var()
    return LambdaBind(stax.serial(
        bound_arg,  # capture input
        _PlaceholderInputs,  # placeholders for input combinators inside fn
        fn(bound_arg)  # feed captured inputs into fn's args
    ))
  # LambdaBind when no args are given:
  else:
    return LambdaBind(fn())
