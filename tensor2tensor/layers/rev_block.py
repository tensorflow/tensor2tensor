# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Reversible Residual Block.

From
[The Reversible Residual Network: Backpropagation Without Storing
Activations](https://arxiv.org/abs/1707.04585).

Also contains the @recompute_grad decorator, which recomputes the forward
function on the backwards pass.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import re

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers
import tensorflow as tf

LAYER_RE = re.compile(".*revlayer_([0-9]*)/([fg])/.*")


def _acc_grads(*lists_of_grads):
  """Accumulates lists of gradients."""
  acc_grads = []
  for grads in zip(*lists_of_grads):
    grads = [g for g in grads if g is not None]
    if grads:
      acc_grads.append(tf.add_n(grads))
    else:
      acc_grads.append(None)
  return acc_grads


def _rev_layer_forward(xs, f, g, f_side_input, g_side_input,
                       gate_outputs=False):
  """Forward for 1 reversible layer."""
  x1, x2 = xs
  y1 = x1 + (f(x2, f_side_input) if f_side_input else f(x2))
  y2 = x2 + (g(y1, g_side_input) if g_side_input else g(y1))
  if gate_outputs:
    return tf.tuple([y1, y2])
  else:
    return (y1, y2)


def _rev_layer_backward(ys, grad_ys, f, g, f_vars, f_side_input, g_vars,
                        g_side_input):
  """Backprop for 1 layer."""
  y1, y2 = ys
  grad_y1, grad_y2 = grad_ys

  # Reconstruct intermediates and inputs (x1, x2)
  # stop_gradients required on fn inputs to prevent infinite recursion into this
  # grad function on the calls to tf.gradients.
  y1_stop = tf.stop_gradient(y1)
  g_side_input = [tf.stop_gradient(t) for t in g_side_input]
  gy1 = g(y1_stop, g_side_input) if g_side_input else g(y1_stop)

  x2 = y2 - gy1
  x2_stop = tf.stop_gradient(x2)
  f_side_input = [tf.stop_gradient(t) for t in f_side_input]
  fx2 = f(x2_stop, f_side_input) if f_side_input else f(x2_stop)

  x1 = y1 - fx2

  # Compute gradients wrt to inputs
  # dL/dy2 * dG(y1)/y1
  grad_gy1_y2 = tf.gradients(gy1, y1_stop, grad_y2)[0]
  grad_x1 = grad_y1 + grad_gy1_y2
  grad_x2 = (
      tf.gradients(fx2, x2_stop, grad_y1)[0] + grad_y2 +
      tf.gradients(fx2, x2_stop, grad_gy1_y2)[0])

  # Compute gradients wrt to vars and side inputs in f and g
  grads1 = tf.gradients(gy1, g_vars + g_side_input, grad_y2)
  grad_g_vars, grad_g_side = grads1[:len(g_vars)], grads1[len(g_vars):]
  grads2 = tf.gradients(fx2, f_vars + f_side_input, grad_y1)
  grad_f_y1, grad_f_side1 = grads2[:len(f_vars)], grads2[len(f_vars):]
  grads3 = tf.gradients(fx2, f_vars + f_side_input, grad_gy1_y2)
  grad_f_y2, grad_f_side2 = grads3[:len(f_vars)], grads3[len(f_vars):]
  grad_f_vars = _acc_grads(grad_f_y1, grad_f_y2)

  grad_f_side = _acc_grads(grad_f_side1, grad_f_side2)

  # Put returns in a tuple to ensure a constant memory budget (i.e. don't want
  # the subsequent layer to start computing and consuming memory based on a
  # subset of these values).
  outputs = ((x1, x2), (grad_x1, grad_x2), (grad_f_vars, grad_f_side),
             (grad_g_vars, grad_g_side))
  tupled = tf.tuple(tf.contrib.framework.nest.flatten(outputs))
  return tf.contrib.framework.nest.pack_sequence_as(outputs, tupled)


def _rev_block_forward(x1,
                       x2,
                       f,
                       g,
                       num_layers=1,
                       f_side_input=None,
                       g_side_input=None,
                       gate_outputs=False):
  """Forward for a series of reversible layers."""
  out = (x1, x2)
  for i in xrange(num_layers):
    out = _rev_layer_forward(
        out, f[i], g[i], f_side_input, g_side_input, gate_outputs=gate_outputs)

  y1, y2 = out
  return y1, y2


class RevBlock(object):
  """Block of reversible layers. See rev_block."""

  def __init__(self,
               f,
               g,
               num_layers=1,
               f_side_input=None,
               g_side_input=None,
               use_efficient_backprop=True):

    if isinstance(f, list):
      assert len(f) == num_layers
    else:
      f = [f] * num_layers

    if isinstance(g, list):
      assert len(g) == num_layers
    else:
      g = [g] * num_layers

    scope_prefix = "revblock/revlayer_%d/"
    f_scope = scope_prefix + "f"
    g_scope = scope_prefix + "g"

    f = [
        tf.make_template(f_scope % i, fn, create_scope_now_=True)
        for i, fn in enumerate(f)
    ]
    g = [
        tf.make_template(g_scope % i, fn, create_scope_now_=True)
        for i, fn in enumerate(g)
    ]

    self.f = f
    self.g = g

    self.num_layers = num_layers
    self.f_side_input = f_side_input or []
    self.g_side_input = g_side_input or []

    self._use_efficient_backprop = use_efficient_backprop

  def _efficient_grad_fn(self, inputs, variables, ys, grad_ys):
    """Custom gradient fn for a block of reversible residual layers."""
    side_inputs = inputs[2:]
    f_side_idxs = [None] * len(self.f_side_input)
    g_side_idxs = [None] * len(self.g_side_input)
    assert len(side_inputs) == len(self.f_side_input) + len(self.g_side_input)

    for i, t in enumerate(side_inputs):
      if t in self.f_side_input:
        f_side_idxs[self.f_side_input.index(t)] = i
      elif t in self.g_side_input:
        g_side_idxs[self.g_side_input.index(t)] = i
      else:
        assert False

    f_vars = [[] for _ in range(self.num_layers)]
    g_vars = [[] for _ in range(self.num_layers)]
    f_vars_idxs = [[] for _ in range(self.num_layers)]
    g_vars_idxs = [[] for _ in range(self.num_layers)]

    for i, t in enumerate(variables):
      ref = common_layers.underlying_variable_ref(t)

      # Use the name to identify the layer number and function (f or g)
      regex = LAYER_RE.match(ref.name)
      layer_no = int(regex.group(1))
      fn_name = regex.group(2)
      if fn_name == "f":
        f_vars[layer_no].append(ref)
        f_vars_idxs[layer_no].append(i)
      else:
        assert fn_name == "g"
        g_vars[layer_no].append(ref)
        g_vars_idxs[layer_no].append(i)

    f_var_grads = []
    g_var_grads = []
    f_side_grads = []
    g_side_grads = []

    # Reverse variable containers to go backward
    f_vars.reverse()
    g_vars.reverse()
    f = list(self.f)
    g = list(self.g)
    f.reverse()
    g.reverse()

    for i in xrange(self.num_layers):
      ys, grad_ys, f_ret, g_ret = _rev_layer_backward(
          ys, grad_ys, f[i], g[i], f_vars[i], self.f_side_input, g_vars[i],
          self.g_side_input)

      grad_f_vars, grad_f_side = f_ret
      grad_g_vars, grad_g_side = g_ret
      f_var_grads.append(grad_f_vars)
      g_var_grads.append(grad_g_vars)
      f_side_grads.append(grad_f_side)
      g_side_grads.append(grad_g_side)

    # Accumulate layer gradients for f_side_input and g_side_input
    acc_f_side_grads = _acc_grads(*f_side_grads)
    acc_g_side_grads = _acc_grads(*g_side_grads)

    # Use the stored idxs to put gradients in the passed-in order.
    side_input_grads = [None] * len(side_inputs)
    variable_grads = [None] * len(variables)

    # Variable gradients were collected in reverse layer order. Reverse to match
    # idxs.
    f_var_grads.reverse()
    g_var_grads.reverse()
    for idxs, grads in list(zip(f_vars_idxs, f_var_grads)) + list(
        zip(g_vars_idxs, g_var_grads)):
      for i, grad in zip(idxs, grads):
        variable_grads[i] = grad

    for i, grad in zip(f_side_idxs, acc_f_side_grads):
      side_input_grads[i] = grad
    for i, grad in zip(g_side_idxs, acc_g_side_grads):
      side_input_grads[i] = grad

    grad_x1, grad_x2 = grad_ys
    return [grad_x1, grad_x2] + side_input_grads, variable_grads

  def forward(self, x1, x2):
    """Run forward through the reversible layers."""

    side_inputs = [self.f_side_input, self.g_side_input]
    flat_side_inputs = tf.contrib.framework.nest.flatten(side_inputs)

    custom_grad_fn = (
        self._efficient_grad_fn if self._use_efficient_backprop else None)

    @common_layers.fn_with_custom_grad(custom_grad_fn)
    def _forward(x1_, x2_, *flat_side_inputs):
      f_side, g_side = tf.contrib.framework.nest.pack_sequence_as(
          side_inputs, flat_side_inputs)
      return _rev_block_forward(
          x1_,
          x2_,
          self.f,
          self.g,
          num_layers=self.num_layers,
          f_side_input=f_side,
          g_side_input=g_side,
          gate_outputs=self._use_efficient_backprop)

    return _forward(x1, x2, *flat_side_inputs)

  def backward(self, y1, y2):
    """Run backward through the reversible layers."""

    f = list(self.f)
    g = list(self.g)
    f.reverse()
    g.reverse()

    for i in xrange(self.num_layers):
      gy1 = g[i](y1, self.g_side_input) if self.g_side_input else g[i](y1)
      x2 = y2 - gy1
      fx2 = f[i](x2, self.f_side_input) if self.f_side_input else f[i](x2)
      x1 = y1 - fx2

      y1, y2 = x1, x2

    return x1, x2


def rev_block(x1,
              x2,
              f,
              g,
              num_layers=1,
              f_side_input=None,
              g_side_input=None,
              is_training=True):
  """A block of reversible residual layers.

  A reversible residual layer is defined as:

  ```
  y1 = x1 + f(x2, f_side_input)
  y2 = x2 + g(y1, g_side_input)
  ```

  A reversible residual block, defined here, is a series of reversible residual
  layers.

  Limitations:
  * f and g must not close over any Tensors; all side inputs to f and g should
    be passed in with f_side_input and g_side_input which will be forwarded to
    f and g.
  * f and g must not change the dimensionality of their inputs in order for the
    addition in the equations above to work.

  Args:
    x1: a float Tensor.
    x2: a float Tensor.
    f: a function, (Tensor) -> (Tensor) (or list of such of length num_layers).
      Should not change the shape of the Tensor. Expected to create variables.
      See f_side_input if there are side inputs.
    g: a function, (Tensor) -> (Tensor) (or list of such of length num_layers).
      Should not change the shape of the Tensor. Expected to create variables.
      See g_side_input if there are side inputs.
    num_layers: int, number of reversible residual layers. Each layer will
      apply f and g according to the equations above, with new variables in each
      layer.
    f_side_input: list of Tensors, side input to f. If not None, signature of f
      should be (Tensor, list<Tensor>) -> (Tensor).
    g_side_input: list of Tensors, side input to g. If not None, signature of g
      should be (Tensor, list<Tensor>) -> (Tensor).
    is_training: bool, whether to actually use the efficient backprop codepath.

  Returns:
    y1, y2: tuple of float Tensors.
  """
  block = RevBlock(f, g, num_layers, f_side_input, g_side_input, is_training)
  return block.forward(x1, x2)


def recompute_grad(fn):
  """Decorator that recomputes the function on the backwards pass.

  Args:
    fn: a function that takes Tensors (all as positional arguments) and returns
      a tuple of Tensors.

  Returns:
    A wrapped fn that is identical to fn when called, but its activations will
    be discarded and recomputed on the backwards pass (i.e. on a call to
    tf.gradients).
  """

  @functools.wraps(fn)
  def wrapped(*args):
    return _recompute_grad(fn, args)

  return wrapped


def _recompute_grad(fn, args):
  """See recompute_grad."""

  cached_vs = []
  cached_arg_scope = []

  def grad_fn(inputs, variables, outputs, output_grads):
    """Recompute outputs for gradient computation."""
    del outputs
    # Recompute outputs
    with tf.control_dependencies(output_grads):
      with tf.contrib.framework.arg_scope(cached_arg_scope[0]):
        with tf.variable_scope(cached_vs[0], reuse=True):
          outputs = fn(*inputs)

    if not (isinstance(outputs, list) or isinstance(outputs, tuple)):
      outputs = [outputs]
    outputs = list(outputs)
    grads = tf.gradients(outputs, inputs + variables, output_grads)
    grad_inputs = grads[:len(inputs)]
    grad_vars = grads[len(inputs):]
    return grad_inputs, grad_vars

  @common_layers.fn_with_custom_grad(grad_fn)
  def fn_with_recompute(*args):
    cached_vs.append(tf.get_variable_scope())
    # TODO(rsepassi): Rm conditional in TF 1.5
    if hasattr(tf.contrib.framework, "current_arg_scope"):
      cached_arg_scope.append(tf.contrib.framework.current_arg_scope())
    else:
      cached_arg_scope.append({})
    return fn(*args)

  return fn_with_recompute(*args)
