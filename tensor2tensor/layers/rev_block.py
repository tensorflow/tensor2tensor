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
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

# Dependency imports

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function

LAYER_RE = re.compile(".*revlayer_([0-9]*)/([fg])/.*")


def _rev_layer_forward(xs, f, g):
  """Forward for 1 reversible layer."""
  x1, x2 = xs
  with tf.variable_scope("f"):
    y1 = x1 + f(x2)
  with tf.variable_scope("g"):
    y2 = x2 + g(y1)
  return tf.tuple([y1, y2])


def _rev_layer_backward(ys, grad_ys, f, g, f_vars, g_vars):
  """Backprop for 1 layer."""
  y1, y2 = ys
  grad_y1, grad_y2 = grad_ys

  # Reconstruct intermediates and inputs (x1, x2)
  # stop_gradients required on y1 and x2 to prevent infinite recursion into this
  # grad function on the calls to tf.gradients.
  y1_stop = tf.stop_gradient(y1)
  with tf.variable_scope("g"):
    gy1 = g(y1_stop)

  x2 = y2 - gy1
  x2_stop = tf.stop_gradient(x2)
  with tf.variable_scope("f"):
    fx2 = f(x2_stop)

  x1 = y1 - fx2

  # Compute gradients wrt to inputs
  # dL/dy2 * dG(y1)/y1
  grad_gy1_y2 = tf.gradients(gy1, y1_stop, grad_y2, gate_gradients=True)[0]
  grad_x1 = grad_y1 + grad_gy1_y2
  grad_x2 = (
      tf.gradients(fx2, x2_stop, grad_y1, gate_gradients=True)[0] + grad_y2 +
      tf.gradients(fx2, x2_stop, grad_gy1_y2, gate_gradients=True)[0])

  # Compute gradients wrt to vars in f and g
  grad_g_vars = tf.gradients(gy1, g_vars, grad_y2, gate_gradients=True)
  grad_f_y1 = tf.gradients(fx2, f_vars, grad_y1, gate_gradients=True)
  grad_f_y2 = tf.gradients(fx2, f_vars, grad_gy1_y2, gate_gradients=True)
  grad_f_vars = [tf.add_n(grads) for grads in zip(grad_f_y1, grad_f_y2)]

  # Put returns in a tuple to ensure a constant memory budget (i.e. don't want
  # the subsequent layer to start computing and consuming memory based on a
  # subset of these values).
  outs = tf.tuple([x1, x2, grad_x1, grad_x2] + grad_f_vars + grad_g_vars)
  x1, x2, grad_x1, grad_x2 = outs[:4]
  grad_f_vars = outs[4:4 + len(grad_f_vars)]
  grad_g_vars = outs[4 + len(grad_f_vars):]

  return (x1, x2), (grad_x1, grad_x2), grad_f_vars, grad_g_vars


def _rev_block_forward(x, f, g, num_layers=1, layer_scopes=None, name=None):
  """Forward for a series of reversible layers."""
  x1, x2 = tf.split(x, 2, axis=len(x.get_shape()) - 1)
  out = (x1, x2)
  with tf.variable_scope(name, default_name="revblock"):
    for i in xrange(num_layers):
      with tf.variable_scope("revlayer_%d" % i) as layer_vs:
        if layer_scopes is not None:
          layer_scopes.append(layer_vs)
        out = _rev_layer_forward(out, f, g)

  y1, y2 = out
  y = tf.concat([y1, y2], axis=-1)
  return y


def rev_block(x, f, g, num_layers=1, is_training=True):
  """A block of reversible residual layers.

  A reversible residual layer is defined as:

  ```
  x1, x2 = tf.split(x, 2, axis=-1)
  y1 = x1 + f(x2)
  y2 = x2 + g(y1)
  y = tf.concat([y1, y2], axis=-1)
  ```

  Args:
    x: a float Tensor, input, will be split evenly across the last dim.
    f: a function, (Tensor) -> (Tensor). Should not change the shape of the
      Tensor. May create variables. Should NOT close over any Tensor values.
    g: a function, (Tensor) -> (Tensor). Should not change the shape of the
      Tensor. May create variables. Should NOT close over any Tensor values.
    num_layers: int, number of reversible residual layers. Each layer will
      apply f and g according to the equations above, with new variables in each
      layer.
    is_training: bool, whether to actually use the efficient backprop codepath.

  Returns:
    y: a float Tensor, output.
  """
  layer_scopes = []

  def rev_block_grad(op, grad_y):
    """Custom gradient fn for a block of reversible residual layers."""
    y = op.outputs[0]
    ys = tf.split(y, 2, axis=len(y.get_shape()) - 1)
    grad_ys = tf.split(grad_y, 2, axis=len(y.get_shape()) - 1)

    # Find all variables from f and from g
    # Keep track of their positions in all_vars
    all_vars = op.inputs[1:]
    f_vars = [[] for _ in range(num_layers)]
    g_vars = [[] for _ in range(num_layers)]
    f_vars_idxs = [[] for _ in range(num_layers)]
    g_vars_idxs = [[] for _ in range(num_layers)]

    for i, v in enumerate(all_vars):
      ref = v.op.inputs[0]
      assert ref.dtype == dtypes.float32_ref
      regex = LAYER_RE.match(v.name)
      layer_no = int(regex.group(1))
      fn_name = regex.group(2)
      if fn_name == "f":
        f_vars[layer_no].append(ref)
        f_vars_idxs[layer_no].append(i)
      else:
        assert fn_name == "g"
        g_vars[layer_no].append(ref)
        g_vars_idxs[layer_no].append(i)

    f_grads = []
    g_grads = []

    # Reverse state containers to go backward
    layer_scopes.reverse()
    f_vars.reverse()
    g_vars.reverse()

    for i in xrange(num_layers):
      with tf.variable_scope(layer_scopes[i], reuse=True):
        ys, grad_ys, grad_f_vars, grad_g_vars = _rev_layer_backward(
            ys, grad_ys, f, g, f_vars[i], g_vars[i])
        f_grads.append(grad_f_vars)
        g_grads.append(grad_g_vars)

    # Gradients were collected in reverse layer order
    f_grads.reverse()
    g_grads.reverse()

    # Reorder the gradients so they match the original order of all_vars
    var_grads = [None] * len(all_vars)
    for idxs, grads in zip(f_vars_idxs, f_grads) + zip(g_vars_idxs, g_grads):
      for i, grad in zip(idxs, grads):
        var_grads[i] = grad

    grad_x = tf.concat(grad_ys, axis=-1)
    all_grads = [grad_x] + var_grads
    return all_grads

  @function.Defun(
      tf.float32,
      python_grad_func=rev_block_grad,
      shape_func=lambda _: [x.get_shape()])
  def rev_block_defun(inp):
    inp.set_shape(x.get_shape())
    return _rev_block_forward(
        inp, f, g, num_layers=num_layers, layer_scopes=layer_scopes)

  if is_training:
    return rev_block_defun(x)
  else:
    return _rev_block_forward(x, f, g, num_layers=num_layers)
