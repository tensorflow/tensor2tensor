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

"""Diet variables are much more memory-efficient than regular variables.

Using diet variables, we can reduce memory overhead per parameter from
16 bytes to 2 bytes, allowing for up to 4B parameters per GPU.

Functions that build subgraphs with variables can be made to use diet variables
by using the fn_with_diet_vars decorator.
"""

from collections import defaultdict
import copy
import math
# Dependency imports
from tensor2tensor.layers import common_layers
import tensorflow as tf


def diet_adam_optimizer_params():
  """Default hyperparameters for a DietAdamOptimizer.

  Returns:
    a hyperparameters object.
  """
  return tf.contrib.training.HParams(
      quantize=True,  # use 16-bit fixed-point
      quantization_scale=10.0 / tf.int16.max,
      optimizer="DietAdam",
      learning_rate=1.0,
      learning_rate_warmup_steps=2000,
      learning_rate_decay_scheme="noam",  # "noam" or "none"
      epsilon=1e-10,
      beta1=0.0,  # we can save memory if beta1=0
      beta2=0.98,
      factored_second_moment_accumulator=True,  # this saves memory
  )


def diet_expert(x, hidden_size, params):
  """A two-layer feed-forward network with relu activation on hidden layer.

  Uses diet variables.
  Recompuets hidden layer on backprop to save activation memory.

  Args:
    x: a Tensor with shape [batch, io_size]
    hidden_size: an integer
    params: a diet variable HParams object.

  Returns:
    a Tensor with shape [batch, io_size]
  """

  @fn_with_diet_vars(params)
  def diet_expert_internal(x):
    dim = x.get_shape().as_list()[-1]
    h = tf.layers.dense(x, hidden_size, activation=tf.nn.relu, use_bias=False)
    y = tf.layers.dense(h, dim, use_bias=False)
    y *= tf.rsqrt(tf.to_float(dim * hidden_size))
    return y

  return diet_expert_internal(x)


class DietVariableOptimizer(object):
  """Base class for Diet variable optimizers."""

  def __init__(self, params):
    self._params = params
    self._global_step = tf.train.get_or_create_global_step()

  @property
  def params(self):
    return self._params

  @property
  def global_step(self):
    return self._global_step

  def create_slots(self, var):
    raise NotImplementedError()

  def update_variable(self, var, grad_var):
    raise NotImplementedError()


class DietAdamOptimizer(DietVariableOptimizer):
  """A memory efficient optimizer for memory-efficient variables.

  We employ the following techniques:
   - 16-bit fixed-point quantization
   - inline updates during backprop, instead of through the optimizer.  This
     keeps the gradients from staying around in memory.
   - momentum is optional - saves a slot if it is off (beta1=0.0).
   - "factored second-moment accumulator"
      (keep row-wise and col-wise averages instead of full accumulator)
   - tighter control over operation ordering to make sure that only a small
     portion of the decompressed variables and of the variable gradients
     are resident in memory at any given time.

  All together these techniques reduce the memory footprint per parameter to
  a little over 2 bytes, allowing for roughly 4B parameters per GPU.   This is
  roughly an 8x improvement over the naive version.

  Usage:

  Diet variables should be created with the
  DietAdamOptimizer.get_variable() method.  The resulting variables
  have extra fields pointing to the otpimizer and to the accumulator
  slots.

  The variable is kept in quantized form, so you need to call
  var.optimizer.dequantize(var) to get the value.

  The variables are created with trainable=False, so that they will
  not be optimized by an ordinary optimizer.  Instead, the user is
  responsible for making sure that var.optimizer.update(var, grad) is
  called during backprop.  The reason for this inline update is to
  avoid keeping around the gradients for all variables at once.  This
  is done with the clever use of defuns and control dependencies.  See
  diet_expert() for an example of how all of this is done.

  To facilitate fixed-point quantization and to make it easier to
  choose a learning rate, all varaibles are initialized with unit
  normal initialization.  If you want smaller values, downscale on the
  outside.
  """

  def create_slots(self, var):
    """Create the factorized Adam accumulators for diet variables."""
    params = self.params
    shape = var.get_shape().as_list()

    if not hasattr(params, "slots"):
      params.slots = defaultdict(dict)

    name = var.op.name
    slots = params.slots[name]

    if params.factored_second_moment_accumulator and len(shape) == 2:
      slots["adam_vr"] = tf.get_variable(
          name + "_adam_vr", [shape[0], 1],
          trainable=False,
          initializer=tf.zeros_initializer())
      slots["adam_vc"] = tf.get_variable(
          name + "_adam_vc", [1, shape[1]],
          trainable=False,
          initializer=tf.zeros_initializer())
    else:
      slots["adam_v"] = tf.get_variable(
          name + "_adam_v",
          shape,
          trainable=False,
          initializer=tf.zeros_initializer())
    if params.beta1 != 0.0:
      slots["adam_m"] = tf.get_variable(
          name + "_adam_m",
          shape,
          trainable=False,
          initializer=tf.zeros_initializer())

  def update_variable(self, var, grad_var):
    """Update the variable and its slots."""
    params = self.params
    global_step = tf.to_float(self.global_step) + 1

    # compute learning rate
    lrate = params.learning_rate
    if params.learning_rate_decay_scheme == "noam":
      lrate *= tf.minimum(global_step * params.learning_rate_warmup_steps**-1.5,
                          global_step**-0.5)
    else:
      assert params.learning_rate_decay_scheme == "none"
      lrate *= tf.minumum(global_step / params.learning_rate_warmup_steps, 1.0)

    # compute adjustment due to second moment
    slots = params.slots[var.op.name]
    grad_squared = tf.square(grad_var)
    beta2_pow = tf.pow(params.beta2, global_step)
    if params.factored_second_moment_accumulator and len(var.shape) == 2:
      vr_update = tf.assign(slots["adam_vr"], slots["adam_vr"] * params.beta2 +
                            tf.reduce_mean(grad_squared, 1, keep_dims=True) *
                            (1.0 - params.beta2))
      vc_update = tf.assign(slots["adam_vc"], slots["adam_vc"] * params.beta2 +
                            tf.reduce_mean(grad_squared, 0, keep_dims=True) *
                            (1.0 - params.beta2))
      with tf.control_dependencies([vr_update, vc_update]):
        vr = tf.sqrt(slots["adam_vr"] / (1.0 - beta2_pow)) + params.epsilon
        vc = tf.sqrt(slots["adam_vc"] / (1.0 - beta2_pow)) + params.epsilon
        vc /= tf.reduce_mean(vc)
        denom = vr * vc
    else:
      v_update = tf.assign(slots["adam_v"],
                           slots["adam_v"] * params.beta2 + grad_squared *
                           (1.0 - params.beta2))
      with tf.control_dependencies([v_update]):
        denom = tf.sqrt(slots["adam_v"] / (1.0 - beta2_pow)) + params.epsilon

    # compute momentum if applicable
    if params.beta1 != 0.0:
      m_update = tf.assign(slots["adam_m"],
                           slots["adam_m"] * params.beta1 + grad_var *
                           (1.0 - params.beta1))
      with tf.control_dependencies([m_update]):
        grad_var = slots["adam_m"]

    # update var
    subtrahend = lrate * grad_var / denom
    new_val = _quantize(_dequantize(var, params) - subtrahend, params)
    return tf.assign(var, new_val)


def _create_diet_optimizer(params):
  if params.optimizer == "DietAdam":
    return DietAdamOptimizer(params)
  else:
    raise ValueError("Unrecognized diet optimizer")


def _quantize(x, params, randomize=True):
  """Quantize x according to params, optionally randomizing the rounding."""
  if not params.quantize:
    return x

  if not randomize:
    return tf.bitcast(
        tf.cast(x / params.quantization_scale, tf.int16), tf.float16)

  abs_x = tf.abs(x)
  sign_x = tf.sign(x)
  y = abs_x / params.quantization_scale
  y = tf.floor(y + tf.random_uniform(common_layers.shape_list(x)))
  y = tf.minimum(y, tf.int16.max) * sign_x
  q = tf.bitcast(tf.cast(y, tf.int16), tf.float16)
  return q


def _dequantize(q, params):
  """Dequantize q according to params."""
  if not params.quantize:
    return q
  return tf.to_float(tf.bitcast(q, tf.int16)) * params.quantization_scale


def make_diet_var_getter(params):
  """Create a custom variable getter for diet variables according to params."""

  def diet_var_initializer(shape, dtype, partition_info=None):
    del dtype
    del partition_info

    with common_layers.fn_device_dependency("diet_init") as out_deps:
      float_range = math.sqrt(3)
      ret = tf.random_uniform(shape, -float_range, float_range)
      if params.quantize:
        ret = _quantize(ret, params, randomize=False)
      out_deps.append(ret)
      return ret

  def diet_var_getter(getter, **kwargs):
    """Get diet variable and return it dequantized."""
    if params.quantize:
      kwargs["dtype"] = tf.float16
    kwargs["initializer"] = diet_var_initializer
    kwargs["trainable"] = False

    base_var = getter(**kwargs)

    dequantized = _dequantize(base_var, params)

    if not hasattr(params, "dequantized"):
      params.dequantized = defaultdict(list)
    params.dequantized[base_var.name].append(dequantized)

    return dequantized

  return diet_var_getter


def _fn_with_diet_vars(fn, args, params):
  """Call function with args; use diet variables according to params."""

  vs_ctr = []

  def grad_fn(inputs, variables, outputs, output_grads):
    del outputs  # recomputing below
    with common_layers.fn_device_dependency("diet_grad",
                                            output_grads[0].device) as out_dep:
      with tf.variable_scope(vs_ctr[0], reuse=True):
        outputs = fn(*inputs)

      variables = [common_layers.underlying_variable_ref(v) for v in variables]
      dequantized_variables = [
          params.dequantized[v.name][-1] for v in variables
      ]

      grads = tf.gradients(outputs, inputs + dequantized_variables,
                           output_grads)
      grad_inputs = grads[:len(inputs)]
      grad_variables = grads[len(inputs):]

      opt = _create_diet_optimizer(params)

      # Apply grad_variables here
      var_updates = []
      for v, dv in zip(variables, grad_variables):
        with tf.variable_scope(vs_ctr[0].name):
          opt.create_slots(v)
        update_op = opt.update_variable(v, dv)
        var_updates.append(update_op)

      with tf.control_dependencies(var_updates):
        grad_inputs = [tf.identity(dx) for dx in grad_inputs]

      out_dep.append(grad_inputs)

      return grad_inputs, [None] * len(variables)

  @common_layers.fn_with_custom_grad(grad_fn, use_global_vars=True)
  def forward(*inputs):
    with tf.variable_scope(
        None, default_name="diet",
        custom_getter=make_diet_var_getter(params)) as vs:
      vs_ctr.append(vs)
      outputs = fn(*inputs)
      return outputs

  with common_layers.fn_device_dependency("diet_forward",
                                          args[0].device) as out_dep:
    outputs = forward(*args)
    out_dep.append(outputs)
  return outputs


def fn_with_diet_vars(params):
  """Decorator for graph-building function to use diet variables."""
  params = copy.copy(params)

  def dec(fn):

    def wrapped(*args):
      return _fn_with_diet_vars(fn, args, params)

    return wrapped

  return dec
