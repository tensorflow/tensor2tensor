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

"""Diet varaibles are much more memory-efficient than regular variables.

Using diet variables, we can reduce memory overhead per parameter from
16 bytes to 2 bytes, allowing for up to 4B parameters per GPU.

This is an idea by rsepassi about how make this more generally useful.
with diet_variable_scope(diet_options=opts):
  custom variable getter that creates vars with diet_options
  per variable have fn that does the optimization acc to diet_options
@forward_with_diet_backwards fn decorator
"""


from collections import defaultdict
import math
# Dependency imports
from tensor2tensor.layers.common_layers import underlying_variable
import tensorflow as tf
from tensorflow.python.framework import function


def diet_adam_optimizer_params():
  """Default hyperparameters for a DietAdamOptimizer.

  Returns:
    a hyperparameters object.
  """
  return tf.contrib.training.HParams(
      quantize=int(True),  # use 16-bit fixed-point
      quantization_scale=10.0 / tf.int16.max,
      optimizer="factored_adam",
      learning_rate=1.0,
      learning_rate_warmup_steps=2000,
      learning_rate_decay_scheme="noam",  # "noam" or "none"
      epsilon=1e-10,
      beta1=0.0,  # we can save memory if beta1=0
      beta2=0.98,
      randomized_updates=int(True),  # use unbiased roundoff in updates
      factored_second_moment_accumulator=int(True),  # this saves memory
  )


class DietAdamOptimizer(object):
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

  def __init__(self, hparams):
    """Create a DietAdamOptimizer.

    Args:
      hparams: a hyperparameters object - see diet_adam_optimizer_params()
    """
    self._hparams = hparams
    self._global_step = tf.to_float(
        tf.contrib.framework.get_global_step()) + 1.0
    self._initializer_dependencies = defaultdict(list)

  @property
  def dtype(self):
    """The data type used for the variables."""
    return tf.float16 if self._hparams.quantize else tf.float32

  def get_variable(self, name, shape):
    """Create a diet variable.

    Args:
      name: a string
      shape: a list of integers

    Returns:
      a variable
    """
    var = tf.get_variable(
        name, shape, trainable=False,
        dtype=self.dtype,
        initializer=self._initializer())
    self._create_slots(var, name)
    var.optimizer = self
    return var

  def _create_slots(self, var, name):
    """Create auxiliary slots for a variable.

    Args:
      var: a tf.Variable
      name: a string
    """
    hparams = self._hparams
    shape = var.get_shape().as_list()
    if hparams.factored_second_moment_accumulator and len(shape) == 2:
      var.adam_vr = tf.get_variable(
          name + "_adam_vr", [shape[0], 1], trainable=False,
          initializer=tf.zeros_initializer())
      var.adam_vc = tf.get_variable(
          name + "_adam_vc", [1, shape[1]], trainable=False,
          initializer=tf.zeros_initializer())
    else:
      var.adam_v = tf.get_variable(
          name + "_adam_v", shape, trainable=False,
          initializer=tf.zeros_initializer())
    if hparams.beta1 != 0.0:
      var.adam_m = tf.get_variable(
          name + "_adam_m", shape, trainable=False,
          initializer=tf.zeros_initializer())

  def _quantize(self, x, randomize=True):
    """Quantize to tf.int16, then bitcast to tf.float16.

    The reason for float16 is that for some reason, tensorflow refuses to put
    integer variables on gpu.

    Args:
      x: a Tensor of type tf.float32
      randomize: a boolean

    Returns:
      a Tensor of type tf.float16
    """
    hparams = self._hparams
    if not hparams.quantize:
      return x
    if not randomize:
      return tf.bitcast(
          tf.cast(x / hparams.quantization_scale, tf.int16), tf.float16)
    abs_x = tf.abs(x)
    sign_x = tf.sign(x)
    y = abs_x / hparams.quantization_scale
    y = tf.floor(y + tf.random_uniform(tf.shape(x)))
    y = tf.minimum(y, tf.int16.max) * sign_x
    q = tf.bitcast(tf.cast(y, tf.int16), tf.float16)
    return q

  def dequantize(self, q):
    """Approximate inverse of _quantize().

    Args:
      q: a Tensor with type tf.float16

    Returns:
      a Tensor with type tf.float32
    """
    hparams = self._hparams
    if hparams.quantize:
      return tf.to_float(tf.bitcast(q, tf.int16)) * hparams.quantization_scale
    else:
      return q

  def _initializer(self):
    """Returns an initializer function.

    Returns:
      a function
    """
    hparams = self._hparams
    device = tf.constant(1.0).device
    def _initializer(shape, dtype=self.dtype, partition_info=None):
      assert dtype == self.dtype
      del partition_info
      # make sure no two initializers run simultaneously (to conserve memory)
      with tf.control_dependencies(self._initializer_dependencies[device]):
        float_range = math.sqrt(3)
        ret = tf.random_uniform(shape, -float_range, float_range)
        if hparams.quantize:
          ret = self._quantize(ret, randomize=False)
        self._initializer_dependencies[device] = [ret]
        return ret
    return _initializer

  def update(self, var, grad):
    """Update a diet varaible given a gradient.

    Args:
      var: a variable
      grad: a Tensor

    Returns:
      an update op.  Make sure that something depends on this
      op if you want it to run.
    """
    hparams = self._hparams
    var = underlying_variable(var)
    # compute learning rate
    lrate = hparams.learning_rate
    if hparams.learning_rate_decay_scheme == "noam":
      lrate *= tf.minimum(
          self._global_step * hparams.learning_rate_warmup_steps ** -1.5,
          self._global_step ** -0.5)
    else:
      assert hparams.learning_rate_decay_scheme == "none"
      lrate *= tf.minumum(
          self._global_step / hparams.learning_rate_warmup_steps, 1.0)
    # compute adjustment due to second moment
    grad_squared = tf.square(grad)
    beta2_pow = tf.pow(hparams.beta2, self._global_step)
    if hparams.factored_second_moment_accumulator and len(var.shape) == 2:
      vr_update = tf.assign(
          var.adam_vr,
          var.adam_vr * hparams.beta2 +
          tf.reduce_mean(grad_squared, 1, keep_dims=True) *
          (1.0 - hparams.beta2))
      vc_update = tf.assign(
          var.adam_vc,
          var.adam_vc * hparams.beta2 +
          tf.reduce_mean(grad_squared, 0, keep_dims=True) *
          (1.0 - hparams.beta2))
      with tf.control_dependencies([vr_update, vc_update]):
        vr = tf.sqrt(var.adam_vr / (1.0 - beta2_pow)) + hparams.epsilon
        vc = tf.sqrt(var.adam_vc / (1.0 - beta2_pow)) + hparams.epsilon
        vc /= tf.reduce_mean(vc)
        denom = vr * vc
    else:
      v_update = tf.assign(
          var.adam_v,
          var.adam_v * hparams.beta2 + grad_squared * (1.0 - hparams.beta2))
      with tf.control_dependencies([v_update]):
        denom = tf.sqrt(var.adam_v / (1.0 - beta2_pow)) + hparams.epsilon
    # compute momentum if applicable
    if hparams.beta1 != 0.0:
      m_update = tf.assign(
          var.adam_m, var.adam_m * hparams.beta1 + grad * (1.0 - hparams.beta1))
      with tf.control_dependencies([m_update]):
        grad = var.adam_m
    subtrahend = lrate * grad / denom
    new_val = self._quantize(self.dequantize(var) - subtrahend)
    return tf.assign(var, new_val)


def dependency_dict():
  """Get or create a defaultdict(list) that is stored in the default graph.

  This is used when we want to make sure that certain operations are performed
  sequentially.

  example use - make sure calls to foo on the same device execute sequentially:

  def foo(x, device)
    key = "foo " + device
    with tf.device(device):
      with tf.control_dependencies(dependency_dict()[key]):
        y = bar(x)
        dependency_dict()[key] = y
        return y

  Returns:
    a defaultdict whose default value is the empty list
  """
  if not hasattr(tf.get_default_graph(), "dependency_dict"):
    setattr(tf.get_default_graph(), "dependency_dict", defaultdict(list))
  return tf.get_default_graph().dependency_dict


def _diet_expert_internal(x, w0, w1):
  h = tf.matmul(x, w0)
  h = tf.nn.relu(h)
  y = tf.matmul(h, w1)
  y *= tf.rsqrt(tf.to_float(tf.shape(w0)[0] * tf.shape(w1)[0]))
  y.set_shape(x.get_shape())
  return y


def _diet_expert_grad(op, dy):
  x, w0, w1 = op.inputs
  w0_var = underlying_variable(w0)
  w1_var = underlying_variable(w1)
  key = "diet_expert_backward_deps " + dy.device
  with tf.control_dependencies(dependency_dict()[key]):
    w0 = w0_var.optimizer.dequantize(w0_var)
    w1 = w1_var.optimizer.dequantize(w1_var)
    y = _diet_expert_internal(x, w0, w1)
    dx, dw0, dw1 = tf.gradients(ys=[y], xs=[x, w0, w1], grad_ys=[dy])
    w0_update = w0_var.optimizer.update(w0_var, dw0)
    w1_update = w1_var.optimizer.update(w1_var, dw1)
    with tf.control_dependencies([w0_update, w1_update]):
      dx = tf.identity(dx)
      dependency_dict()[key] = [dx]
      return dx, None, None


def diet_expert(x, hidden_size, optimizer):
  """A two-layer feed-forward network with relu activation on hidden layer.

  Uses diet variables.
  Recompuets hidden layer on backprop to save activation memory.

  Args:
    x: a Tensor with shape [batch, io_size]
    hidden_size: an integer
    optimizer: a DietAdamOptimizer or some such class

  Returns:
    a Tensor with shape [batch, io_size]
  """
  @function.Defun(python_grad_func=_diet_expert_grad,
                  shape_func=lambda _: (x.get_shape(),))
  def _diet_expert_fn(x, w0, w1):
    w0 = optimizer.dequantize(w0)
    w1 = optimizer.dequantize(w1)
    return _diet_expert_internal(x, w0, w1)

  with tf.device(x.device):
    _, io_size = x.get_shape().as_list()
    w0_var = optimizer.get_variable("w0", [io_size, hidden_size])
    w1_var = optimizer.get_variable("w1", [hidden_size, io_size])
    key = "diet_expert_forward_deps " + x.device
    with tf.control_dependencies(dependency_dict()[key]):
      ret = _diet_expert_fn(x, w0_var, w1_var)
      dependency_dict()[key] = [ret]
      return ret
