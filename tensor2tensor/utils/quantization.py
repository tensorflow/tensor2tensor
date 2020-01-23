# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Utilities related to using bfloat16 activations and/or parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import function


def bfloat16_activations_var_getter(getter, *args, **kwargs):
  """A custom getter function for float32 parameters and bfloat16 activations.

  Args:
    getter: custom getter
    *args: arguments
    **kwargs: keyword arguments
  Returns:
    variables with the correct dtype.
  Raises:
    KeyError: if "dtype" is not provided as a kwarg.
  """
  requested_dtype = kwargs["dtype"]
  if requested_dtype == tf.bfloat16:
    kwargs["dtype"] = tf.float32
  var = getter(*args, **kwargs)
  # This if statement is needed to guard the cast, because batch norm
  # assigns directly to the return value of this custom getter. The cast
  # makes the return value not a variable so it cannot be assigned. Batch
  # norm variables are always in fp32 so this if statement is never
  # triggered for them.
  if var.dtype.base_dtype != requested_dtype:
    var = tf.cast(var, requested_dtype)
  return var


def float16_activations_var_getter(getter, *args, **kwargs):
  """A custom getter function for float32 parameters and float16 activations.

  This function ensures the following:
    1. All variables requested with type fp16 are stored as type fp32.
    2. All variables requested with type fp32 are returned as type fp16.
  See https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/
  #training_tensorflow for more information on this strategy.

  Args:
    getter: custom getter
    *args: arguments
    **kwargs: keyword arguments

  Returns:
    variables with the correct dtype.

  Raises:
    KeyError: if "dtype" is not provided as a kwarg.
  """
  requested_dtype = kwargs["dtype"]

  if requested_dtype == tf.float16:
    kwargs["dtype"] = tf.float32

  if requested_dtype == tf.float32:
    requested_dtype = tf.float16
  var = getter(*args, **kwargs)
  # This if statement is needed to guard the cast, because batch norm
  # assigns directly to the return value of this custom getter. The cast
  # makes the return value not a variable so it cannot be assigned. Batch
  # norm variables are always in fp32 so this if statement is never
  # triggered for them.
  if var.dtype.base_dtype != requested_dtype:
    var = tf.cast(var, requested_dtype)
  return var


def simulated_quantize(x, num_bits, noise):
  """Simulate quantization to num_bits bits, with externally-stored scale.

  num_bits is the number of bits used to store each value.
  noise is a float32 Tensor containing values in [0, 1).
  Each value in noise should take different values across
  different steps, approximating a uniform distribution over [0, 1).
  In the case of replicated TPU training, noise should be identical
  across replicas in order to keep the parameters identical across replicas.

  The natural choice for noise would be tf.random_uniform(),
  but this is not possible for TPU, since there is currently no way to seed
  the different cores to produce identical values across replicas.  Instead we
  use noise_from_step_num() (see below).

  The quantization scheme is as follows:

  Compute the maximum absolute value by row (call this max_abs).
  Store this either in an auxiliary variable or in an extra column.

  Divide the parameters by (max_abs / (2^(num_bits-1)-1)).  This gives a
  float32 value in the range [-2^(num_bits-1)-1, 2^(num_bits-1)-1]

  Unbiased randomized roundoff by adding noise and rounding down.

  This produces a signed integer with num_bits bits which can then be stored.

  Args:
    x: a float32 Tensor
    num_bits: an integer between 1 and 22
    noise: a float Tensor broadcastable to the shape of x.

  Returns:
    a float32 Tensor
  """
  shape = x.get_shape().as_list()
  if not (len(shape) >= 2 and shape[-1] > 1):
    return x
  max_abs = tf.reduce_max(tf.abs(x), -1, keepdims=True) + 1e-9
  max_int = 2 ** (num_bits - 1) - 1
  scale = max_abs / max_int
  x /= scale
  x = tf.floor(x + noise)
  # dequantize before storing (since this is a simulation)
  x *= scale
  return x


def noise_from_step_num():
  """Quantization noise equal to (phi * (step_num + 1)) mod 1.0.

  Not using random_uniform here due to a problem on TPU in that random seeds
  are not respected, which may cause the parameters on different replicas
  to go out-of-sync.

  Returns:
    a float32 scalar
  """
  step = tf.to_int32(tf.train.get_or_create_global_step()) + 1
  phi = ((5 ** 0.5) - 1) / 2
  # Naive computation tf.mod(phi * step, 1.0) in float32 would be disastrous
  # due to loss of precision when the step number gets large.
  # Computation in doubles does not work on TPU, so we use this complicated
  # alternative computation which does not suffer from these roundoff errors.
  ret = 0.0
  for i in range(30):
    ret += (((phi * (2 ** i)) % 1.0)  # double-precision computation in python
            * tf.to_float(tf.mod(step // (2 ** i), 2)))
  return tf.mod(ret, 1.0)


def _randomized_roundoff_to_bfloat16(x, noise, cand1, cand2):
  """Round-off x to cand1 or to cand2 in an unbiased way.

  Cand1 and cand2 are the same shape as x.
  For every element of x, the corresponding elements of cand1 and cand2 should
  be the two closest bfloat16 values to x.  Order does not matter.
  cand1 and cand2 must differ from each other.

  Args:
    x: A float32 Tensor.
    noise: A Tensor broadcastable to the shape of x containing
    random uniform values in [0.0, 1.0].
    cand1: A bfloat16 Tensor the same shape as x.
    cand2: A bfloat16 Tensor the same shape as x.

  Returns:
    A bfloat16 Tensor.
  """
  cand1_f = tf.to_float(cand1)
  cand2_f = tf.to_float(cand2)
  step_size = cand2_f - cand1_f
  fpart = (x - cand1_f) / step_size
  ret = tf.where(tf.greater(fpart, noise), cand2, cand1)
  return ret


def _to_bfloat16_unbiased(x, noise):
  """Convert a float32 to a bfloat16 using randomized roundoff.

  Args:
    x: A float32 Tensor.
    noise: a float32 Tensor with values in [0, 1), broadcastable to tf.shape(x)
  Returns:
    A float32 Tensor.
  """
  x_sign = tf.sign(x)
  # Make sure x is positive.  If it is zero, the two candidates are identical.
  x = x * x_sign + 1e-30
  cand1 = tf.to_bfloat16(x)
  cand1_f = tf.to_float(cand1)
  # This relies on the fact that for a positive bfloat16 b,
  # b * 1.005 gives you the next higher bfloat16 and b*0.995 gives you the
  # next lower one. Both 1.005 and 0.995 are ballpark estimation.
  cand2 = tf.to_bfloat16(
      tf.where(tf.greater(x, cand1_f), cand1_f * 1.005, cand1_f * 0.995))
  ret = _randomized_roundoff_to_bfloat16(x, noise, cand1, cand2)
  return ret * tf.to_bfloat16(x_sign)


class ParameterEncoding(object):
  """Helper class for encoding weights as bfloat16.

  For now, the parameters are always stored (encoded) as bfloat16 and decoded
  to bfloat32.  Confusingly, the custom getter then converts the bfloat32 back
  to a bfloat16 to use as an activation, assuming that we use bfloat16 for
  activations.

  TODO(noam): Add options for activation dtype=float32, and for different
  storage dtypes.
  """

  def encode(self, x, noise):
    """Encode float32 to bfloat16.

    Args:
      x: a float32 Tensor
      noise: a float32 Tensor with values in [0, 1), broadcastable to shape(x)

    Returns:
      a bfloat16 Tensor
    """
    raise NotImplementedError("encode not implemented")

  def decode(self, x):
    """Decode bfloat16 to float32."""
    raise NotImplementedError("decode not implemented")

  def _decode_with_identity_gradient(self, x):
    # identity backprop through the decoder.
    # This means that the optimizer must call encode when updating weights.
    @function.Defun(python_grad_func=lambda op, dy: dy,
                    shape_func=lambda op: [op.inputs[0].get_shape()])
    def my_fn(x):
      return self.decode(x)
    return my_fn(x)

  def custom_getter(self, activation_dtype=tf.bfloat16):
    """A custom getter that uses the encoding for bfloat16 and float32 vars.

    When a bfloat16 or float32 variable is requsted, an encoded float16
    varaible is created, which is then decoded and cast to a bfloat16
    activation.

    Args:
      activation_dtype: a dtype to which to convert the decoded value.

    Returns:
      a function.
    """
    def getter_fn(getter, *args, **kwargs):
      requested_dtype = kwargs["dtype"]
      if requested_dtype in (tf.bfloat16, tf.float32):
        kwargs["dtype"] = tf.bfloat16
        kwargs["initializer"] = _EncodingInitializer(
            kwargs["initializer"], self)
        ret = self._decode_with_identity_gradient(getter(*args, **kwargs))
        return tf.cast(ret, activation_dtype)
      return getter(*args, **kwargs)
    return getter_fn


class _EncodingInitializer(object):
  """Helper class for ParameterEncoding.

  Initializes variables by calling base initializer, then encoding.
  """

  def __init__(self, base_initializer, parameter_encoding):
    self._base_initializer = base_initializer
    self._parameter_encoding = parameter_encoding

  def __call__(self, shape, dtype, partition_info=None):
    if self._base_initializer is None:
      # mimic default initialization in tf.get_variable()
      if dtype.is_floating:
        ret = tf.glorot_uniform_initializer()(shape, dtype)
      else:
        ret = tf.zeros(shape, dtype)
    else:
      ret = self._base_initializer(shape, dtype, partition_info=partition_info)
    noise = 0.0  # no random noise in the initializer.
    return tf.cast(self._parameter_encoding.encode(ret, noise), dtype)


class EighthPowerEncoding(ParameterEncoding):
  """enc(x) = sign(x) * (abs(x)*128)^8.

  This provides less range and more resolution.
  The range of representable positive values is approximately [2^-23, 2^9]
  Resolution is 8x better than bfloat16.
  """

  def encode(self, x, noise):
    x = tf.to_float(x)
    # we can't use tf.pow(..., 8.0) because of a high-error approximation
    # on TPU.  Instead we square three times.
    x = tf.sign(x) * tf.square(tf.square(tf.square(tf.abs(x) * 128.0)))
    x = _to_bfloat16_unbiased(x, noise)
    return x

  def decode(self, x):
    x = tf.to_float(x)
    # we can't use tf.pow(..., 0.125) because of a high-error approximation
    # on TPU.  Instead we sqrt three times.
    return tf.sign(x) * (tf.sqrt(tf.sqrt(tf.sqrt(tf.abs(x)))) / 128.0)
