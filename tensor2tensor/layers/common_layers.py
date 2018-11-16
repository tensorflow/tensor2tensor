# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Layers common to multiple models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import contextlib
import functools
from functools import partial
import math

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import inplace_ops


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


def is_xla_compiled():
  """Whether we are building graph that will be compiled by XLA.

  This checks whether the code is executing within an XLA context.

  If True, model authors should ensure the graph they build is compilable by
  XLA. Specifically, they should ensure that all ops have XLA implementations
  and that all shapes are statically known.

  Returns:
    bool, whether the current graph will be compiled for XLA.
  """
  ctxt = tf.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  return control_flow_util.GetContainingXLAContext(ctxt) is not None


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
  """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.

  Instead of specifying noise_shape, this function takes broadcast_dims -
  a list of dimension numbers in which noise_shape should be 1.  The random
  keep/drop tensor has dimensionality 1 along these dimensions.

  Args:
    x: a floating point tensor.
    keep_prob: A scalar Tensor with the same type as x.
      The probability that each element is kept.
    broadcast_dims: an optional list of integers
      the dimensions along which to broadcast the keep/drop flags.
    **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".

  Returns:
    Tensor of the same shape as x.
  """
  assert "noise_shape" not in kwargs
  if broadcast_dims:
    shape = tf.shape(x)
    ndims = len(x.get_shape())
    # Allow dimensions like "-1" as well.
    broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in range(ndims)
    ]
  return tf.nn.dropout(x, keep_prob, **kwargs)


def comma_separated_string_to_integer_list(s):
  return [int(i) for i in s.split(",") if i]


def saturating_sigmoid(x):
  """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
  with tf.name_scope("saturating_sigmoid", values=[x]):
    y = tf.sigmoid(x)
    return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1))


def hard_sigmoid(x, saturation_limit=0.9):
  saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
  x_shifted = 0.5 * x + 0.5
  return tf.minimum(1.0, tf.nn.relu(x_shifted)), saturation_cost


def hard_tanh(x, saturation_limit=0.9):
  saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
  return tf.minimum(1.0, tf.maximum(x, -1.0)), saturation_cost


def inverse_exp_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay exponentially from 0.01 to 1.0 reached at max_step."""
  inv_base = tf.exp(tf.log(min_value) / float(max_step))
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = tf.to_float(step)
  return inv_base**tf.maximum(float(max_step) - step, 0.0)


def inverse_lin_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay linearly from 0.01 to 1.0 reached at max_step."""
  if step is None:
    step = tf.train.get_global_step()
  if step is None:
    return 1.0
  step = tf.to_float(step)
  progress = tf.minimum(step / float(max_step), 1.0)
  return progress * (1.0 - min_value) + min_value


def shakeshake2_py(x, y, equal=False, individual=False):
  """The shake-shake sum of 2 tensors, python version."""
  if equal:
    alpha = 0.5
  elif individual:
    alpha = tf.random_uniform(tf.get_shape(x)[:1])
  else:
    alpha = tf.random_uniform([])

  return alpha * x + (1.0 - alpha) * y


@function.Defun()
def shakeshake2_grad(x1, x2, dy):
  """Overriding gradient for shake-shake of 2 tensors."""
  y = shakeshake2_py(x1, x2)
  dx = tf.gradients(ys=[y], xs=[x1, x2], grad_ys=[dy])
  return dx


@function.Defun()
def shakeshake2_indiv_grad(x1, x2, dy):
  """Overriding gradient for shake-shake of 2 tensors."""
  y = shakeshake2_py(x1, x2, individual=True)
  dx = tf.gradients(ys=[y], xs=[x1, x2], grad_ys=[dy])
  return dx


@function.Defun()
def shakeshake2_equal_grad(x1, x2, dy):
  """Overriding gradient for shake-shake of 2 tensors."""
  y = shakeshake2_py(x1, x2, equal=True)
  dx = tf.gradients(ys=[y], xs=[x1, x2], grad_ys=[dy])
  return dx


@function.Defun(grad_func=shakeshake2_grad)
def shakeshake2(x1, x2):
  """The shake-shake function with a different alpha for forward/backward."""
  return shakeshake2_py(x1, x2)


@function.Defun(grad_func=shakeshake2_indiv_grad)
def shakeshake2_indiv(x1, x2):
  return shakeshake2_py(x1, x2, individual=True)


@function.Defun(grad_func=shakeshake2_equal_grad)
def shakeshake2_eqgrad(x1, x2):
  """The shake-shake function with a different alpha for forward/backward."""
  return shakeshake2_py(x1, x2)


def shakeshake(xs, equal_grad=False):
  """Multi-argument shake-shake, currently approximated by sums of 2."""
  if len(xs) == 1:
    return xs[0]
  div = (len(xs) + 1) // 2
  arg1 = shakeshake(xs[:div], equal_grad=equal_grad)
  arg2 = shakeshake(xs[div:], equal_grad=equal_grad)
  if equal_grad:
    return shakeshake2_eqgrad(arg1, arg2)
  return shakeshake2(arg1, arg2)


def convert_rgb_to_real(x):
  """Conversion of pixel values to real numbers."""
  with tf.name_scope("rgb_to_real", values=[x]):
    x = tf.to_float(x)
    x /= 255.0
    return x


def convert_rgb_to_symmetric_real(x):
  """Conversion of pixel values to real numbers."""
  with tf.name_scope("rgb_to_real", values=[x]):
    x = tf.to_float(x)
    # Convert each pixel intensity in [0, 1, 2, ..., 255] into a real number in
    # the range [-1, 1].
    x = (x / 127.5) - 1
    return x


def convert_real_to_rgb(x):
  """Conversion of real numbers to pixel values."""
  with tf.name_scope("real_to_rgb", values=[x]):
    x *= 255.0
    return x


def expand_squeeze_to_nd(x, n, squeeze_dim=2, expand_dim=-1):
  """Make x n-d with squeeze and expand_dims."""
  if len(x.shape) > n:
    while len(x.shape) != n:
      x = tf.squeeze(x, [squeeze_dim])
  else:
    while len(x.shape) != n:
      x = tf.expand_dims(x, expand_dim)
  return x


def standardize_images(x):
  """Image standardization on batches and videos."""
  with tf.name_scope("standardize_images", [x]):
    x_shape = shape_list(x)
    x = tf.to_float(tf.reshape(x, [-1] + x_shape[-3:]))
    x_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x_variance = tf.reduce_mean(
        tf.square(x - x_mean), axis=[1, 2], keepdims=True)
    num_pixels = tf.to_float(x_shape[-2] * x_shape[-3])
    x = (x - x_mean) / tf.maximum(tf.sqrt(x_variance), tf.rsqrt(num_pixels))
    return tf.reshape(x, x_shape)


def flatten4d3d(x):
  """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
  xshape = shape_list(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  return result


# TODO(noam): remove this function after TPUs do gather faster.
def gather(params, indices):
  """Version of tf.gather that works faster on tpu."""
  indices_flat = tf.reshape(indices, [-1])
  out = tf.gather(params, indices_flat)
  out = reshape_like(out, tf.expand_dims(indices, -1))
  return out


# TODO(noam): remove this function after TPUs do cumsum faster.
def cumsum(x, axis=0, exclusive=False):
  """TPU hack for tf.cumsum.

  This is equivalent to tf.cumsum and is faster on TPU as of 04/2018 unless
  the axis dimension is very large.

  Args:
    x: a Tensor
    axis: an integer
    exclusive: a boolean

  Returns:
    Tensor of the same shape as x.
  """
  if not is_xla_compiled():
    return tf.cumsum(x, axis=axis, exclusive=exclusive)
  x_shape = shape_list(x)
  rank = len(x_shape)
  length = x_shape[axis]
  my_range = tf.range(length)
  comparator = tf.less if exclusive else tf.less_equal
  mask = tf.cast(
      comparator(tf.expand_dims(my_range, 1), tf.expand_dims(my_range, 0)),
      x.dtype)
  ret = tf.tensordot(x, mask, axes=[[axis], [0]])
  if axis != rank - 1:
    ret = tf.transpose(
        ret,
        list(range(axis)) + [rank - 1] + list(range(axis, rank - 1)))
  return ret


def dropout_no_scaling(x, keep_prob):
  """Like tf.nn.dropout, but does not scale up.  Works on integers also.

  Args:
    x: a Tensor
    keep_prob: a floating point number

  Returns:
    Tensor of the same shape as x.
  """
  if keep_prob == 1.0:
    return x
  mask = tf.less(tf.random_uniform(tf.shape(x)), keep_prob)
  return x * cast_like(mask, x)


def embedding(x,
              vocab_size,
              dense_size,
              name=None,
              reuse=None,
              multiplier=1.0,
              symbol_dropout_rate=0.0,
              embedding_var=None,
              dtype=tf.float32):
  """Embed x of type int64 into dense vectors, reducing to max 4 dimensions."""
  with tf.variable_scope(
      name, default_name="embedding", values=[x], reuse=reuse, dtype=dtype):
    if embedding_var is None:
      embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
    # On the backwards pass, we want to convert the gradient from
    # an indexed-slices to a regular tensor before sending it back to the
    # parameter server. This avoids excess computation on the parameter server.
    if not tf.contrib.eager.in_eager_mode():
      embedding_var = convert_gradient_to_tensor(embedding_var)
    x = dropout_no_scaling(x, 1.0 - symbol_dropout_rate)
    emb_x = gather(embedding_var, x)
    if multiplier != 1.0:
      emb_x *= multiplier
    static_shape = emb_x.shape.as_list()
    if len(static_shape) < 5:
      return emb_x
    assert len(static_shape) == 5
    # If we had an extra channel dimension, assume it's 1, i.e. shape[3] == 1.
    return tf.squeeze(emb_x, 3)


def shift_right(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
  return shifted_targets


def shift_right_3d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
  return shifted_targets


def shift_right_2d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0]])[:, :-1]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1]
  return shifted_targets


def conv_stride2_multistep(x, nbr_steps, output_filters, name=None, reuse=None):
  """Use a strided convolution to downsample x by 2, `nbr_steps` times.

  We use stride and filter size 2 to avoid the checkerboard problem of deconvs.
  As detailed in http://distill.pub/2016/deconv-checkerboard/.

  Args:
    x: a `Tensor` with shape `[batch, spatial, depth]` or
     `[batch, spatial_1, spatial_2, depth]`
    nbr_steps: number of halving downsample rounds to apply
    output_filters: an int specifying the filter count for the convolutions
    name: a string
    reuse: a boolean

  Returns:
    a `Tensor` with shape `[batch, spatial / (2**nbr_steps), output_filters]` or
     `[batch, spatial_1 / (2**nbr_steps), spatial_2 / (2**nbr_steps),
       output_filters]`
  """
  with tf.variable_scope(
      name, default_name="conv_stride2_multistep", values=[x], reuse=reuse):
    if nbr_steps == 0:
      out = conv(x, output_filters, (1, 1))
      return out, [out]
    hidden_layers = [x]
    for i in range(nbr_steps):
      hidden_layers.append(
          conv(
              hidden_layers[-1],
              output_filters, (2, 2),
              strides=2,
              activation=tf.nn.relu,
              name="conv" + str(i)))
    return hidden_layers[-1], hidden_layers


def deconv_stride2_multistep(x,
                             nbr_steps,
                             output_filters,
                             name=None,
                             reuse=None):
  """Use a deconvolution to upsample x by 2**`nbr_steps`.

  Args:
    x: a `Tensor` with shape `[batch, spatial, depth]` or
     `[batch, spatial_1, spatial_2, depth]`
    nbr_steps: an int specifying the number of doubling upsample rounds to
     apply.
    output_filters: an int specifying the filter count for the deconvolutions
    name: a string
    reuse: a boolean

  Returns:
    a `Tensor` with shape `[batch, spatial * (2**nbr_steps), output_filters]` or
     `[batch, spatial_1 * (2**nbr_steps), spatial_2 * (2**nbr_steps),
       output_filters]`
  """
  with tf.variable_scope(
      name, default_name="deconv_stride2_multistep", values=[x], reuse=reuse):

    def deconv1d(cur, i):
      cur_shape = shape_list(cur)
      thicker = conv(
          cur,
          output_filters * 2, (1, 1),
          padding="SAME",
          activation=tf.nn.relu,
          name="deconv1d" + str(i))
      return tf.reshape(thicker,
                        [cur_shape[0], cur_shape[1] * 2, 1, output_filters])

    def deconv2d(cur, i):
      thicker = conv(
          cur,
          output_filters * 4, (1, 1),
          padding="SAME",
          activation=tf.nn.relu,
          name="deconv2d" + str(i))
      return tf.depth_to_space(thicker, 2)

    cur = x
    for i in range(nbr_steps):
      if cur.get_shape()[2] == 1:
        cur = deconv1d(cur, i)
      else:
        cur_dim = shape_list(cur)[2]
        if isinstance(cur_dim, int):
          if cur_dim == 1:
            cur = deconv1d(cur, i)
          else:
            cur = deconv2d(cur, i)
        else:
          cur = tf.cond(
              tf.equal(cur_dim, 1),
              lambda idx=i: deconv1d(cur, idx),
              lambda idx=i: deconv2d(cur, idx))
    return cur


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4. "
                     "Shape: " + str(static_shape))
  # Add support for left padding.
  if kwargs.get("padding") == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    # Set middle two dimensions to None to prevent convolution from complaining
    inputs.set_shape([static_shape[0], None, None, static_shape[3]])
    kwargs["padding"] = "VALID"

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    name = "{}_{}".format(kwargs.get("name", "conv"), name_suffix)
    original_name = kwargs.pop("name", None)
    original_force2d = kwargs.pop("force2d", None)
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result

  return conv2d_kernel(kernel_size, "single")


def conv(inputs, filters, kernel_size, dilation_rate=(1, 1), **kwargs):
  return conv_internal(
      tf.layers.conv2d,
      inputs,
      filters,
      kernel_size,
      dilation_rate=dilation_rate,
      **kwargs)


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
  return tf.squeeze(
      conv(
          tf.expand_dims(inputs, 2),
          filters, (kernel_size, 1),
          dilation_rate=(dilation_rate, 1),
          **kwargs), 2)


def separable_conv(inputs, filters, kernel_size, **kwargs):
  return conv_internal(tf.layers.separable_conv2d, inputs, filters, kernel_size,
                       **kwargs)


def subseparable_conv(inputs, filters, kernel_size, **kwargs):
  """Sub-separable convolution. If separability == 0 it's a separable_conv."""

  def conv_fn(inputs, filters, kernel_size, **kwargs):
    """Sub-separable convolution, splits into separability-many blocks."""
    separability = None
    if "separability" in kwargs:
      separability = kwargs.pop("separability")
    if separability:
      parts = []
      abs_sep = separability if separability > 0 else -1 * separability
      for split_idx, split in enumerate(tf.split(inputs, abs_sep, axis=3)):
        with tf.variable_scope("part_%d" % split_idx):
          if separability > 0:
            parts.append(
                tf.layers.conv2d(split, filters // separability, kernel_size,
                                 **kwargs))
          else:
            parts.append(
                tf.layers.separable_conv2d(split, filters // abs_sep,
                                           kernel_size, **kwargs))
      if separability > 1:
        result = tf.layers.conv2d(tf.concat(parts, axis=3), filters, (1, 1))
      elif abs_sep == 1:  # If we have just one block, return it.
        assert len(parts) == 1
        result = parts[0]
      else:
        result = tf.concat(parts, axis=3)
    else:
      result = tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                          **kwargs)
    if separability is not None:
      kwargs["separability"] = separability
    return result

  return conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs)


def tpu_conv1d(inputs, filters, kernel_size, padding="SAME", name="tpu_conv1d"):
  """Version of conv1d that works on TPU (as of 11/2017).

  Args:
    inputs: a Tensor with shape [batch, length, input_depth].
    filters: an integer.
    kernel_size: an integer.
    padding: a string - "SAME" or "LEFT".
    name: a string.

  Returns:
    a Tensor with shape [batch, length, filters].
  """
  if kernel_size == 1:
    return dense(inputs, filters, name=name, use_bias=True)
  if padding == "SAME":
    assert kernel_size % 2 == 1
    first_offset = -((kernel_size - 1) // 2)
  else:
    assert padding == "LEFT"
    first_offset = -(kernel_size - 1)
  last_offset = first_offset + kernel_size - 1
  results = []
  padded = tf.pad(inputs, [[0, 0], [-first_offset, last_offset], [0, 0]])
  for i in range(kernel_size):
    shifted = tf.slice(padded, [0, i, 0], tf.shape(inputs)) if i else inputs
    shifted.set_shape(inputs.get_shape())
    results.append(
        dense(shifted, filters, use_bias=(i == 0), name=name + "_%d" % i))
  ret = tf.add_n(results)
  ret *= kernel_size**-0.5
  return ret


def layer_norm_vars(filters):
  """Create Variables for layer norm."""
  scale = tf.get_variable(
      "layer_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
  return scale, bias


def layer_norm_compute(x, epsilon, scale, bias):
  """Layer norm raw computation."""
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x * scale + bias


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    scale, bias = layer_norm_vars(filters)
    return layer_norm_compute(x, epsilon, scale, bias)


def group_norm(x, filters=None, num_groups=8, epsilon=1e-5):
  """Group normalization as in https://arxiv.org/abs/1803.08494."""
  x_shape = shape_list(x)
  if filters is None:
    filters = x_shape[-1]
  assert len(x_shape) == 4
  assert filters % num_groups == 0
  # Prepare variables.
  scale = tf.get_variable(
      "group_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "group_norm_bias", [filters], initializer=tf.zeros_initializer())
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  # Reshape and compute group norm.
  x = tf.reshape(x, x_shape[:-1] + [num_groups, filters // num_groups])
  # Calculate mean and variance on heights, width, channels (not groups).
  mean, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return tf.reshape(norm_x, x_shape) * scale + bias


def noam_norm(x, epsilon=1.0, name=None):
  """One version of layer normalization."""
  with tf.name_scope(name, default_name="noam_norm", values=[x]):
    shape = x.get_shape()
    ndims = len(shape)
    return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
        tf.to_float(shape[-1])))


def l2_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalization with l2 norm."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(name, default_name="l2_norm", values=[x], reuse=reuse):
    scale = tf.get_variable(
        "l2_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "l2_norm_bias", [filters], initializer=tf.zeros_initializer())
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    l2norm = tf.reduce_sum(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(l2norm + epsilon)
    return norm_x * scale + bias


def apply_spectral_norm(x):
  """Normalizes x using the spectral norm.

  The implementation follows Algorithm 1 of
  https://arxiv.org/abs/1802.05957. If x is not a 2-D Tensor, then it is
  reshaped such that the number of channels (last-dimension) is the same.

  Args:
    x: Tensor with the last dimension equal to the number of filters.

  Returns:
    x: Tensor with the same shape as x normalized by the spectral norm.
    assign_op: Op to be run after every step to update the vector "u".
  """
  weights_shape = shape_list(x)
  other, num_filters = tf.reduce_prod(weights_shape[:-1]), weights_shape[-1]

  # Reshape into a 2-D matrix with outer size num_filters.
  weights_2d = tf.reshape(x, (other, num_filters))

  # v = Wu / ||W u||
  with tf.variable_scope("u", reuse=tf.AUTO_REUSE):
    u = tf.get_variable(
        "u", [num_filters, 1],
        initializer=tf.truncated_normal_initializer(),
        trainable=False)
  v = tf.nn.l2_normalize(tf.matmul(weights_2d, u))

  # u_new = vW / ||v W||
  u_new = tf.nn.l2_normalize(tf.matmul(tf.transpose(v), weights_2d))

  # s = v*W*u
  spectral_norm = tf.squeeze(
      tf.matmul(tf.transpose(v), tf.matmul(weights_2d, tf.transpose(u_new))))

  # set u equal to u_new in the next iteration.
  assign_op = tf.assign(u, tf.transpose(u_new))
  return tf.divide(x, spectral_norm), assign_op


def apply_norm(x, norm_type, depth, epsilon):
  """Apply Normalization."""
  if norm_type == "layer":
    return layer_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "group":
    return group_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "batch":
    return tf.layers.batch_normalization(x, epsilon=epsilon)
  if norm_type == "noam":
    return noam_norm(x, epsilon)
  if norm_type == "l2":
    return l2_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "none":
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'lr', 'none'.")


def zero_add(previous_value, x, name=None, reuse=None):
  """Resnet connection with zero initialization.

  Another type of resnet connection which returns previous_value + gamma * x.
  gamma is a trainable scalar and initialized with zero. It is useful when a
  module is plugged into a trained model and we want to make sure it matches the
  original model's performance.

  Args:
    previous_value:  A tensor.
    x: A tensor.
    name: name of variable scope; defaults to zero_add.
    reuse: reuse scope.

  Returns:
    previous_value + gamma * x.
  """
  with tf.variable_scope(name, default_name="zero_add", reuse=reuse):
    gamma = tf.get_variable("gamma", (), initializer=tf.zeros_initializer())
    return previous_value + gamma * x


def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         default_name,
                         name=None,
                         dropout_broadcast_dims=None):
  """Apply a sequence of functions to the input or output of a layer.

  The sequence is specified as a string which may contain the following
  characters:
    a: add previous_value
    n: apply normalization
    d: apply dropout
    z: zero add

  For example, if sequence=="dna", then the output is
    previous_value + normalize(dropout(x))

  Args:
    previous_value: A Tensor, to be added as a residual connection ('a')
    x: A Tensor to be transformed.
    sequence: a string.
    dropout_rate: a float
    norm_type: a string (see apply_norm())
    depth: an integer (size of last dimension of x).
    epsilon: a float (parameter for normalization)
    default_name: a string
    name: a string
    dropout_broadcast_dims:  an optional list of integers less than 3
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.

  Returns:
    a Tensor
  """
  with tf.variable_scope(name, default_name=default_name):
    if sequence == "none":
      return x
    for c in sequence:
      if c == "a":
        x += previous_value
      elif c == "z":
        x = zero_add(previous_value, x)
      elif c == "n":
        x = apply_norm(x, norm_type, depth, epsilon)
      else:
        assert c == "d", ("Unknown sequence step %s" % c)
        x = dropout_with_broadcast_dims(
            x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    return x


def layer_preprocess(layer_input, hparams):
  """Apply layer preprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_preprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    hparams: a hyperparameters object.

  Returns:
    a Tensor
  """
  assert "a" not in hparams.layer_preprocess_sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  assert "z" not in hparams.layer_preprocess_sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
      None,
      layer_input,
      sequence=hparams.layer_preprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      norm_type=hparams.norm_type,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_prepostprocess")


def layer_postprocess(layer_input, layer_output, hparams):
  """Apply layer postprocessing.

  See layer_prepostprocess() for details.

  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:

    layer_postprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon

  Args:
    layer_input: a Tensor
    layer_output: a Tensor
    hparams: a hyperparameters object.

  Returns:
    a Tensor
  """
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence=hparams.layer_postprocess_sequence,
      dropout_rate=hparams.layer_prepostprocess_dropout,
      norm_type=hparams.norm_type,
      depth=None,
      epsilon=hparams.norm_epsilon,
      dropout_broadcast_dims=comma_separated_string_to_integer_list(
          getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
      default_name="layer_postprocess")


def conv_block_internal(conv_fn,
                        inputs,
                        filters,
                        dilation_rates_and_kernel_sizes,
                        first_relu=True,
                        use_elu=False,
                        separabilities=None,
                        **kwargs):
  """A block of convolutions.

  Args:
    conv_fn: convolution function, e.g. conv or separable_conv.
    inputs: a Tensor
    filters: an Integer
    dilation_rates_and_kernel_sizes: a list of tuples (dilation, (k_w, k_h))
    first_relu: whether to do a relu at start (defaults to True)
    use_elu: whether to use ELUs instead of ReLUs (defaults to False)
    separabilities: list of separability factors (per-layer).
    **kwargs: additional arguments (e.g., pooling)

  Returns:
     a Tensor.
  """

  name = kwargs.pop("name") if "name" in kwargs else None
  mask = kwargs.pop("mask") if "mask" in kwargs else None

  # Usage for normalize_fn kwarg:
  # if not specified, use layer norm
  # if given normalize_fn=None, don't use any normalization
  # if given normalize_fn=norm, use the specified norm function

  use_layer_norm = "normalizer_fn" not in kwargs
  norm = kwargs.pop("normalizer_fn", None)
  use_normalizer_fn = use_layer_norm or norm

  if use_layer_norm:
    norm = lambda x, name: layer_norm(x, filters, name=name)

  with tf.variable_scope(name, "conv_block", [inputs]):
    cur, counter = inputs, -1
    for dilation_rate, kernel_size in dilation_rates_and_kernel_sizes:
      counter += 1
      if first_relu or counter > 0:
        cur = tf.nn.elu(cur) if use_elu else tf.nn.relu(cur)
      if mask is not None:
        cur *= mask
      if separabilities:
        cur = conv_fn(
            cur,
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            name="conv_block_%d" % counter,
            use_bias=norm is None,
            separability=separabilities[counter],
            **kwargs)
      else:
        cur = conv_fn(
            cur,
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            name="conv_block_%d" % counter,
            use_bias=norm is None,
            **kwargs)
      if use_normalizer_fn:
        cur = norm(cur, name="conv_block_norm_%d" % counter)
    return cur


def conv_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard 2d convolutions."""
  return conv_block_internal(conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def conv1d_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard 1d convolutions."""
  return conv_block_internal(conv1d, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def separable_conv_block(inputs, filters, dilation_rates_and_kernel_sizes,
                         **kwargs):
  """A block of separable convolutions."""
  return conv_block_internal(separable_conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def subseparable_conv_block(inputs, filters, dilation_rates_and_kernel_sizes,
                            **kwargs):
  """A block of separable convolutions."""
  return conv_block_internal(subseparable_conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def pool(inputs, window_size, pooling_type, padding, strides=(1, 1)):
  """Pooling (supports "LEFT")."""
  with tf.name_scope("pool", values=[inputs]):
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
      raise ValueError("Inputs to conv must have statically known rank 4.")
    # Add support for left padding.
    if padding == "LEFT":
      assert window_size[0] % 2 == 1 and window_size[1] % 2 == 1
      if len(static_shape) == 3:
        width_padding = 2 * (window_size[1] // 2)
        padding_ = [[0, 0], [width_padding, 0], [0, 0]]
      else:
        height_padding = 2 * (window_size[0] // 2)
        cond_padding = tf.cond(
            tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
            lambda: tf.constant(2 * (window_size[1] // 2)))
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding_ = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
      inputs = tf.pad(inputs, padding_)
      inputs.set_shape([static_shape[0], None, None, static_shape[3]])
      padding = "VALID"

  return tf.nn.pool(inputs, window_size, pooling_type, padding, strides=strides)


def conv_block_downsample(x,
                          kernel,
                          strides,
                          padding,
                          separability=0,
                          name=None,
                          reuse=None):
  """Implements a downwards-striding conv block, like Xception exit flow."""
  with tf.variable_scope(
      name, default_name="conv_block_downsample", values=[x], reuse=reuse):
    hidden_size = int(x.get_shape()[-1])
    res = conv_block(
        x,
        int(1.25 * hidden_size), [((1, 1), kernel)],
        padding=padding,
        strides=strides,
        name="res_conv")

    x = subseparable_conv_block(
        x,
        hidden_size, [((1, 1), kernel)],
        padding=padding,
        separability=separability,
        name="conv0")
    x = subseparable_conv_block(
        x,
        int(1.25 * hidden_size), [((1, 1), kernel)],
        padding=padding,
        separability=separability,
        name="conv1")
    x = pool(x, kernel, "MAX", padding, strides=strides)

    x += res

    x = subseparable_conv_block(
        x,
        2 * hidden_size, [((1, 1), kernel)],
        first_relu=False,
        padding=padding,
        separability=separability,
        name="conv2")
    x = subseparable_conv_block(
        x,
        int(2.5 * hidden_size), [((1, 1), kernel)],
        padding=padding,
        separability=separability,
        name="conv3")
    return x


def get_timing_signal(length,
                      min_timescale=1,
                      max_timescale=1e4,
                      num_timescales=16):
  """Create Tensor of sinusoids of different frequencies.

  Args:
    length: Length of the Tensor to create, i.e. Number of steps.
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int

  Returns:
    Tensor of shape (length, 2*num_timescales)
  """
  positions = tf.to_float(tf.range(length))
  log_timescale_increment = (
      math.log(max_timescale / min_timescale) / (num_timescales - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
  return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)


def add_timing_signal(x, min_timescale=1, max_timescale=1e4, num_timescales=16):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  This allows attention to learn to use absolute and relative positions.
  The timing signal should be added to some precursor of both the source
  and the target of the attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the depth dimension, padded with zeros to be the same depth as the input,
  and added into input.

  Args:
    x: a Tensor with shape [?, length, ?, depth]
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int <= depth/2

  Returns:
    a Tensor the same shape as x.
  """
  length = shape_list(x)[1]
  depth = shape_list(x)[3]
  signal = get_timing_signal(length, min_timescale, max_timescale,
                             num_timescales)
  padded_signal = tf.pad(signal, [[0, 0], [0, depth - 2 * num_timescales]])
  return x + tf.reshape(padded_signal, [1, length, 1, depth])


def mask_from_embedding(emb):
  """Input embeddings -> padding mask.

  We have hacked symbol_modality to return all-zero embeddings for padding.
  Returns a mask with 0.0 in the padding positions and 1.0 elsewhere.

  Args:
    emb: a Tensor with shape [batch, width, height, depth].
  Returns:
    a 0.0/1.0 Tensor with shape [batch, width, height, 1].
  """
  return weights_nonzero(tf.reduce_sum(tf.abs(emb), axis=3, keepdims=True))


def length_from_embedding(emb):
  """Compute the length of each sequence in the batch.

  Args:
    emb: a sequence embedding Tensor with shape [batch, max_time, 1, depth].
  Returns:
    a Tensor with shape [batch].
  """
  return tf.cast(tf.reduce_sum(mask_from_embedding(emb), [1, 2, 3]), tf.int32)


def mask_leq(target_length, source_length):
  """A mask with 1.0 wherever source_pos <= target_pos and 0.0 elsewhere.

  Args:
    target_length: an integer
    source_length: an integer
  Returns:
    a Tensor with shape [1, target_length, source_length]
  """
  return ones_matrix_band_part(
      target_length,
      source_length,
      -1,
      0,
      out_shape=[1, target_length, source_length])


def relu_density_logit(x, reduce_dims):
  """logit(density(x)).

  Useful for histograms.

  Args:
    x: a Tensor, typically the output of tf.relu
    reduce_dims: a list of dimensions

  Returns:
    a Tensor
  """
  frac = tf.reduce_mean(tf.to_float(x > 0.0), reduce_dims)
  scaled = tf.log(frac + math.exp(-10)) - tf.log((1.0 - frac) + math.exp(-10))
  return scaled


def maybe_zero_out_padding(inputs, kernel_size, nonpadding_mask):
  """If necessary, zero out inputs to a conv for padding positions.

  Args:
    inputs: a Tensor with shape [batch, length, ...]
    kernel_size: an integer or pair of integers
    nonpadding_mask: a Tensor with shape [batch, length]

  Returns:
    Tensor of the same shape as inputs.
  """
  if (kernel_size != 1 and kernel_size != (1, 1) and
      nonpadding_mask is not None):
    while nonpadding_mask.get_shape().ndims < inputs.get_shape().ndims:
      nonpadding_mask = tf.expand_dims(nonpadding_mask, -1)
    return inputs * nonpadding_mask

  return inputs


def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  layer_name = "%s_{}" % name if name else "{}"
  h = dense(
      inputs,
      filter_size,
      use_bias=True,
      activation=tf.nn.relu,
      name=layer_name.format("conv1"))

  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = dense(
      h,
      output_size,
      activation=output_activation,
      use_bias=True,
      name=layer_name.format("conv2"))
  return o


def dense_dropconnect(inputs,
                      output_size,
                      dropconnect_dropout=0.0,
                      name="dense_dropconnect",
                      **kwargs):
  """Dense layer with dropconnect."""

  if dropconnect_dropout != 0.0:
    tf.logging.info("Applying dropconnect as the kernel regularization.")
    kwargs["kernel_regularizer"] = partial(
        tf.nn.dropout, keep_prob=1.0 - dropconnect_dropout)

  return dense(inputs, output_size, use_bias=True, name=name, **kwargs)


def conv_relu_conv(inputs,
                   filter_size,
                   output_size,
                   first_kernel_size=3,
                   second_kernel_size=3,
                   padding="SAME",
                   nonpadding_mask=None,
                   dropout=0.0,
                   name=None,
                   cache=None,
                   decode_loop_step=None):
  """Hidden layer with RELU activation followed by linear projection.

  Args:
    inputs: A tensor.
    filter_size: An integer.
    output_size: An integer.
    first_kernel_size: An integer.
    second_kernel_size: An integer.
    padding: A string.
    nonpadding_mask: A tensor.
    dropout: A float.
    name: A string.
    cache: A dict, containing Tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU. If it is not None, the function
        will do inplace update for the cache instead of concatenating the
        current result to the cache.

  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, "conv_relu_conv", [inputs]):
    inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)

    if cache:
      if decode_loop_step is None:
        inputs = cache["f"] = tf.concat([cache["f"], inputs], axis=1)
      else:
        # Inplace update is required for inference on TPU.
        # Inplace_ops only supports inplace_update on the first dimension.
        # The performance of current implementation is better than updating
        # the tensor by adding the result of matmul(one_hot,
        # update_in_current_step)
        tmp_f = tf.transpose(cache["f"], perm=[1, 0, 2])
        tmp_f = inplace_ops.alias_inplace_update(
            tmp_f,
            decode_loop_step * tf.shape(inputs)[1],
            tf.transpose(inputs, perm=[1, 0, 2]))
        inputs = cache["f"] = tf.transpose(tmp_f, perm=[1, 0, 2])
      inputs = cache["f"] = inputs[:, -first_kernel_size:, :]

    h = tpu_conv1d(
        inputs, filter_size, first_kernel_size, padding=padding, name="conv1")

    if cache:
      h = h[:, -1:, :]

    h = tf.nn.relu(h)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
    return tpu_conv1d(
        h, output_size, second_kernel_size, padding=padding, name="conv2")


def sepconv_relu_sepconv(inputs,
                         filter_size,
                         output_size,
                         first_kernel_size=(1, 1),
                         second_kernel_size=(1, 1),
                         padding="LEFT",
                         nonpadding_mask=None,
                         dropout=0.0,
                         name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  with tf.variable_scope(name, "sepconv_relu_sepconv", [inputs]):
    inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)
    if inputs.get_shape().ndims == 3:
      is_3d = True
      inputs = tf.expand_dims(inputs, 2)
    else:
      is_3d = False
    h = separable_conv(
        inputs,
        filter_size,
        first_kernel_size,
        activation=tf.nn.relu,
        padding=padding,
        name="conv1")
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
    ret = separable_conv(
        h, output_size, second_kernel_size, padding=padding, name="conv2")
    if is_3d:
      ret = tf.squeeze(ret, 2)
    return ret


# DEPRECATED - use dense_relu_dense, conv_relu_conv, sepconv_relu_sepconv
def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     second_kernel_size=(1, 1),
                     dropout=0.0,
                     **kwargs):
  """Hidden layer with RELU activation followed by linear projection."""
  name = kwargs.pop("name") if "name" in kwargs else None
  with tf.variable_scope(name, "conv_hidden_relu", [inputs]):
    if inputs.get_shape().ndims == 3:
      is_3d = True
      inputs = tf.expand_dims(inputs, 2)
    else:
      is_3d = False
    conv_f1 = conv if kernel_size == (1, 1) else separable_conv
    h = conv_f1(
        inputs,
        hidden_size,
        kernel_size,
        activation=tf.nn.relu,
        name="conv1",
        **kwargs)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    conv_f2 = conv if second_kernel_size == (1, 1) else separable_conv
    ret = conv_f2(h, output_size, second_kernel_size, name="conv2", **kwargs)
    if is_3d:
      ret = tf.squeeze(ret, 2)
    return ret


def conv_gru(x,
             kernel_size,
             filters,
             padding="SAME",
             dilation_rate=(1, 1),
             name=None,
             reuse=None):
  """Convolutional GRU in 1 dimension."""

  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start, padding):
    return conv(
        args,
        filters,
        kernel_size,
        padding=padding,
        dilation_rate=dilation_rate,
        bias_initializer=tf.constant_initializer(bias_start),
        name=name)

  # Here comes the GRU gate.
  with tf.variable_scope(
      name, default_name="conv_gru", values=[x], reuse=reuse):
    reset = saturating_sigmoid(do_conv(x, "reset", 1.0, padding))
    gate = saturating_sigmoid(do_conv(x, "gate", 1.0, padding))
    candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0, padding))
    return gate * x + (1 - gate) * candidate


def gru_feedfwd(a_t, h_prev, filters, name=None):
  """position-wise Feed-fwd GRU gates following the MPNN.

  Args:
    a_t: Tensor of shape [batch, length, depth] of current input
    h_prev: Tensor of shape [batch, length, depth] of prev input
    filters: an integer specifying number of dimensions of the filters
    name: A string
  Returns:
    h_t: [batch, length, filters] hidden state
  """

  with tf.variable_scope(name, default_name="GRU", values=[a_t, h_prev]):
    # we use right matrix multiplication to handle batches
    # W_z and W_r have shape 2d, d. U_z U_r have shape d,d
    z_t = (
        tf.sigmoid(
            tpu_conv1d(a_t, filters, 1, padding="SAME", name="W_z") +
            tpu_conv1d(h_prev, filters, 1, padding="SAME", name="U_z")))
    r_t = (
        tf.sigmoid(
            tpu_conv1d(a_t, filters, 1, padding="SAME", name="W_r") +
            tpu_conv1d(h_prev, filters, 1, padding="SAME", name="U_r")))
    h_tilde = (
        tf.tanh(
            tpu_conv1d(a_t, filters, 1, padding="SAME", name="W") +
            tpu_conv1d(r_t * h_prev, filters, 1, padding="SAME", name="U")))
    h_t = (1. - z_t) * h_prev + z_t * h_tilde

  return h_t


def conv_lstm(x,
              kernel_size,
              filters,
              padding="SAME",
              dilation_rate=(1, 1),
              name=None,
              reuse=None):
  """Convolutional LSTM in 1 dimension."""
  with tf.variable_scope(
      name, default_name="conv_lstm", values=[x], reuse=reuse):
    gates = conv(
        x,
        4 * filters,
        kernel_size,
        padding=padding,
        dilation_rate=dilation_rate)
    g = tf.split(layer_norm(gates, 4 * filters), 4, axis=3)
    new_cell = tf.sigmoid(g[0]) * x + tf.sigmoid(g[1]) * tf.tanh(g[3])
    return tf.sigmoid(g[2]) * tf.tanh(new_cell)


def diagonal_conv_gru(x,
                      kernel_size,
                      filters,
                      dropout=0.0,
                      name=None,
                      reuse=None):
  """Diagonal Convolutional GRU as in https://arxiv.org/abs/1702.08727."""

  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start):
    return conv(
        args,
        filters,
        kernel_size,
        padding="SAME",
        bias_initializer=tf.constant_initializer(bias_start),
        name=name)

  # Here comes the GRU gate.
  with tf.variable_scope(
      name, default_name="diagonal_conv_gru", values=[x], reuse=reuse):
    reset, reset_cost = hard_sigmoid(do_conv(x, "reset", 0.5))
    gate, gate_cost = hard_sigmoid(do_conv(x, "gate", 0.7))
    candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0))

    if dropout > 0.0:
      candidate = tf.nn.dropout(candidate, 1.0 - dropout)

    # Diagonal shift.
    shift_filters = filters // 3
    base_filter = ([[0, 1, 0]] * (filters - 2 * shift_filters) +
                   [[1, 0, 0]] * shift_filters + [[0, 0, 1]] * shift_filters)
    shift_filter = tf.constant(np.transpose(base_filter), dtype=tf.float32)
    shift_filter = tf.expand_dims(tf.expand_dims(shift_filter, 0), 3)
    x_shifted = tf.nn.depthwise_conv2d(
        x, shift_filter, [1, 1, 1, 1], padding="SAME")

    # Return the gated result and cost.
    total_cost_avg = 0.5 * (reset_cost + gate_cost)
    return gate * x_shifted + (1 - gate) * candidate, total_cost_avg


def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
  """Pad tensors x and y on axis 1 so that they have the same length."""
  if axis not in [1, 2]:
    raise ValueError("Only axis=1 and axis=2 supported for now.")
  with tf.name_scope("pad_to_same_length", values=[x, y]):
    x_length = shape_list(x)[axis]
    y_length = shape_list(y)[axis]
    if (isinstance(x_length, int) and isinstance(y_length, int) and
        x_length == y_length and final_length_divisible_by == 1):
      return x, y
    max_length = tf.maximum(x_length, y_length)
    if final_length_divisible_by > 1:
      # Find the nearest larger-or-equal integer divisible by given number.
      max_length += final_length_divisible_by - 1
      max_length //= final_length_divisible_by
      max_length *= final_length_divisible_by
    length_diff1 = max_length - x_length
    length_diff2 = max_length - y_length

    def padding_list(length_diff, arg):
      if axis == 1:
        return [[[0, 0], [0, length_diff]],
                tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
      return [[[0, 0], [0, 0], [0, length_diff]],
              tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

    paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
    paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
    res_x = tf.pad(x, paddings1)
    res_y = tf.pad(y, paddings2)
    # Static shapes are the same except for axis=1.
    x_shape = x.shape.as_list()
    x_shape[axis] = None
    res_x.set_shape(x_shape)
    y_shape = y.shape.as_list()
    y_shape[axis] = None
    res_y.set_shape(y_shape)
    return res_x, res_y


def pad_with_zeros(logits, labels):
  """Pad labels on the length dimension to match logits length."""
  with tf.name_scope("pad_with_zeros", values=[logits, labels]):
    logits, labels = pad_to_same_length(logits, labels)
    if len(labels.shape) == 3:  # 2-d labels.
      logits, labels = pad_to_same_length(logits, labels, axis=2)
    return logits, labels


def weights_nonzero(labels):
  """Assign weight 1.0 to all labels except for padding (id=0)."""
  return tf.to_float(tf.not_equal(labels, 0))


def weights_prepend_inputs_to_targets(labels):
  """Assign weight 1.0 to only the "targets" portion of the labels.

  Weight 1.0 is assigned to all nonzero labels past the first zero.
  See prepend_mode in common_hparams.py

  Args:
    labels: A Tensor of int32s.

  Returns:
    A Tensor of floats.
  """
  past_first_zero = tf.cumsum(tf.to_float(tf.equal(labels, 0)), axis=1)
  nonzero = tf.to_float(labels)
  return tf.to_float(tf.not_equal(past_first_zero * nonzero, 0))


def weights_multi_problem(labels, taskid=-1):
  """Assign weight 1.0 to only the "targets" portion of the labels.

  Weight 1.0 is assigned to all labels past the taskid.

  Args:
    labels: A Tensor of int32s.
    taskid: an int32 representing the task id for a problem.

  Returns:
    A Tensor of floats.

  Raises:
    ValueError: The Task ID must be valid.
  """
  if taskid < 0:
    raise ValueError("Task ID must be non-negative.")

  past_taskid = tf.cumsum(tf.to_float(tf.equal(labels, taskid)), axis=1)
  # Additionally zero out the task id location
  past_taskid *= tf.to_float(tf.not_equal(labels, taskid))
  non_taskid = tf.to_float(labels)
  return tf.to_float(tf.not_equal(past_taskid * non_taskid, 0))


def weights_multi_problem_all(labels, taskid=-1):
  """Assign weight 1.0 to only examples from the given task."""
  weights = tf.to_float(tf.not_equal(labels, 0))
  if taskid < 0:
    raise ValueError("Task ID must be non-negative.")

  past_taskid = tf.cumsum(tf.to_float(tf.equal(labels, taskid)), axis=1)
  # Additionally zero out the task id location
  past_taskid *= tf.to_float(tf.not_equal(labels, taskid))
  non_taskid = tf.to_float(labels)
  example_mask = tf.to_float(tf.not_equal(past_taskid * non_taskid, 0))
  example_mask = tf.reduce_sum(example_mask, axis=1)
  example_mask = tf.to_float(
      tf.greater(example_mask, tf.zeros_like(example_mask)))

  return weights * tf.expand_dims(example_mask, axis=-1)


def weights_multi_problem_input(labels, taskid=-1):
  """Assign weight 1.0 to only the inputs for the given task."""
  weights_all_tokens = weights_multi_problem_all(labels, taskid)
  weights_target = weights_multi_problem(labels, taskid)
  return weights_all_tokens - weights_target


def weights_all(labels):
  """Assign weight 1.0 to all labels."""
  return tf.ones_like(labels, dtype=tf.float32)


def weights_concatenated(labels):
  """Assign weight 1.0 to the "target" part of the concatenated labels.

  The labels look like:
    source English I love you . ID1 target French Je t'aime . ID1 source
      English the cat ID1 target French le chat ID1 source English ...

  We want to assign weight 1.0 to all words in the target text (including the
  ID1 end symbol), but not to the source text or the boilerplate.  In the
  above example, the target words that get positive weight are:
    Je t'aime . ID1 le chat ID1

  Args:
    labels: a Tensor
  Returns:
    a Tensor
  """
  eos_mask = tf.to_int32(tf.equal(labels, 1))
  sentence_num = tf.cumsum(eos_mask, axis=1, exclusive=True)
  in_target = tf.equal(tf.mod(sentence_num, 2), 1)
  # first two tokens of each sentence are boilerplate.
  sentence_num_plus_one = sentence_num + 1
  shifted = tf.pad(sentence_num_plus_one,
                   [[0, 0], [2, 0], [0, 0], [0, 0]])[:, :-2, :, :]
  nonboilerplate = tf.equal(sentence_num_plus_one, shifted)
  ret = tf.to_float(tf.logical_and(nonboilerplate, in_target))
  return ret


def padded_cross_entropy(logits,
                         labels,
                         label_smoothing,
                         weights_fn=weights_nonzero,
                         reduce_sum=True,
                         cutoff=0.0,
                         gaussian=False):
  """Compute cross-entropy assuming 0s are padding.

  Computes a loss numerator (the sum of losses), and loss denominator
  (the number of non-padding tokens).

  Args:
    logits: a `Tensor` with shape `[batch, timesteps, vocab_size]`.
      optionally a FactoredTensor.
    labels: an integer `Tensor` with shape `[batch, timesteps]`.
    label_smoothing: a floating point `Scalar`.
    weights_fn: A function from labels to weights.
    reduce_sum: a Boolean, whether to sum at the end or not.
    cutoff: a float, at which point to have no loss.
    gaussian: If true, use a Gaussian distribution for label smoothing

  Returns:
    loss_numerator: a `Scalar`.  Sum of losses.
    loss_denominator: a `Scalar.  The number of non-padding target tokens.

  Raises:
    ValueError: in case of unsupported argument types.
  """
  if isinstance(logits, FactoredTensor):
    if gaussian:
      raise ValueError("Factored padded cross entropy with Gaussian smoothing "
                       "is not implemented yet.")
    return padded_cross_entropy_factored(
        logits,
        labels,
        label_smoothing,
        weights_fn=weights_fn,
        reduce_sum=reduce_sum)
  confidence = 1.0 - label_smoothing
  logits_shape = shape_list(logits)
  vocab_size = logits_shape[-1]
  with tf.name_scope("padded_cross_entropy", values=[logits, labels]):
    if len(logits_shape) == 2:
      # Deal with the case where we did not insert extra dimensions due to
      # TPU issues.  No pad-to-same-length happens in this case.
      # TODO(noam): remove this logic once TPU can handle extra dimensions.
      labels = tf.reshape(labels, [-1])
    else:
      logits, labels = pad_with_zeros(logits, labels)
    logits = tf.reshape(
        logits,
        shape_list(labels) + [vocab_size],
        name="padded_cross_entropy_size_check")
    logits = tf.cast(logits, tf.float32)
    xent = smoothing_cross_entropy(
        logits, labels, vocab_size, confidence, gaussian=gaussian)
    weights = weights_fn(labels)
    if cutoff > 0.0:
      xent = tf.nn.relu(xent - cutoff)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def _weights_one_third(labels):
  """Returns Tensor of shape [batch, height, width]. Each element is 1/3."""
  return tf.ones(tf.shape(labels)[:-1]) / 3.


def dml_loss(pred, labels, weights_fn=_weights_one_third, reduce_sum=True):
  """Discretized mixture of logistics loss.

  Args:
    pred: A [batch, height, width, num_mixtures*10] tensor of floats
      comprising one unconstrained mixture probability, three means
      (one per channel), three standard deviations (one per channel),
      and three coefficients which linearly parameterize dependence across
      channels.
    labels: A [batch, height, width, channels] tensor of 8-bit pixel
      intensities. The computation assumes channels is 3.
    weights_fn: A function of labels, returning a Tensor of shape
      [batch, height, width] which weights each loss term. Default is to scale
      each loss term by 1/3 so that they capture the average across channels.
    reduce_sum: A boolean, to return scalar loss instead of per position.

  Returns:
    Tuple of loss tensors for numerator and denominator, each a scalar if
    reduce_sum else of shape [batch, height, width]. The sum of their divisions
    is the number of nats for each pixel in labels.
  """
  real_labels = convert_rgb_to_symmetric_real(labels)
  dml_loss_value = discretized_mix_logistic_loss(pred=pred, labels=real_labels)
  weights = weights_fn(labels)
  loss_num = weights * dml_loss_value
  loss_den = weights_nonzero(weights)
  if reduce_sum:
    loss_num = tf.reduce_sum(loss_num)
    loss_den = tf.reduce_sum(loss_den)
  return loss_num, loss_den


def split_to_discretized_mix_logistic_params(inputs):
  """Splits input tensor into parameters of discretized mixture logistic.

  Args:
    inputs: A [batch, height, width, num_mixtures*10] tensor of floats
      comprising one unconstrained mixture probability, three means
      (one per channel), three standard deviations (one per channel),
      and three coefficients which linearly parameterize dependence across
      channels.

  Returns:
    Tuple of unconstrained mixture probabilities, locations, scales, and
    coefficient parameters of the distribution. The mixture probability has
    shape [batch, height, width, num_mixtures]. Other parameters have shape
    [batch, height, width, num_mixtures, 3].
  """
  batch, height, width, output_dim = shape_list(inputs)
  num_mixtures = output_dim // 10
  logits, locs, log_scales, coeffs = tf.split(
      inputs,
      num_or_size_splits=[
          num_mixtures, num_mixtures * 3, num_mixtures * 3, num_mixtures * 3
      ],
      axis=-1)
  split_shape = [batch, height, width, num_mixtures, 3]
  locs = tf.reshape(locs, split_shape)
  log_scales = tf.reshape(log_scales, split_shape)
  log_scales = tf.maximum(log_scales, -7.)
  coeffs = tf.reshape(coeffs, split_shape)
  coeffs = tf.tanh(coeffs)
  return logits, locs, log_scales, coeffs


def discretized_mix_logistic_loss(pred, labels):
  """Computes negative log probability for the discretized mixture of logistics.

  The distribution of a whole pixel is a mixture of 3-dimensional discretized
  logistic distributions. The 3-D discretized logistic factorizes as 3 1-D
  discretized logistic distributions, one for each channel. It defines

  ```none
  P(X = x)
  = sum_{k=1}^K probs[k] * P(X = x | locs[k], scales[k])
  = sum_{k=1}^K probs[k] * [
      prod_{c=1}^3 DiscretizedLogistic(X[c] = x[c] | means[k][c], scales[k]) ]
  ```

  The means tensor is a linear combination of location parameters and previous
  channels. The discretized logistic distribution assigns probability mass to an
  event P(X=x) via logistic CDFs: P(X <= x + 0.5) - P(X > x - 0.5) for 1 < x <
  254; P(X <= 0.5) for x = 0; and 1 - P(X > 245.5) for x = 255. Instead of
  8-bit inputs, this implementation assumes the events are rescaled to [-1, 1].

  Args:
    pred: A [batch, height, width, num_mixtures*10] tensor of floats
      comprising one unconstrained mixture probability, three means
      (one per channel), three standard deviations (one per channel),
      and three coefficients which linearly parameterize dependence across
      channels.
    labels: A [batch, height, width, channels] tensor of true pixel intensities
      rescaled to [-1, 1]. The computation assumes channels is 3.

  Returns:
    A [batch, height, width] tensor of the negative log conditional probability
    of each pixel given all previous pixels.
  """

  logits, locs, log_scales, coeffs = split_to_discretized_mix_logistic_params(
      pred)

  # Tile labels to broadcast compute across the mixture dimension.
  batch, height, width, num_mixtures = shape_list(logits)
  labels = tf.tile(
      tf.reshape(labels, [batch, height, width, 1, 3]),
      [1, 1, 1, num_mixtures, 1])

  # p(x) = sigmoid((x - means_i + 1/255.)/scale_i) -
  #        sigmoid((x - means_i - 1/255.)/scale_i)
  # for each channel i. The means are linearly parameterized.
  means_0 = locs[..., 0]
  means_1 = locs[..., 1] + coeffs[..., 0] * labels[..., 0]
  means_2 = (
      locs[..., 2] + coeffs[..., 1] * labels[..., 0] +
      coeffs[..., 2] * labels[..., 1])
  means = tf.stack([means_0, means_1, means_2], axis=-1)
  centered_labels = labels - means
  inv_stdv = tf.exp(-log_scales)
  plus_in = inv_stdv * (centered_labels + 1. / 255.)
  min_in = inv_stdv * (centered_labels - 1. / 255.)
  cdf_plus = tf.nn.sigmoid(plus_in)
  cdf_min = tf.nn.sigmoid(min_in)

  # Compute log probability for edge case of 0 (before scaling), 255 (before
  # scaling), and all other cases respectively.
  log_prob_0 = plus_in - tf.nn.softplus(plus_in)
  log_prob_255 = -tf.nn.softplus(min_in)
  prob_event = tf.maximum(cdf_plus - cdf_min, 1e-12)
  log_prob_event = tf.log(prob_event)

  # Robustly select log-prob based on numerical edge-cases: (a) [-1, -1+eps);
  # (b) (1-eps, 1]; (c) NaNs during `tf.gradients` of `tf.select`, which may
  # cause `tf.log(0.)`; (d) p(x) < 1e-5.
  mid_in = inv_stdv * centered_labels
  log_prob_event_approx = (
      mid_in - log_scales - 2. * tf.nn.softplus(mid_in) - np.log(127.5))
  log_probs = tf.where(
      labels < -0.999, log_prob_0,
      tf.where(
          labels > 0.999, log_prob_255,
          tf.where(prob_event > 1e-5, log_prob_event, log_prob_event_approx)))

  # Sum over channels and compute log-probability of each mixture.
  log_probs = tf.reduce_sum(log_probs, -1) + tf.nn.log_softmax(logits, axis=-1)
  output = -tf.reduce_logsumexp(log_probs, axis=-1)
  return output


def sample_from_discretized_mix_logistic(pred, seed=None):
  """Sampling from a discretized mixture of logistics.

  Args:
    pred: A [batch, height, width, num_mixtures*10] tensor of floats
      comprising one unconstrained mixture probability, three means
      (one per channel), three standard deviations (one per channel),
      and three coefficients which linearly parameterize dependence across
      channels.
    seed: Random seed.

  Returns:
    A tensor of shape [batch, height, width, 3] with real intensities scaled
    between -1 and 1.
  """

  logits, locs, log_scales, coeffs = split_to_discretized_mix_logistic_params(
      pred)

  # Sample mixture indicator given logits using the gumbel max trick.
  num_mixtures = shape_list(logits)[-1]
  gumbel_noise = -tf.log(-tf.log(
      tf.random_uniform(
          tf.shape(logits), minval=1e-5, maxval=1. - 1e-5, seed=seed)))
  sel = tf.one_hot(
      tf.argmax(logits + gumbel_noise, -1),
      depth=num_mixtures,
      dtype=tf.float32)

  # Select mixture component's parameters.
  sel = tf.expand_dims(sel, -1)
  locs = tf.reduce_sum(locs * sel, 3)
  log_scales = tf.reduce_sum(log_scales * sel, 3)
  coeffs = tf.reduce_sum(coeffs * sel, 3)

  # Sample from 3-D logistic & clip to interval. Note we don't round to the
  # nearest 8-bit value when sampling.
  uniform_noise = tf.random_uniform(
      tf.shape(locs), minval=1e-5, maxval=1. - 1e-5, seed=seed)
  logistic_noise = tf.log(uniform_noise) - tf.log(1. - uniform_noise)
  x = locs + tf.exp(log_scales) * logistic_noise
  x0 = x[..., 0]
  x1 = x[..., 1] + coeffs[..., 0] * x0
  x2 = x[..., 2] + coeffs[..., 1] * x0 + coeffs[..., 2] * x1
  x = tf.stack([x0, x1, x2], axis=-1)
  x = tf.clip_by_value(x, -1., 1.)
  return x


def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False):
  """Cross entropy with label smoothing to limit over-confidence.

  Args:
    logits: Tensor of shape [batch_size, ?, ?, ?, vocab_size].
    labels: Tensor of shape [batch_size, ?, ?, ?].
    vocab_size: Tensor representing the size of the vocabulary.
    confidence: Used to determine on and off values for label smoothing.
      If `gaussian` is true, `confidence` is the variance to the Gaussian
      distribution.
    gaussian: Uses a Gaussian distribution for label smoothing

  Returns:
    Tensor of shape [batch_size, ?, ?, ?].
  """
  with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(
        confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
        low_confidence * tf.log(low_confidence + 1e-20))

    if gaussian and confidence > 0.0:
      labels = tf.cast(labels, tf.float32)

      normal_dist = tfp.distributions.Normal(loc=labels, scale=confidence)
      # Locations to evaluate the probability distributions.
      soft_targets = normal_dist.prob(
          tf.cast(tf.range(vocab_size), tf.float32)[:, None, None, None, None])
      # Reordering soft_targets from [vocab_size, batch_size, ?, ?, ?] to match
      # logits: [batch_size, ?, ?, ?, vocab_size]
      soft_targets = tf.transpose(soft_targets, perm=[1, 2, 3, 4, 0])
    else:
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing


def global_pool_1d(inputs, pooling_type="MAX", mask=None):
  """Pool elements across the last dimension.

  Useful to convert a list of vectors into a single vector so as
  to get a representation of a set.

  Args:
    inputs: A tensor of shape [batch_size, sequence_length, input_dims]
      containing the sequences of input vectors.
    pooling_type: the pooling type to use, MAX or AVR
    mask: A tensor of shape [batch_size, sequence_length] containing a
      mask for the inputs with 1's for existing elements, and 0's elsewhere.

  Returns:
    A tensor of shape [batch_size, input_dims] containing the sequences of
    transformed vectors.
  """
  with tf.name_scope("global_pool", values=[inputs]):
    if mask is not None:
      mask = tf.expand_dims(mask, axis=2)
      inputs = tf.multiply(inputs, mask)

    if pooling_type == "MAX":
      # A tf.pool can be used here, but reduce is cleaner
      output = tf.reduce_max(inputs, axis=1)
    elif pooling_type == "AVR":
      if mask is not None:
        # Some elems are dummy elems so we can't just reduce the average.
        output = tf.reduce_sum(inputs, axis=1)
        num_elems = tf.reduce_sum(mask, axis=1, keepdims=True)
        output = tf.div(output, tf.maximum(num_elems, 1))
      else:
        output = tf.reduce_mean(inputs, axis=1)

  return output


def running_global_pool_1d(inputs, pooling_type="MAX"):
  """Same global pool, but only for the elements up to the current element.

  Useful for outputs where the state of future elements is not known.
  Takes no mask as all elements up to the current element are assumed to exist.
  Currently only supports maximum. Equivalent to using a lower triangle bias.

  Args:
    inputs: A tensor of shape [batch_size, sequence_length, input_dims]
      containing the sequences of input vectors.
    pooling_type: Pooling type to use. Currently only supports 'MAX'.

  Returns:
    A tensor of shape [batch_size, sequence_length, input_dims] containing the
    running 'totals'.
  """
  del pooling_type
  with tf.name_scope("running_global_pool", values=[inputs]):
    scan_fct = tf.maximum
    # Permute inputs so seq_length is first.
    elems = tf.transpose(inputs, [1, 0, 2])
    # Perform scan.
    cumulatives = tf.scan(scan_fct, elems, swap_memory=True)
    # Permute output to get back to original order.
    output = tf.transpose(cumulatives, [1, 0, 2])
  return output


def gated_linear_unit_layer(x, name=None):
  """Gated linear unit layer.

  Paper: Language Modeling with Gated Convolutional Networks.
  Link: https://arxiv.org/abs/1612.08083
  x = Wx * sigmoid(W'x).

  Args:
    x: A tensor
    name: A string

  Returns:
    A tensor of the same shape as x.
  """
  with tf.variable_scope(name, default_name="glu_layer", values=[x]):
    depth = shape_list(x)[-1]
    x = tf.layers.dense(x, depth * 2, activation=None)
    x, gating_x = tf.split(x, 2, axis=-1)
    return x * tf.nn.sigmoid(gating_x)


def sru_with_scan(x,
                  num_layers=2,
                  activation=None,
                  initial_state=None,
                  name=None,
                  reuse=None):
  """SRU cell as in https://arxiv.org/abs/1709.02755.

  This implementation uses tf.scan and can incur overhead, see the full SRU
  function doc for details and an implementation that is sometimes faster.

  Args:
    x: A tensor of shape [batch, ..., channels] ; ... is treated as time.
    num_layers: How many SRU layers; default is 2 as results for 1 disappoint.
    activation: Optional activation function, try tf.nn.tanh or tf.nn.relu.
    initial_state: Optional initial c-state, set to zeros if None.
    name: Optional name, "sru" by default.
    reuse: Optional reuse.

  Returns:
    A tensor of the same shape as x.

  Raises:
    ValueError: if num_layers is not positive.
  """
  if num_layers < 1:
    raise ValueError("Number of layers must be positive: %d" % num_layers)
  with tf.variable_scope(name, default_name="sru", values=[x], reuse=reuse):
    # We assume x is [batch, ..., channels] and treat all ... as time.
    x_shape = shape_list(x)
    x = tf.reshape(x, [x_shape[0], -1, x_shape[-1]])
    x = tf.transpose(x, [1, 0, 2])  # Scan assumes time on axis 0.
    initial_state = initial_state or tf.zeros([x_shape[0], x_shape[-1]])

    # SRU state manipulation function.
    def next_state(cur_state, args_tup):
      cur_x_times_one_minus_f, cur_f = args_tup
      return cur_f * cur_state + cur_x_times_one_minus_f

    # Calculate SRU on each layer.
    for i in range(num_layers):
      # The parallel part of the SRU.
      x_orig = x
      x, f, r = tf.split(
          tf.layers.dense(x, 3 * x_shape[-1], name="kernel_%d" % i), 3, axis=-1)
      f, r = tf.sigmoid(f), tf.sigmoid(r)
      x_times_one_minus_f = x * (1.0 - f)  # Compute in parallel for speed.
      # Calculate states.
      c_states = tf.scan(
          next_state, (x_times_one_minus_f, f),
          initializer=initial_state,
          parallel_iterations=2,
          name="scan_%d" % i)
      # Final output.
      if activation is not None:
        c_states = activation(c_states)
      h = c_states * r + (1.0 - r) * x_orig
      x = h  # Next layer.
    # Transpose back to batch-major.
    x = tf.transpose(x, [1, 0, 2])
    return tf.reshape(x, x_shape)


class CumsumprodCell(object):
  """Cumulative sum and product object for use with functional_rnn API."""

  def __init__(self, initializer):
    self._initializer = initializer

  @property
  def output_size(self):
    return int(shape_list(self._initializer)[-1])

  def zero_state(self, batch_size, dtype):
    dtype = dtype or tf.float32
    return tf.zeros([batch_size, self.output_size], dtype=dtype)

  def __call__(self, inputs_t, state_t):
    cur_x_times_one_minus_f, cur_f = tf.split(inputs_t, 2, axis=-1)
    state_next = cur_f * state_t + cur_x_times_one_minus_f
    outputs_t = state_next
    return outputs_t, state_next


def sru(x,
        num_layers=2,
        activation=None,
        initial_state=None,
        name=None,
        reuse=None):
  """SRU cell as in https://arxiv.org/abs/1709.02755.

  As defined in the paper:
  (1) x'_t = W x_t
  (2) f_t = sigmoid(Wf x_t + bf)
  (3) r_t = sigmoid(Wr x_t + br)
  (4) c_t = f_t * c_{t-1} + (1 - f_t) * x'_t
  (5) h_t = r_t * activation(c_t) + (1 - r_t) * x_t

  This version uses functional ops to be faster on GPUs with TF-1.9+.

  Args:
    x: A tensor of shape [batch, ..., channels] ; ... is treated as time.
    num_layers: How many SRU layers; default is 2 as results for 1 disappoint.
    activation: Optional activation function, try tf.nn.tanh or tf.nn.relu.
    initial_state: Optional initial c-state, set to zeros if None.
    name: Optional name, "sru" by default.
    reuse: Optional reuse.

  Returns:
    A tensor of the same shape as x.

  Raises:
    ValueError: if num_layers is not positive.
  """
  if num_layers < 1:
    raise ValueError("Number of layers must be positive: %d" % num_layers)
  if is_xla_compiled():  # On TPU the XLA does a good job with while.
    return sru_with_scan(x, num_layers, activation, initial_state, name, reuse)
  try:
    from tensorflow.contrib.recurrent.python.ops import functional_rnn  # pylint: disable=g-import-not-at-top
  except ImportError:
    tf.logging.info("functional_rnn not found, using sru_with_scan instead")
    return sru_with_scan(x, num_layers, activation, initial_state, name, reuse)

  with tf.variable_scope(name, default_name="sru", values=[x], reuse=reuse):
    # We assume x is [batch, ..., channels] and treat all ... as time.
    x_shape = shape_list(x)
    x = tf.reshape(x, [x_shape[0], -1, x_shape[-1]])
    initial_state = initial_state or tf.zeros([x_shape[0], x_shape[-1]])
    cell = CumsumprodCell(initial_state)
    # Calculate SRU on each layer.
    for i in range(num_layers):
      # The parallel part of the SRU.
      x_orig = x
      x, f, r = tf.split(
          tf.layers.dense(x, 3 * x_shape[-1], name="kernel_%d" % i), 3, axis=-1)
      f, r = tf.sigmoid(f), tf.sigmoid(r)
      x_times_one_minus_f = x * (1.0 - f)  # Compute in parallel for speed.
      # Calculate states.
      concat = tf.concat([x_times_one_minus_f, f], axis=-1)
      c_states, _ = functional_rnn.functional_rnn(
          cell, concat, time_major=False)
      # Final output.
      if activation is not None:
        c_states = activation(c_states)
      h = c_states * r + (1.0 - r) * x_orig
      x = h  # Next layer.
    return tf.reshape(x, x_shape)


def linear_set_layer(layer_size,
                     inputs,
                     context=None,
                     activation_fn=tf.nn.relu,
                     dropout=0.0,
                     name=None):
  """Basic layer type for doing funky things with sets.

  Applies a linear transformation to each element in the input set.
  If a context is supplied, it is concatenated with the inputs.
    e.g. One can use global_pool_1d to get a representation of the set which
    can then be used as the context for the next layer.

  TODO: Add bias add (or control the biases used).

  Args:
    layer_size: Dimension to transform the input vectors to.
    inputs: A tensor of shape [batch_size, sequence_length, input_dims]
      containing the sequences of input vectors.
    context: A tensor of shape [batch_size, context_dims] containing a global
      statistic about the set.
    activation_fn: The activation function to use.
    dropout: Dropout probability.
    name: name.

  Returns:
    Tensor of shape [batch_size, sequence_length, output_dims] containing the
    sequences of transformed vectors.
  """
  with tf.variable_scope(
      name, default_name="linear_set_layer", values=[inputs]):
    # Apply 1D convolution to apply linear filter to each element
    # along the 2nd dimension.
    outputs = conv1d(inputs, layer_size, 1, activation=None, name="set_conv")

    # Apply the context if it exists.
    if context is not None:
      # Unfortunately tf doesn't support broadcasting via concat, but we can
      # simply add the transformed context to get the same effect.
      if len(context.get_shape().as_list()) == 2:
        context = tf.expand_dims(context, axis=1)
      cont_tfm = conv1d(
          context, layer_size, 1, activation=None, name="cont_conv")
      outputs += cont_tfm

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    if dropout != 0.0:
      outputs = tf.nn.dropout(outputs, 1.0 - dropout)

    return outputs


def ravanbakhsh_set_layer(layer_size,
                          inputs,
                          mask=None,
                          sequential=False,
                          activation_fn=tf.nn.tanh,
                          dropout=0.0,
                          name=None):
  """Layer from Deep Sets paper: https://arxiv.org/abs/1611.04500 .

  More parameter-efficient version of a linear-set-layer with context.

  Args:
    layer_size: Dimension to transform the input vectors to.
    inputs: A tensor of shape [batch_size, sequence_length, vector]
      containing the sequences of input vectors.
    mask: A tensor of shape [batch_size, sequence_length] containing a
      mask for the inputs with 1's for existing elements, and 0's elsewhere.
    sequential: If true, will use a running global pool so each element will
      only depend on those before it. Set true if this layer is being used in
      an output sequence.
    activation_fn: The activation function to use.
    dropout: dropout.
    name: name.

  Returns:
    Tensor of shape [batch_size, sequence_length, vector] containing the
    sequences of transformed vectors.
  """
  del dropout
  with tf.variable_scope(name, "ravanbakhsh_set_layer", [inputs]):
    if sequential:
      return linear_set_layer(
          layer_size,
          inputs - running_global_pool_1d(inputs),
          activation_fn=activation_fn,
          name=name)
    return linear_set_layer(
        layer_size,
        inputs - tf.expand_dims(global_pool_1d(inputs, mask=mask), axis=1),
        activation_fn=activation_fn,
        name=name)


def fn_device_dependency_dict():
  """State container for fn_device_dependency."""
  if not hasattr(tf.get_default_graph(), "dependency_dict"):
    setattr(tf.get_default_graph(), "dependency_dict", defaultdict(list))
  return tf.get_default_graph().dependency_dict


@contextlib.contextmanager
def fn_device_dependency(name, device=""):
  """Add control deps for name and device."""
  key = name + "_" + device
  outs = []

  def body():
    with tf.control_dependencies(fn_device_dependency_dict()[key]):
      yield outs
      assert outs

      deps = outs
      if isinstance(outs[0], (list, tuple)):
        assert len(outs) == 1
        deps = outs[0]
      fn_device_dependency_dict()[key] = deps

  if device:
    with tf.device(device):
      return body()
  else:
    return body()


def underlying_variable_ref(t):
  """Find the underlying variable ref.

  Traverses through Identity, ReadVariableOp, and Enter ops.
  Stops when op type has Variable or VarHandle in name.

  Args:
    t: a Tensor

  Returns:
    a Tensor that is a variable ref, or None on error.
  """
  while t.op.type in ["Identity", "ReadVariableOp", "Enter"]:
    t = t.op.inputs[0]

  op_type = t.op.type
  if "Variable" in op_type or "VarHandle" in op_type:
    return t
  else:
    return None


def underlying_variable(t):
  """Find the underlying tf.Variable object.

  Args:
    t: a Tensor

  Returns:
    tf.Variable.
  """
  t = underlying_variable_ref(t)
  assert t is not None
  # make sure that the graph has a variable index and that it is up-to-date
  if not hasattr(tf.get_default_graph(), "var_index"):
    tf.get_default_graph().var_index = {}
  var_index = tf.get_default_graph().var_index
  for v in tf.global_variables()[len(var_index):]:
    var_index[v.name] = v
  return var_index[t.name]


def approximate_split(x, num_splits, axis=0):
  """Split approximately equally into num_splits parts.

  Args:
    x: a Tensor
    num_splits: an integer
    axis: an integer.

  Returns:
    a list of num_splits Tensors.
  """
  size = shape_list(x)[axis]
  size_splits = [tf.div(size + i, num_splits) for i in range(num_splits)]
  return tf.split(x, size_splits, axis=axis)


class FactoredTensor(object):
  """A concise factored representation of Tensor as two tensors.

  This class represents the tensor tf.matmul(a, b, transpose_b=True)
  by storing the values of Tensors a and b.

  The reason for this is that the product may be too big to fully realize at
  once, so it can be realized a part at a time.

  "a" may have extra leading dimensions, in which case they are flattened out
  before computing the matrix product, then re-expanded afterwards.
  """

  def __init__(self, a, b):
    self._a = a
    self._b = b

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  def to_tensor(self):
    """Convert to Tensor."""
    a_shape = shape_list(self.a)
    b_shape = shape_list(self.b)
    inner_dim = b_shape[1]
    result_dim = b_shape[0]
    flat_a = tf.reshape(self.a, [-1, inner_dim])
    product = tf.matmul(flat_a, self.b, transpose_b=True)
    product_shape = a_shape[:-1] + [result_dim]
    product = tf.reshape(product, product_shape)
    product.set_shape(self.a.get_shape().as_list()[:-1] +
                      [self.b.get_shape()[0]])
    return product


def _convert_factored_tensor_to_tensor(value, *args, **kwargs):
  # call ops.convert_to_tensor to handle optional arguments appropriately
  return ops.internal_convert_to_tensor(value.to_tensor(), *args, **kwargs)


tf.register_tensor_conversion_function(FactoredTensor,
                                       _convert_factored_tensor_to_tensor)


def smoothing_cross_entropy_factored_grad(op, dy):
  """Gradient function for smoothing_cross_entropy_factored."""
  a = op.inputs[0]
  b = op.inputs[1]
  labels = op.inputs[2]
  confidence = op.inputs[3]
  num_splits = 16
  vocab_size = shape_list(b)[0]
  labels = approximate_split(labels, num_splits)
  a = approximate_split(a, num_splits)
  dy = approximate_split(dy, num_splits)
  b_grad = None
  a_grad_parts = []
  deps = []
  for part in range(num_splits):
    with tf.control_dependencies(deps):
      logits = tf.matmul(a[part], b, transpose_b=True)
      output_part = smoothing_cross_entropy(logits, labels[part], vocab_size,
                                            confidence)
      a_grad_part, b_grad_part = tf.gradients(
          ys=[output_part], xs=[a[part], b], grad_ys=[dy[part]])
      a_grad_parts.append(a_grad_part)
      if part > 0:
        b_grad += b_grad_part
      else:
        b_grad = b_grad_part
      deps = [b_grad, a_grad_part]
  a_grad = tf.concat(a_grad_parts, 0)
  return a_grad, b_grad, None, None


@function.Defun(
    noinline=True,
    python_grad_func=smoothing_cross_entropy_factored_grad,
    compiled=True,
    separate_compiled_gradients=True)
def smoothing_cross_entropy_factored(a, b, labels, confidence):
  """Memory-efficient computation of smoothing cross-entropy.

  Avoids realizing the entire logits matrix at once.

  Args:
    a: a Tensor with shape [batch, inner_dim]
    b: a Tensor with shape [vocab_size, inner_dim]
    labels: an integer Tensor with shape [batch]
    confidence: a float

  Returns:
    A Tensor with shape [batch]
  """
  num_splits = 16
  vocab_size = shape_list(b)[0]
  labels = approximate_split(labels, num_splits)
  a = approximate_split(a, num_splits)
  parts = []
  for part in range(num_splits):
    with tf.control_dependencies(parts[-1:]):
      logits = tf.matmul(a[part], b, transpose_b=True)
      parts.append(
          smoothing_cross_entropy(logits, labels[part], vocab_size, confidence))
  return tf.concat(parts, 0)


def padded_cross_entropy_factored(factored_logits,
                                  labels,
                                  label_smoothing,
                                  weights_fn=weights_nonzero,
                                  reduce_sum=True):
  """Memory-efficient computation of smoothing cross-entropy.

  Avoids realizing the entire logits matrix at once.

  Args:
    factored_logits: a `FactoredTensor` representing a Tensor
       with shape `[batch, timesteps, vocab_size]`.
    labels: an integer `Tensor` with shape `[batch, timesteps]`.
    label_smoothing: a floating point `Scalar`.
    weights_fn: A function from labels to weights.
    reduce_sum: a Boolean, whether to sum at the end or not.

  Returns:
    loss_numerator: a `Scalar`.  Sum of losses.
    loss_denominator: a `Scalar.  The number of non-padding target tokens.
  """
  a = factored_logits.a
  b = factored_logits.b
  confidence = 1.0 - label_smoothing
  with tf.name_scope("padded_cross_entropy_factored", values=[a, b, labels]):
    labels_flat = tf.reshape(labels, [-1])
    a_flat = tf.reshape(a, [-1, shape_list(b)[1]])
    xent = smoothing_cross_entropy_factored(a_flat, b, labels_flat,
                                            tf.convert_to_tensor(confidence))
    xent = tf.reshape(xent, shape_list(labels))
    weights = weights_fn(labels)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def fn_with_custom_grad(grad_fn, use_global_vars=False):
  """Decorator to create a subgraph with a custom gradient function.

  The subgraph created by the decorated function is NOT put in a Defun and so
  does not suffer from the limitations of the Defun (all subgraph ops on the
  same device, no summaries).

  Args:
    grad_fn: function with signature
      (inputs, variables, outputs, output_grads) -> (grad_inputs, grad_vars),
      all of which are lists of Tensors.
    use_global_vars: if True, variables will be the global variables created.
      If False, will be the trainable variables.

  Returns:
    Decorator for function such that the gradient is defined by grad_fn.
  """

  def dec(fn):

    @functools.wraps(fn)
    def wrapped(*args):
      return _fn_with_custom_grad(
          fn, args, grad_fn, use_global_vars=use_global_vars)

    return wrapped

  return dec


def _fn_with_custom_grad(fn, inputs, grad_fn, use_global_vars=False):
  """Create a subgraph with a custom gradient.

  Args:
    fn: function that takes inputs as arguments and produces 1 or more Tensors.
    inputs: list<Tensor>, will be passed as fn(*inputs).
    grad_fn: function with signature
      (inputs, vars, outputs, output_grads) -> (grad_inputs, grad_vars),
      all of which are lists of Tensors.
    use_global_vars: if True, variables will be the global variables created.
      If False, will be the trainable variables.

  Returns:
    fn(*inputs)
  """
  vs = tf.get_variable_scope()
  get_vars_fn = (
      vs.global_variables if use_global_vars else vs.trainable_variables)
  len_before_vars = len(get_vars_fn())
  inputs = list(inputs)
  outputs = fn(*inputs)
  train_vars = get_vars_fn()[len_before_vars:]

  if grad_fn is None:
    return outputs

  if not isinstance(outputs, (tuple, list)):
    outputs = [outputs]
  outputs = list(outputs)

  defun_inputs = [inputs, train_vars, outputs]

  def custom_grad_fn(op, *dys):
    """Custom grad fn applying grad_fn for identity Defun."""
    fn_inputs, fn_vars, fn_outputs = tf.contrib.framework.nest.pack_sequence_as(
        defun_inputs, list(op.inputs))
    dys = list(dys)
    assert len(fn_outputs) == len(outputs)
    assert len(fn_outputs) == len(dys)

    grad_inputs, grad_vars = grad_fn(fn_inputs, fn_vars, fn_outputs, dys)
    grad_outputs = [None] * len(fn_outputs)
    return tuple(grad_inputs + grad_vars + grad_outputs)

  # The Defun takes as input the original inputs, the trainable variables
  # created in fn, and the outputs. In the forward it passes through the
  # outputs. In the backwards, it produces gradients for the original inputs
  # and the trainable variables.
  in_types = [t.dtype for t in inputs]
  out_types = [t.dtype for t in outputs]
  var_types = [t.dtype for t in train_vars]

  @function.Defun(
      *(in_types + var_types + out_types),
      func_name="identity_custom_grad%d" % ops.uid(),
      python_grad_func=custom_grad_fn,
      shape_func=lambda _: [t.get_shape() for t in outputs])
  def identity(*args):
    _, _, outs = tf.contrib.framework.nest.pack_sequence_as(defun_inputs, args)
    return tuple([tf.identity(t) for t in outs])

  flat_inputs = tf.contrib.framework.nest.flatten(defun_inputs)
  id_out = identity(*flat_inputs)
  return id_out


_function_cache = {}


def conv_hidden_relu_memory_efficient(x,
                                      filter_size,
                                      epsilon=1e-6,
                                      forget=True,
                                      test_vars=None,
                                      name=None):
  """LayerNorm, Conv, ReLU, Conv.

  All convolutions have kernel size 1.

  returns conv(relu(conv(layer_norm(x))))

  Args:
    x: input Tensor with shape [batch, length, io_size]
    filter_size: an integer - size of the hidden layer.
    epsilon: a float (for layer norm)
    forget: a boolean - forget forwards activations and recompute on backprop
    test_vars: optional tuple of variables for testing purposes
    name: an optional string

  Returns:
    a Tensor with shape [batch, length, io_size]
  """
  io_size = x.get_shape().as_list()[-1]

  def forward_internal(x, f1, f2, scale, bias):
    """Forward function."""
    # split batch-wise to avoid exhausting memory in cast the batch is large
    # and the hidden layer is large.
    num_splits = 4
    x_flat = tf.reshape(x, [-1, 1, shape_list(x)[2]])
    xs = approximate_split(x_flat, num_splits)
    ys = []
    for i in range(num_splits):
      with tf.control_dependencies(ys[-1:]):
        n = layer_norm_compute(xs[i], epsilon, scale, bias)
        y = tf.nn.conv1d(n, f1, 1, "SAME")
        y = tf.nn.relu(y)
        y = tf.nn.conv1d(y, f2, 1, "SAME")
        ys.append(y)
    y = tf.concat(ys, 0)
    y = tf.reshape(y, shape_list(x))
    return y

  key = ("conv_hidden_relu_memory_efficient %s" % epsilon)
  if not forget:
    forward_fn = forward_internal
  elif key in _function_cache:
    forward_fn = _function_cache[key]
  else:

    @function.Defun(compiled=True)
    def grad_fn(x, f1, f2, scale, bias, dy):
      """Gradient for efficiency."""
      with tf.control_dependencies([dy]):
        num_splits = 4
        x_shape = shape_list(x)
        flat_shape = [-1, 1, x_shape[2]]
        x = tf.reshape(x, flat_shape)
        dy = tf.reshape(dy, flat_shape)
        xs = approximate_split(x, num_splits)
        dys = approximate_split(dy, num_splits)
        dxs = []
        df1 = 0
        df2 = 0
        dscale = 0
        dbias = 0
        deps = []
        for i in range(num_splits):
          with tf.control_dependencies(deps):
            n = layer_norm_compute(xs[i], epsilon, scale, bias)
            y = tf.nn.conv1d(n, f1, 1, "SAME")
            y = tf.nn.relu(y)
            y = tf.nn.conv1d(y, f2, 1, "SAME")
            dxi, pdf1, pdf2, pdscale, pdbias = tf.gradients(
                ys=[y], xs=[xs[i], f1, f2, scale, bias], grad_ys=[dys[i]])
            df1 += pdf1
            df2 += pdf2
            dscale += pdscale
            dbias += pdbias
            dxs.append(dxi)
            deps = [dxi, df1, df2, dscale, dbias]
        with tf.control_dependencies(deps):
          dx = tf.concat(dxs, 0)
          dx = tf.reshape(dx, x_shape)
          return dx, df1, df2, dscale, dbias

    @function.Defun(
        grad_func=grad_fn, compiled=True, separate_compiled_gradients=True)
    def forward_fn(x, f1, f2, scale, bias):
      return forward_internal(x, f1, f2, scale, bias)

  with tf.variable_scope(name, default_name="ffn2", values=[x]):
    # TODO(noam): it would be nice to save memory by casting x to float16
    # here, but this causes problems with the gradients.  Figure out if there
    # is a way to leave the gradients as float32.
    if test_vars is not None:
      f1, f2, scale, bias = list(test_vars)
    else:
      f1 = tf.get_variable("f1", [1, io_size, filter_size])
      f2 = tf.get_variable("f2", [1, filter_size, io_size])
      scale, bias = layer_norm_vars(io_size)
    if forget:
      y = forward_fn(x, f1, f2, scale, bias)
    else:
      y = forward_internal(x, f1, f2, scale, bias)
    y.set_shape(x.get_shape())
    return y


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def list_product(els):
  prod = els[0]
  for el in els[1:]:
    prod *= el
  return prod


def sample_with_temperature(logits, temperature):
  """Either argmax or random sampling.

  Args:
    logits: a Tensor.
    temperature: a float  0.0=argmax 1.0=random

  Returns:
    a Tensor with one fewer dimension than logits.
  """
  if temperature == 0.0:
    # TF argmax doesn't handle >5 dimensions, so we reshape here.
    logits_shape = shape_list(logits)
    argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=1)
    return tf.reshape(argmax, logits_shape[:-1])
  else:
    assert temperature > 0.0
    reshaped_logits = (
        tf.reshape(logits, [-1, shape_list(logits)[-1]]) / temperature)
    choices = tf.multinomial(reshaped_logits, 1)
    choices = tf.reshape(choices,
                         shape_list(logits)[:logits.get_shape().ndims - 1])
    return choices


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones.

  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_shape: shape to reshape output by.

  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.matrix_band_part(
        tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band


def reshape_like_all_dims(a, b):
  """Reshapes a to match the shape of b."""
  ret = tf.reshape(a, tf.shape(b))
  if not tf.contrib.eager.in_eager_mode():
    ret.set_shape(b.get_shape())
  return ret


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
    variables = [underlying_variable_ref(v) for v in variables]
    # Recompute outputs
    with tf.control_dependencies(output_grads):
      with tf.contrib.framework.arg_scope(cached_arg_scope[0]):
        with tf.variable_scope(cached_vs[0], reuse=True):
          outputs = fn(*inputs)

    if not isinstance(outputs, (list, tuple)):
      outputs = [outputs]
    outputs = list(outputs)
    grads = tf.gradients(outputs, inputs + variables, output_grads)
    grad_inputs = grads[:len(inputs)]
    grad_vars = grads[len(inputs):]
    # TODO(rsepassi): Make fn_with_custom_grad work with bfloat16.
    # If the input gradients are bfloat16, it's assumed the variables are
    # bfloat16. This is a hack to ensure that grad_vars are the right type.
    if grad_inputs[0].dtype == tf.bfloat16:
      grad_vars = [tf.cast(grad_var, tf.bfloat16) for grad_var in grad_vars]
    return grad_inputs, grad_vars

  @fn_with_custom_grad(grad_fn)
  def fn_with_recompute(*args):
    cached_vs.append(tf.get_variable_scope())
    cached_arg_scope.append(tf.contrib.framework.current_arg_scope())
    return fn(*args)

  return fn_with_recompute(*args)


def dense(x, units, **kwargs):
  """Identical to tf.layers.dense."""
  return tf.layers.dense(x, units, **kwargs)


def batch_dense(inputs,
                units,
                activation=None,
                kernel_initializer=None,
                reuse=None,
                name=None):
  """Multiply a batch of input matrices by a batch of parameter matrices.

  Each input matrix is multiplied by the corresponding parameter matrix.

  This is useful in a mixture-of-experts where the batch represents different
  experts with different inputs.

  Args:
    inputs: a Tensor with shape [batch, length, input_units]
    units: an integer
    activation: an optional activation function to apply to the output
    kernel_initializer: an optional initializer
    reuse: whether to reuse the varaible scope
    name: an optional string

  Returns:
    a Tensor with shape [batch, length, units]

  Raises:
    ValueError: if the "batch" or "input_units" dimensions of inputs are not
      statically known.
  """
  inputs_shape = shape_list(inputs)
  if len(inputs_shape) != 3:
    raise ValueError("inputs must have 3 dimensions")
  batch = inputs_shape[0]
  input_units = inputs_shape[2]
  if not isinstance(batch, int) or not isinstance(input_units, int):
    raise ValueError("inputs must have static dimensions 0 and 2")
  with tf.variable_scope(
      name,
      default_name="batch_dense",
      values=[inputs],
      reuse=reuse,
      dtype=inputs.dtype):
    if kernel_initializer is None:
      kernel_initializer = tf.random_normal_initializer(
          stddev=input_units**-0.5)
    w = tf.get_variable(
        "w", [batch, input_units, units],
        initializer=kernel_initializer,
        dtype=inputs.dtype)
    y = tf.matmul(inputs, w)
    if activation is not None:
      y = activation(y)
    return y


def mix(x1,
        x2,
        steps,
        is_training,
        min_prob=0.0,
        max_prob=1.0,
        mode="lin",
        simple=False,
        broadcast_last=False):
  """Mix starting with x2, mixing mixing, going towards x1."""
  with tf.name_scope("mix"):
    if not is_training:
      if max_prob >= 1.0:
        return x1
      alpha_shape = shape_list(x1)
      if broadcast_last:
        alpha_shape = alpha_shape[:-1] + [1]
      alpha = tf.random_uniform(alpha_shape)
      alpha = tf.to_float(tf.less(alpha, max_prob))
      return alpha * x1 + (1.0 - alpha) * x2

    def get_res():
      """Create the result.

      Separate function to speed it up later (see below).

      Returns:
        Tensor of mixed inputs.
      """
      if mode == "lin":
        alpha_p = inverse_lin_decay(steps)
      else:
        alpha_p = inverse_exp_decay(steps)
      alpha_p = alpha_p * (max_prob - min_prob) + min_prob
      if simple:
        return alpha_p * x1 + (1.0 - alpha_p) * x2
      alpha_shape = shape_list(x1)
      if broadcast_last:
        alpha_shape = alpha_shape[:-1] + [1]
      alpha = tf.random_uniform(alpha_shape)
      alpha = tf.to_float(tf.less(alpha, alpha_p))
      return alpha * x1 + (1.0 - alpha) * x2

    if max_prob < 1.0:
      return get_res()

    # Prevent sampling after steps is passed to speed it up.
    if is_xla_compiled():
      return get_res()
    else:
      cur_step = tf.train.get_global_step()
      if cur_step is None:
        return x1  # Step not available, probably eval mode, don't mix.
      return tf.cond(tf.less(cur_step, steps), get_res, lambda: x1)


def brelu(x):
  """Bipolar ReLU as in https://arxiv.org/abs/1709.04054."""
  x_shape = shape_list(x)
  x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
  y1 = tf.nn.relu(x1)
  y2 = -tf.nn.relu(-x2)
  return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def belu(x):
  """Bipolar ELU as in https://arxiv.org/abs/1709.04054."""
  x_shape = shape_list(x)
  x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
  y1 = tf.nn.elu(x1)
  y2 = -tf.nn.elu(-x2)
  return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def nac(x, depth, name=None, reuse=None):
  """NAC as in https://arxiv.org/abs/1808.00508."""
  with tf.variable_scope(name, default_name="nac", values=[x], reuse=reuse):
    x_shape = shape_list(x)
    w = tf.get_variable("w", [x_shape[-1], depth])
    m = tf.get_variable("m", [x_shape[-1], depth])
    w = tf.tanh(w) * tf.nn.sigmoid(m)
    x_flat = tf.reshape(x, [-1, x_shape[-1]])
    res_flat = tf.matmul(x_flat, w)
    return tf.reshape(res_flat, x_shape[:-1] + [depth])


def nalu(x, depth, epsilon=1e-30, name=None, reuse=None):
  """NALU as in https://arxiv.org/abs/1808.00508."""
  with tf.variable_scope(name, default_name="nalu", values=[x], reuse=reuse):
    x_shape = shape_list(x)
    x_flat = tf.reshape(x, [-1, x_shape[-1]])
    gw = tf.get_variable("w", [x_shape[-1], depth])
    g = tf.nn.sigmoid(tf.matmul(x_flat, gw))
    g = tf.reshape(g, x_shape[:-1] + [depth])
    a = nac(x, depth, name="nac_lin")
    log_x = tf.log(tf.abs(x) + epsilon)
    m = nac(log_x, depth, name="nac_log")
    return g * a + (1 - g) * tf.exp(m)


def argmax_with_score(logits, axis=None):
  """Argmax along with the value."""
  axis = axis or len(logits.get_shape()) - 1
  predictions = tf.argmax(logits, axis=axis)

  logits_shape = shape_list(logits)
  prefix_shape, vocab_size = logits_shape[:-1], logits_shape[-1]
  prefix_size = 1
  for d in prefix_shape:
    prefix_size *= d

  # Flatten to extract scores
  flat_logits = tf.reshape(logits, [prefix_size, vocab_size])
  flat_predictions = tf.reshape(predictions, [prefix_size])
  flat_indices = tf.stack(
      [tf.range(tf.to_int64(prefix_size)),
       tf.to_int64(flat_predictions)],
      axis=1)
  flat_scores = tf.gather_nd(flat_logits, flat_indices)

  # Unflatten
  scores = tf.reshape(flat_scores, prefix_shape)

  return predictions, scores


def log_prob_from_logits(logits, reduce_axis=-1):
  return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)


def top_1_tpu(inputs):
  """find max and argmax over the last dimension.

  Works well on TPU

  Args:
    inputs: A tensor with shape [..., depth]

  Returns:
    values: a Tensor with shape [...]
    indices: a Tensor with shape [...]
  """
  inputs_max = tf.reduce_max(inputs, axis=-1, keepdims=True)
  mask = tf.to_int32(tf.equal(inputs_max, inputs))
  index = tf.range(tf.shape(inputs)[-1]) * mask
  return tf.squeeze(inputs_max, -1), tf.reduce_max(index, axis=-1)


def index_last_dim_with_indices(x, indices):
  """Use indices to index into the last axis of x.

  This can be useful for recovering the actual probabilities of a sample from a
  probability distribution.

  Args:
    x: Tensor, n-d.
    indices: Tensor, (n-1)-d, where the dimension sizes match the first (n-1)
      dimensions of x. The values of indices will be used to index into the last
      axis of x.

  Returns:
    Tensor, (n-1)-d.
  """
  assert len(x.shape) == len(indices.shape) + 1

  x_shape = shape_list(x)
  vocab_size = x_shape[-1]

  flat_x = tf.reshape(x, [list_product(x_shape[:-1]), vocab_size])
  flat_indices = tf.reshape(indices, [list_product(x_shape[:-1])])

  idx = tf.stack(
      [
          tf.range(tf.to_int64(shape_list(flat_indices)[0])),
          tf.to_int64(flat_indices)
      ],
      axis=1)
  flat_x_idx = tf.gather_nd(flat_x, idx)

  x_idx = tf.reshape(flat_x_idx, x_shape[:-1])

  return x_idx


def should_generate_summaries():
  """Is this an appropriate context to generate summaries.

  Returns:
    a boolean
  """
  name_scope = tf.contrib.framework.get_name_scope()
  if name_scope and "while/" in name_scope:
    # Summaries don't work well within tf.while_loop()
    return False
  if tf.get_variable_scope().reuse:
    # Avoid generating separate summaries for different data shards
    return False
  return True


def reshape_like(a, b):
  """Reshapes a to match the shape of b in all but the last dimension."""
  ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
  if not tf.contrib.eager.in_eager_mode():
    ret.set_shape(b.get_shape().as_list()[:-1] + a.get_shape().as_list()[-1:])
  return ret


def summarize_video(video, prefix, max_outputs=1):
  """Summarize the video using image summaries starting with prefix."""
  video_shape = shape_list(video)
  if len(video_shape) != 5:
    raise ValueError("Assuming videos given as tensors in the format "
                     "[batch, time, height, width, channels] but got one "
                     "of shape: %s" % str(video_shape))
  if tf.contrib.eager.in_eager_mode():
    return
  if video.get_shape().as_list()[1] is None:
    tf.summary.image(
        "%s_last_frame" % prefix,
        tf.cast(video[:, -1, :, :, :], tf.uint8),
        max_outputs=max_outputs)
  else:
    for k in range(video_shape[1]):
      tf.summary.image(
          "%s_frame_%d" % (prefix, k),
          tf.cast(video[:, k, :, :, :], tf.uint8),
          max_outputs=max_outputs)


def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x.name,
                       x.device, cast_x.device)
  return cast_x


def make_even_size(x):
  """Pad x to be even-sized on axis 1 and 2, but only if necessary."""
  x_shape = x.get_shape().as_list()
  assert len(x_shape) > 2, "Only 3+-dimensional tensors supported."
  shape = [dim if dim is not None else -1 for dim in x_shape]
  new_shape = x_shape  # To make sure constant shapes remain constant.
  if x_shape[1] is not None:
    new_shape[1] = 2 * int(math.ceil(x_shape[1] * 0.5))
  if x_shape[2] is not None:
    new_shape[2] = 2 * int(math.ceil(x_shape[2] * 0.5))
  if shape[1] % 2 == 0 and shape[2] % 2 == 0:
    return x
  if shape[1] % 2 == 0:
    x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=2)
    x.set_shape(new_shape)
    return x
  if shape[2] % 2 == 0:
    x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=1)
    x.set_shape(new_shape)
    return x
  x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=1)
  x, _ = pad_to_same_length(x, x, final_length_divisible_by=2, axis=2)
  x.set_shape(new_shape)
  return x


def sliced_gan_loss(input1,
                    input2,
                    discriminator,
                    num_vecs,
                    do_random_vecs=True,
                    do_tanh=True,
                    return_logits=False):
  """Loss inspired by the sliced WGAN paper: https://arxiv.org/abs/1804.01947.

  Puts input1 and input2 through the provided discriminator to get logits.
  Then, computes num_vecs random projections of the logits, sorts them on
  the batch dimension and returns the L2 loss between the sorted vectors.
  See the above-mentioned paper for the reasoning behind it.

  Args:
    input1: first discriminator inputs.
    input2: second discriminator inputs.
    discriminator: inputs -> logits function.
    num_vecs: how many random vectors to use for projections.
    do_random_vecs: whether to use random vectors or just tanh of the logits.
    do_tanh: if true (default) we'll also just use tanh of the logits.
    return_logits: Whether or not to return the logits.

  Returns:
    The generator loss, i.e., the sliced approximation of the distance between
    the projected distributions (warning: discriminator should maximize it).
  """
  with tf.variable_scope("sliced_gan"):
    with tf.variable_scope("discriminator"):
      logits1 = discriminator(input1)
    with tf.variable_scope("discriminator", reuse=True):
      logits2 = discriminator(input2)

    if do_random_vecs:
      random_vecs = tf.nn.l2_normalize(
          tf.random_uniform([shape_list(logits1)[-1], num_vecs]), axis=0)

    def get_sorted_projections(x):
      """Make projections of x and sort them on the batch dimension."""
      x = tf.reshape(x, [-1, shape_list(x)[-1]])
      batch_size = shape_list(x)[0]
      if do_random_vecs and do_tanh:
        n = tf.nn.l2_normalize(x, axis=1)
        proj = tf.concat([tf.matmul(n, random_vecs), tf.tanh(n)], axis=1)
      elif do_random_vecs:
        n = tf.nn.l2_normalize(x, axis=1)
        proj = tf.matmul(n, random_vecs)
      else:
        proj = tf.tanh(x)
      proj = tf.transpose(proj, [1, 0])  # [num_vecs, batch] after this.

      if is_xla_compiled():
        proj_dtype = proj.dtype
        proj = tf.cast(proj, tf.bfloat16)

        # Currently TPU only supports 1-D top_k calls.
        map_fn = lambda x: tf.nn.top_k(x, k=batch_size, sorted=True)[0]
        values = tf.map_fn(map_fn, proj)

        values = tf.cast(values, proj_dtype)
      else:
        values, _ = tf.nn.top_k(proj, k=batch_size, sorted=True)

      return values

    proj1 = get_sorted_projections(logits1)
    proj2 = get_sorted_projections(logits2)
    dist = tf.reduce_mean(tf.square(proj1 - proj2))
    if return_logits:
      return dist, logits1, logits2
    return dist


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.maximum(input_, leak * input_, name=name)


def deep_discriminator(x,
                       batch_norm,
                       is_training,
                       filters=64,
                       filter_size=4,
                       stride=2,
                       output_size=1024):
  """Discriminator architecture based on InfoGAN."""
  with tf.variable_scope(
      "discriminator", initializer=tf.random_normal_initializer(stddev=0.02)):
    batch_size, height, width = shape_list(x)[:3]
    net = tf.layers.conv2d(
        x, filters, filter_size, strides=stride, padding="SAME", name="conv1")
    net = lrelu(net)
    net = tf.layers.conv2d(
        net,
        2 * filters,
        filter_size,
        strides=stride,
        padding="SAME",
        name="conv2")
    # [bs, h/4, w/4, 128]
    if batch_norm:
      net = tf.layers.batch_normalization(
          net, training=is_training, momentum=0.999, name="d_bn2")
    net = lrelu(net)
    size = height * width
    x_shape = x.get_shape().as_list()
    if x_shape[1] is None or x_shape[2] is None:
      net = tf.reduce_mean(net, axis=[1, 2])
    else:
      net = tf.reshape(net, [batch_size, size * 8])
    net = tf.layers.dense(net, output_size, name="d_fc3")
    if batch_norm:
      net = tf.layers.batch_normalization(
          net, training=is_training, momentum=0.999, name="d_bn3")
    net = lrelu(net)
    return net


def instance_norm(x):
  """Instance normalization layer."""
  with tf.variable_scope("instance_norm"):
    epsilon = 1e-5
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    scale = tf.get_variable(
        "scale", [x.get_shape()[-1]],
        initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
    offset = tf.get_variable(
        "offset", [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
    out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

    return out


def general_conv(x,
                 num_filters=64,
                 filter_size=7,
                 stride=1,
                 stddev=0.02,
                 padding="VALID",
                 name="conv",
                 do_norm="instance",
                 do_relu=True,
                 relufactor=0):
  """Generalized convolution layer."""
  with tf.variable_scope(name):
    x = tf.layers.conv2d(
        x,
        num_filters,
        filter_size,
        stride,
        padding,
        activation=None,
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        bias_initializer=tf.constant_initializer(0.0))
    if do_norm == "layer":
      x = tf.contrib.layers.layer_norm(x)
    elif do_norm == "instance":
      x = instance_norm(x)

    if do_relu:
      if relufactor == 0:
        x = tf.nn.relu(x, "relu")
      else:
        x = lrelu(x, leak=relufactor)

    return x


def patch_discriminator(x, filters=64, filter_size=5, n=4,
                        name="patch_discrim"):
  """Patch descriminator."""
  with tf.variable_scope(name):
    x_shape = shape_list(x)
    spatial_dims = [x_shape[1] // 4, x_shape[2] // 4]
    x = tf.random_crop(x, [x_shape[0]] + spatial_dims + [x_shape[3]])
    for i in range(n):
      x = general_conv(
          x=x,
          num_filters=filters * 2**i,
          filter_size=filter_size,
          stride=2 if i != n - 1 else 1,
          stddev=0.02,
          padding="SAME",
          name="c%d" % i,
          do_norm="instance" if i != 0 else False,
          do_relu=i != n - 1,
          relufactor=0.2)
    x = tf.reduce_mean(x, [1, 2])
    return x


def mean_with_attention(x, name, num_heads=4):
  """Mean and attention to reduce spatial dimensions."""
  with tf.variable_scope(name):
    shape = shape_list(x)
    m = tf.reduce_mean(x, [1, 2])
    a = tf.layers.dense(x, num_heads, name="mean_attn")
    s = tf.reshape(a, [shape[0], -1, num_heads])
    s = tf.nn.softmax(s, axis=1)
    s = tf.reshape(s, shape[:-1] + [1, num_heads])
    am = tf.reduce_mean(tf.expand_dims(x, axis=-1) * s, [1, 2])
    l = tf.concat([am, tf.expand_dims(m, axis=-1)], axis=-1)
    return tf.layers.dense(tf.reshape(l, [shape[0], (num_heads+1) * shape[-1]]),
                           2 * shape[-1], name="mean_attn_final")


def single_discriminator(x, filters=128, kernel_size=8,
                         strides=4, pure_mean=False):
  """A simple single-layer convolutional discriminator."""
  with tf.variable_scope("discriminator"):
    net = tf.layers.conv2d(
        x, filters, kernel_size, strides=strides, padding="SAME", name="conv1")
    if pure_mean:
      net = tf.reduce_mean(net, [1, 2])
    else:
      net = mean_with_attention(net, "mean_with_attention")
    return net


def double_discriminator(x, filters1=128, filters2=None,
                         kernel_size=8, strides=4, pure_mean=False):
  """A convolutional discriminator with 2 layers and concatenated output."""
  if filters2 is None:
    filters2 = 4 * filters1
  with tf.variable_scope("discriminator"):
    batch_size = shape_list(x)[0]
    net = tf.layers.conv2d(
        x, filters1, kernel_size, strides=strides, padding="SAME", name="conv1")
    if pure_mean:
      net1 = tf.reduce_mean(net, [1, 2])
    else:
      net1 = mean_with_attention(net, "mean_with_attention1")
      tf.reshape(net, [batch_size, -1])
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(
        x, filters2, kernel_size, strides=strides, padding="SAME", name="conv2")
    if pure_mean:
      net2 = tf.reduce_mean(net, [1, 2])
    else:
      net2 = mean_with_attention(net, "mean_with_attention2")
    return tf.concat([net1, net2], axis=-1)


def upscale(inputs, f, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
  """Upscaling the image by a factor of f."""
  height, width = shape_list(inputs)[1:3]
  return tf.image.resize_images(inputs, (height * f, width * f), method)


def tpu_safe_image_summary(image):
  if is_xla_compiled():
    # We only support float32 images at the moment due to casting complications.
    if image.dtype != tf.float32:
      image = tf.to_float(image)
  else:
    image = tf.cast(image, tf.uint8)
  return image


# This has been (shamefully) copied from
# GitHub tensorflow/models/blob/master/research/slim/nets/cyclegan.py
#
# tensorflow/models cannot be pip installed, and even if it were we don't want
# to depend on all the models in it.
#
# Therefore copying and forgoing any more bugfixes into it is the most
# expedient way to use this function.
def cyclegan_upsample(net, num_outputs, stride, method="conv2d_transpose"):
  """Upsamples the given inputs.

  Args:
    net: A Tensor of size [batch_size, height, width, filters].
    num_outputs: The number of output filters.
    stride: A list of 2 scalars or a 1x2 Tensor indicating the scale,
      relative to the inputs, of the output dimensions. For example, if kernel
      size is [2, 3], then the output height and width will be twice and three
      times the input size.
    method: The upsampling method: 'nn_upsample_conv',
      'bilinear_upsample_conv', or 'conv2d_transpose'.

  Returns:
    A Tensor which was upsampled using the specified method.

  Raises:
    ValueError: if `method` is not recognized.
  """

  with tf.variable_scope("upconv"):
    net_shape = tf.shape(net)
    height = net_shape[1]
    width = net_shape[2]

    # Reflection pad by 1 in spatial dimensions (axes 1, 2 = h, w) to make a
    # 3x3 "valid" convolution produce an output with the same dimension as the
    # input.
    spatial_pad_1 = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])

    if method == "nn_upsample_conv":
      net = tf.image.resize_nearest_neighbor(
          net, [stride[0] * height, stride[1] * width])
      net = tf.pad(net, spatial_pad_1, "REFLECT")
      net = tf.contrib.layers.conv2d(
          net, num_outputs, kernel_size=[3, 3], padding="valid")
    elif method == "bilinear_upsample_conv":
      net = tf.image.resize_bilinear(net,
                                     [stride[0] * height, stride[1] * width])
      net = tf.pad(net, spatial_pad_1, "REFLECT")
      net = tf.contrib.layers.conv2d(
          net, num_outputs, kernel_size=[3, 3], padding="valid")
    elif method == "conv2d_transpose":
      # This corrects 1 pixel offset for images with even width and height.
      # conv2d is left aligned and conv2d_transpose is right aligned for even
      # sized images (while doing "SAME" padding).
      # Note: This doesn"t reflect actual model in paper.
      net = tf.contrib.layers.conv2d_transpose(
          net, num_outputs, kernel_size=[3, 3], stride=stride, padding="valid")
      net = net[:, 1:, 1:, :]
    else:
      raise ValueError("Unknown method: [%s]" % method)

    return net


def weight_targeting(w, k):
  """Weight-level magnitude pruning."""
  k = tf.to_int32(k)
  w_shape = shape_list(w)
  size = tf.to_int32(tf.reduce_prod(w_shape[:-1]))
  w = tf.reshape(w, [size, w_shape[-1]])

  transpose_w = tf.transpose(w)
  thres = tf.contrib.framework.sort(tf.abs(transpose_w), axis=1)[:, k]
  mask = tf.to_float(thres[None, :] >= tf.abs(w))

  return tf.reshape(mask, w_shape)


def unit_targeting(w, k):
  """Unit-level magnitude pruning."""
  k = tf.to_int32(k)
  w_shape = shape_list(w)
  size = tf.to_int32(tf.reduce_prod(w_shape[:-1]))
  w = tf.reshape(w, [size, w_shape[-1]])

  norm = tf.norm(w, axis=0)
  thres = tf.contrib.framework.sort(norm, axis=0)[k]
  mask = tf.to_float(thres >= norm)[None, :]
  mask = tf.tile(mask, [size, 1])

  return tf.reshape(mask, w_shape)


def td_conv(inputs,
            filters,
            kernel_size,
            targeting_count,
            targeting_fn,
            keep_prob,
            is_training,
            do_prune=True,
            strides=(1, 1),
            padding="valid",
            data_format="channels_last",
            dilation_rate=(1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            name=None,
            reuse=None):
  """Apply targeted dropout to the weights of a convolution."""
  with tf.variable_scope(name, default_name="td_conv", reuse=reuse):
    nhwc = data_format == "channels_last"
    in_dim = shape_list(inputs)[-1] if nhwc else shape_list(inputs)[1]

    kernel_shape = [kernel_size, kernel_size, in_dim, filters]
    w = tf.get_variable(
        "DW", shape=kernel_shape, initializer=kernel_initializer)
    if use_bias:
      b = tf.get_variable("b", shape=[filters], initializer=bias_initializer)

    if keep_prob < 1.0:
      w = targeted_dropout(
          w,
          targeting_count,
          keep_prob,
          targeting_fn,
          is_training,
          do_prune=do_prune)

    if isinstance(strides, int):
      strides = [strides, strides]
    if isinstance(dilation_rate, int):
      dilation_rate = [dilation_rate, dilation_rate]

    if nhwc:
      strides = [1, strides[0], strides[1], 1]
      dilation_rate = [1, dilation_rate[0], dilation_rate[1], 1]
    else:
      strides = [1, 1, strides[0], strides[1]]
      dilation_rate = [1, 1, dilation_rate[0], dilation_rate[1]]

    y = tf.nn.conv2d(
        inputs,
        w,
        strides,
        padding,
        data_format="NHWC" if nhwc else "NCHW",
        dilations=dilation_rate,
        name=None)

    if use_bias:
      y += b

    if activation:
      y = activation(y)

    return y


def targeted_dropout(inputs,
                     k,
                     keep_prob,
                     targeting_fn,
                     is_training,
                     do_prune=False):
  """Applies targeted dropout.

  Applies dropout at a rate of `1 - keep_prob` to only those elements of
  `inputs` marked by `targeting_fn`. See below and paper for more detail:

  "Targeted Dropout for Posthoc Pruning" Aidan N. Gomez, Ivan Zhang,
    Kevin Swersky, Yarin Gal, and Geoffrey E. Hinton.

  Args:
    inputs: Tensor, inputs to apply targeted dropout to.
    k: Scalar Tensor or python scalar, sets the number of elements to target in
      `inputs`. Must be within `[0, tf.shape(x)[-1]]` and compatible with
      second argument of `targeting_fn`.
    keep_prob: Scalar Tensor, passed as `tf.nn.dropout`'s `keep_prob` argument.
    targeting_fn: callable `fn(inputs, k) -> Boolean Tensor`, produces a
      boolean mask the same shape as `inputs` where True indicates an element
      will be dropped, and False not.
    is_training: bool, indicates whether currently training.
    do_prune: bool, indicates whether to prune the `k * (1 - keep_prob)`
      elements of `inputs` expected to be dropped each forwards pass.

  Returns:
    Tensor, same shape and dtype as `inputs`.
  """
  if not is_training and do_prune:
    k = tf.round(tf.to_float(k) * tf.to_float(1. - keep_prob))

  mask = targeting_fn(inputs, k)
  mask = tf.cast(mask, inputs.dtype)

  if is_training:
    return inputs * (1 - mask) + tf.nn.dropout(inputs, keep_prob) * mask
  elif do_prune:
    return inputs * (1 - mask)
  else:
    return inputs


def kl_divergence(mu, log_var, mu_p=0.0, log_var_p=0.0):
  """KL divergence of diagonal gaussian N(mu,exp(log_var)) and N(0,1).

  Args:
    mu: mu parameter of the distribution.
    log_var: log(var) parameter of the distribution.
    mu_p: optional mu from a learned prior distribution
    log_var_p: optional log(var) from a learned prior distribution
  Returns:
    the KL loss.
  """

  batch_size = shape_list(mu)[0]
  prior_distribution = tfp.distributions.Normal(
      mu_p, tf.exp(tf.multiply(0.5, log_var_p)))
  posterior_distribution = tfp.distributions.Normal(
      mu, tf.exp(tf.multiply(0.5, log_var)))
  kld = tfp.distributions.kl_divergence(posterior_distribution,
                                        prior_distribution)
  return tf.reduce_sum(kld) / tf.to_float(batch_size)


def sparse_equals_constant(constant, tensor):
  return tf.SparseTensor(
      indices=tensor.indices,
      dense_shape=tensor.dense_shape,
      values=tf.equal(tensor.values, constant))


def sparse_expand_dims(tensor, current_num_dims, axis=0):
  if axis == -1:
    axis = current_num_dims

  new_col = tf.zeros([tf.shape(tensor.indices)[0]], dtype=tf.int64)
  cols = tf.unstack(tensor.indices, axis=1, num=current_num_dims)
  shape = tf.unstack(tensor.dense_shape, num=current_num_dims)
  new_indices = tf.stack(cols[:axis] + [new_col] + cols[axis:], axis=1)
  return tf.SparseTensor(
      indices=new_indices,
      values=tensor.values,
      dense_shape=tf.stack(shape[:axis] + [1] + shape[axis:]))


def sparse_add_constant(constant, tensor):
  return tf.SparseTensor(
      indices=tensor.indices,
      values=constant + tensor.values,
      dense_shape=tensor.dense_shape)


def sparse_eye(size):
  indices = tf.cast(tf.stack([tf.range(size), tf.range(size)]), tf.int64)
  values = tf.ones(size)
  dense_shape = [tf.cast(size, tf.int64), tf.cast(size, tf.int64)]

  return tf.SparseTensor(
      indices=indices, values=values, dense_shape=dense_shape)


# modification from https://github.com/tensorflow/tensorflow/pull/21276
# without special initialization for g
class WeightNorm(tf.keras.layers.Wrapper):
  """ This wrapper reparameterizes a layer by decoupling the weight's
  magnitude and direction. This speeds up convergence by improving the
  conditioning of the optimization problem.

  Weight Normalization: A Simple Reparameterization to Accelerate
  Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
  Tim Salimans, Diederik P. Kingma (2016)

  WeightNorm wrapper works for keras and tf layers.

  ```python
    net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
           input_shape=(32, 32, 3), data_init=True)(x)
    net = WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu'),
                     data_init=True)
    net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                     data_init=True)(net)
    net = WeightNorm(tf.keras.layers.Dense(n_classes),
                     data_init=True)(net)
  ```

  Arguments:
    layer: a layer instance.
    data_init: If `True` use data dependent variable initialization

  Raises:
    ValueError: If not initialized with a `Layer` instance.
    ValueError: If `Layer` does not contain a `kernel` of weights
    NotImplementedError: If `data_init` is True and running graph execution
  """

  def __init__(self, layer, data_init=False, **kwargs):
    if not isinstance(layer, tf.keras.layers.Layer):
      raise ValueError(
          "Please initialize `WeightNorm` layer with a "
          "`Layer` instance. You passed: {input}".format(input=layer))

    super(WeightNorm, self).__init__(layer, **kwargs)
    self._track_checkpointable(layer, name="layer")

  def _compute_weights(self):
    """Generate weights with normalization."""
    with tf.variable_scope("compute_weights"):
      self.layer.kernel = tf.nn.l2_normalize(
          self.layer.v, axis=self.norm_axes) * self.layer.g

  def _init_norm(self, weights):
    """Set the norm of the weight vector."""
    with tf.variable_scope("init_norm"):
      flat = tf.reshape(weights, [-1, self.layer_depth])
      return tf.reshape(tf.norm(flat, axis=0), (self.layer_depth,))

  def _data_dep_init(self, inputs):
    """Data dependent initialization for eager execution."""

    with tf.variable_scope("data_dep_init"):
      # Generate data dependent init values
      activation = self.layer.activation
      self.layer.activation = None
      x_init = self.layer.call(inputs)
      m_init, v_init = tf.moments(x_init, self.norm_axes)
      scale_init = 1. / tf.sqrt(v_init + 1e-10)

    # Assign data dependent init values
    self.layer.g = self.layer.g * scale_init
    self.layer.bias = (-m_init * scale_init)
    self.layer.activation = activation
    self.initialized = True

  def build(self, input_shape=None):
    """Build `Layer`."""
    input_shape = tf.TensorShape(input_shape).as_list()
    self.input_spec = tf.layers.InputSpec(shape=input_shape)

    if not self.layer.built:
      self.layer.build(input_shape)
      self.layer.built = False

      if not hasattr(self.layer, "kernel"):
        raise ValueError("`WeightNorm` must wrap a layer that"
                         " contains a `kernel` for weights")

      # The kernel's filter or unit dimension is -1
      self.layer_depth = int(self.layer.kernel.shape[-1])
      self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

      self.layer.v = self.layer.kernel
      self.layer.g = self.layer.add_variable(
          name="g",
          shape=(self.layer_depth,),
          initializer=tf.ones_initializer,
          dtype=self.layer.kernel.dtype,
          trainable=True)

      # with ops.control_dependencies([self.layer.g.assign(
      #     self._init_norm(self.layer.v))]):
      #   self._compute_weights()
      self._compute_weights()

      self.layer.built = True

    super(WeightNorm, self).build()
    self.built = True

  def call(self, inputs):
    """Call `Layer`."""
    # if context.executing_eagerly():
    #   if not self.initialized:
    #     self._data_dep_init(inputs)
    self._compute_weights()  # Recompute weights for each forward pass

    output = self.layer.call(inputs)
    return output

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(
        self.layer.compute_output_shape(input_shape).as_list())
