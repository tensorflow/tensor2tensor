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
import math
import random

# Dependency imports

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.utils import expert_utils as eu

import tensorflow as tf

from tensorflow.python.eager import context as tfe_context
from tensorflow.python.framework import function
from tensorflow.python.framework import ops

# This is a global setting. When turned off, no @function.Defun is used.
allow_defun = False


def is_on_tpu():
  return tf.contrib.framework.get_name_scope().startswith("TPUReplicate")


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
    A Tensor with the same size and shape as x.
  """
  assert "noise_shape" not in kwargs
  if broadcast_dims:
    shape = tf.shape(x)
    ndims = len(x.get_shape())
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in xrange(ndims)]
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


def inverse_exp_decay(max_step, min_value=0.01):
  """Inverse-decay exponentially from 0.01 to 1.0 reached at max_step."""
  inv_base = tf.exp(tf.log(min_value) / float(max_step))
  step = tf.to_float(tf.train.get_global_step())
  return inv_base**tf.maximum(float(max_step) - step, 0.0)


def inverse_lin_decay(max_step, min_value=0.01):
  """Inverse-decay linearly from 0.01 to 1.0 reached at max_step."""
  step = tf.to_float(tf.train.get_global_step())
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
    # Use the formula (value/128) - 1 to convert each channel value into a
    # real number in the range -1 to 1.
    x = (x / 128) - 1
    return x


def flatten4d3d(x):
  """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
  xshape = shape_list(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  return result


# TODO(noam): remove this function after TPUs do gather faster.
def gather(params, indices):
  """Version of tf.gather that works faster on tpu."""
  if not is_on_tpu():
    return tf.gather(params, indices)
  vocab_size = params.get_shape().as_list()[0]
  indices_flat = tf.reshape(indices, [-1])
  out = tf.matmul(tf.one_hot(indices_flat, vocab_size), params)
  out = eu.reshape_like(out, tf.expand_dims(indices, -1))
  return out


def dropout_no_scaling(x, keep_prob):
  """Like tf.nn.dropout, but does not scale up.  Works on integers also.

  Args:
    x: a Tensor
    keep_prob: a floating point number
  Returns:
    a Tensor of the same size and shape as x
  """
  if keep_prob == 1.0:
    return x
  return x * tf.cast(
      tf.less(tf.random_uniform(tf.shape(x)), keep_prob), x.dtype)


def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0,
              symbol_dropout_rate=0.0, embedding_var=None):
  """Embed x of type int64 into dense vectors, reducing to max 4 dimensions."""
  with tf.variable_scope(
      name, default_name="embedding", values=[x], reuse=reuse):
    if embedding_var is None:
      embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
    # On the backwards pass, we want to convert the gradient from
    # an indexed-slices to a regular tensor before sending it back to the
    # parameter server. This avoids excess computation on the parameter server.
    if not tfe_context.in_eager_mode():
      embedding_var = eu.convert_gradient_to_tensor(embedding_var)
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
    for i in xrange(nbr_steps):
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
    for i in xrange(nbr_steps):
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


def conv(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
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
  for i in xrange(kernel_size):
    shifted = tf.slice(padded, [0, i, 0], tf.shape(inputs)) if i else inputs
    shifted.set_shape(inputs.get_shape())
    results.append(dense(
        shifted, filters, use_bias=(i == 0), name=name + "_%d" % i))
  ret = tf.add_n(results)
  ret *= kernel_size ** -0.5
  return ret


def layer_norm_vars(filters):
  """Create Variables for layer norm."""
  scale = tf.get_variable(
      "layer_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
  return scale, bias


def layer_norm_compute_python(x, epsilon, scale, bias):
  """Layer norm raw computation."""
  mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x * scale + bias


@function.Defun(compiled=True)
def layer_norm_compute_grad(x, epsilon, scale, bias, dy):
  y = layer_norm_compute_python(x, epsilon, scale, bias)
  dx = tf.gradients(ys=[y], xs=[x, epsilon, scale, bias], grad_ys=[dy])
  return dx


@function.Defun(
    compiled=True,
    separate_compiled_gradients=True,
    grad_func=layer_norm_compute_grad)
def layer_norm_compute(x, epsilon, scale, bias):
  return layer_norm_compute_python(x, epsilon, scale, bias)


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = x.get_shape()[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    scale = tf.get_variable(
        "layer_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
    if allow_defun:
      result = layer_norm_compute(x, tf.constant(epsilon), scale, bias)
      result.set_shape(x.get_shape())
    else:
      result = layer_norm_compute_python(x, epsilon, scale, bias)
    return result


def noam_norm(x, epsilon=1.0, name=None):
  """One version of layer normalization."""
  with tf.name_scope(name, default_name="noam_norm", values=[x]):
    shape = x.get_shape()
    ndims = len(shape)
    return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
        tf.to_float(shape[-1])))


def apply_norm(x, norm_type, depth, epsilon):
  """Apply Normalization."""
  if norm_type == "layer":
    return layer_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "batch":
    return tf.layers.batch_normalization(x, epsilon=epsilon)
  if norm_type == "noam":
    return noam_norm(x, epsilon)
  if norm_type == "none":
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'none'.")


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

  A hyperparemeters object is passed for convenience.  The hyperparameters
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

  A hyperparemeters object is passed for convenience.  The hyperparameters
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


def decompress_seqcnn(x,
                      targets,
                      targets_vocab_size,
                      dilations_and_kernels,
                      block_size,
                      is_2d=False,
                      embedding_var=None,
                      name=None,
                      reuse=None):
  """Decompress x into targets size using a Sequence CNN at every element."""
  with tf.variable_scope(
      name,
      default_name="decompress_batch_seqcnn",
      values=[x, targets],
      reuse=reuse):
    # We assume targets are [batch x block_size * N x block_size * N x C] if
    # is_2d=True or [batch, block_size * N, 1, C] otherwise, and C is static.
    # Let's shift targets to depth and embed.
    targets_shape = shape_list(targets)
    channels = targets_shape[-1]
    hidden_size = x.get_shape()[-1]
    if is_2d:
      depth_targets = tf.space_to_depth(targets, block_size)
      factor = channels * block_size * block_size
    else:
      depth_targets = tf.reshape(targets, [
          targets_shape[0], targets_shape[1] // block_size, 1,
          channels * block_size
      ])
      factor = channels * block_size
    if embedding_var is None:
      embedding_var = tf.get_variable("targets_embedding",
                                      [targets_vocab_size, hidden_size])
    targets_emb = tf.gather(embedding_var, depth_targets)
    # Flatten x and embedded targets. Flat targets are factor* larger on axis=1.
    flat_x = tf.reshape(x, [-1, 1, 1, hidden_size])
    flat_targets = tf.reshape(targets_emb, [-1, factor, 1, hidden_size])
    shifted_targets = shift_right(flat_targets)
    # Run a SeqCNN large-batch to produce factor outputs out of every target.
    flat_x += tf.zeros_like(shifted_targets)  # Broadcast on axis=1.
    flat_outputs = conv_block(
        tf.concat([flat_x, shifted_targets], axis=3),
        hidden_size,
        dilations_and_kernels,
        padding="LEFT")
    # Reshape back to embedded targets shape.
    targets_emb_shape = shape_list(targets_emb)
    outputs = tf.reshape(flat_outputs, [
        targets_emb_shape[0], targets_emb_shape[1], targets_emb_shape[2],
        factor * hidden_size
    ])
    # Move depth back to target space.
    if is_2d:
      outputs = tf.depth_to_space(outputs, 2)
    else:
      outputs = tf.reshape(outputs, [
          shape_list(outputs)[0], block_size * shape_list(outputs)[1], 1,
          hidden_size
      ])
    # Final reshape before prediction to ensure target size.
    outputs = tf.reshape(outputs, [
        targets_shape[0], targets_shape[1], targets_shape[2], channels,
        hidden_size
    ])
    return dense(outputs, targets_vocab_size)


def simple_attention(target, source, bias=None):
  """A simple attention function.

  Args:
    target: a `Tensor` with shape `[batch, target_timesteps, depth]` or
     `[batch, target_timesteps_1, target_timesteps_2, depth]`
    source: a `Tensor` with shape `[batch, source_timesteps, depth]` or
     `[batch, source_timesteps_1, source_timesteps_2, depth]`
    bias: an optional `Tensor` with shape `[batch, timesteps, 1, 1]` used
     to mask the attention to not attend to padding of input.

  Returns:
    a `Tensor` with same shape as `target`
  """
  with tf.name_scope("simple_attention", values=[target, source]):
    target_shape = shape_list(target)
    source_shape = shape_list(source)
    target = tf.reshape(
        target,
        [target_shape[0], target_shape[1] * target_shape[2], target_shape[3]])
    source = tf.reshape(
        source,
        [source_shape[0], source_shape[1] * source_shape[2], source_shape[3]])
    attention = tf.matmul(target, source, transpose_b=True)
    attention *= tf.rsqrt(tf.to_float(shape_list(target)[2]))
    if bias is not None:
      attention += tf.expand_dims(tf.squeeze(bias, axis=[2, 3]), axis=1)
    attention = tf.nn.softmax(attention)
    if eu.should_generate_summaries():
      tf.summary.image("attention", tf.expand_dims(attention, 3), max_outputs=5)
    attended = tf.matmul(attention, source)
    return tf.reshape(attended, target_shape)


def multiscale_conv_sum(inputs, output_size, dilation_rates_and_kernel_sizes,
                        pooling_type, **kwargs):
  """Sum of several dilated convolutions.

  For all convolutions with dilation_rate > 1, we first pool the input with
  width dilation_rate.

  Args:
    inputs: a Tensor
    output_size: an Integer
    dilation_rates_and_kernel_sizes: a list of pairs (dilation, kernel_size)
    pooling_type: "AVG" or "MAX"
    **kwargs: additional

  Returns:
     a Tensor.
  """
  name = kwargs.pop("name") if "name" in kwargs else None
  with tf.variable_scope(name, "multiscale_conv_sum", [inputs]):
    padding = kwargs["padding"]
    results, counter = [], -1
    for dilation_rate, kernel_size in dilation_rates_and_kernel_sizes:
      counter += 1
      if dilation_rate[0] > 1:
        pooled = pool(inputs, kernel_size, pooling_type, padding)
      else:
        pooled = inputs
      results.append(
          conv(
              pooled,
              output_size,
              kernel_size,
              dilation_rate=dilation_rate,
              name="conv_layer%d" % counter,
              **kwargs))
    return tf.add_n(results) * (len(results)**-0.5)


def multiscale_conv_and_attention(x, padding, hparams, source=None):
  """A common part of t2t layers.

  First, do a linear multiscale convolution
  Second, do attention (if source is not None)

  Applies residuals and normalization on both steps.

  Args:
    x: a Tensor.
    padding: a padding type
    hparams: hyperparameters for model
    source: optional source tensor for attention. (encoder output)

  Returns:
    a Tensor.
  """
  # TODO(noam): The number of different scales should be a hyperparameter.
  conv_sum = multiscale_conv_sum(
      x,
      hparams.hidden_size,
      [((hparams.kernel_height**i, hparams.kernel_width**i),
        (hparams.kernel_height, hparams.kernel_width)) for i in xrange(3)],
      "AVG",
      padding=padding)
  # For residuals a rescale if necessary if channels differ.
  if x.get_shape().as_list()[-1] != conv_sum.get_shape().as_list()[-1]:
    x = conv(x, hparams.hidden_size, (1, 1))
  x = noam_norm(x + conv_sum)
  if source is not None:
    x = noam_norm(x + simple_attention(x, source))
  return x


def conv_with_pools(inputs, output_size, kernel_size, pool_sizes, pooling_type,
                    **kwargs):
  """Convolution plus 1x1 convolution applied to specified pools.

  For example we might do a regular convolution with kernel size (3, 1),
  and pools of sizes [(9, 1), (27, 1)].

  Args:
    inputs: a Tensor
    output_size: an Integer
    kernel_size: a tuple of integers
    pool_sizes: a list of tuples of integers.
    pooling_type: "AVG" or "MAX"
    **kwargs: additional keyword args for conv

  Returns:
     a Tensor.
  """
  name = kwargs.pop("name") if "name" in kwargs else None
  with tf.variable_scope(name, "conv_with_pools", [inputs]):
    padding = kwargs["padding"]
    results = []
    results.append(conv(inputs, output_size, kernel_size, **kwargs))
    for i, pool_size in enumerate(pool_sizes):
      pooled = pool(inputs, pool_size, pooling_type, padding)
      results.append(
          conv(pooled, output_size, (1, 1), name="pool_%d" % i, **kwargs))
    return tf.add_n(results) * (len(results)**-0.5)


def conv_with_pools_and_attention(x, padding, hparams, source=None):
  """A common part of t2t layers.

  First, do conv_with_pools
  Second, do attention (if source is not None)

  Applies residuals and normalization on both steps.

  Args:
    x: a Tensor.
    padding: a padding type
    hparams: hyperparameters for model
    source: optional source tensor for attention. (encoder output)

  Returns:
    a Tensor.
  """
  conv_sum = conv_with_pools(
      x,
      hparams.hidden_size, (hparams.kernel_height, hparams.kernel_width),
      hparams.pool_sizes,
      "AVG",
      padding=padding)
  if x.get_shape().as_list()[-1] == conv_sum.get_shape().as_list()[-1]:
    conv_sum += x
  x = noam_norm(conv_sum)
  if source is not None:
    x = noam_norm(x + simple_attention(x, source))
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
  experessed in terms of y, sin(x) and cos(x).

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
  return weights_nonzero(tf.reduce_sum(tf.abs(emb), axis=3, keep_dims=True))


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


def attention_1d_v0(source,
                    target,
                    attention_size,
                    output_size,
                    num_heads,
                    mask=None,
                    transform_source=True,
                    transform_target=True,
                    transform_output=True,
                    name=None):
  """multi-headed attention.

  TODO(noam): this could probably be extended to 2d.

  Args:
    source: a Tensor of shape [batch, source_length, source_depth]
    target: a Tensor of shape [batch, target_length, target_depth]
    attention_size: an integer
    output_size: an integer
    num_heads: an integer divisor of attention_size
    mask: a float32 Tensor of shape [batch, target_length, source_length]
          1.0 means can-see; 0.0 means can't-see.
          Any dimension can be 1 (supports broadcasting).
    transform_source: a boolean
    transform_target: a boolean
    transform_output: a boolean
    name: an optional string

  Returns:
    a Tensor of shape [batch, length, output_size]
  """
  with tf.variable_scope(name, default_name="attention", values=[target]):
    source_shape = shape_list(source)
    source_length = source_shape[1]
    target_length = shape_list(target)[1]
    batch = source_shape[0]

    def _maybe_transform(t, size, should_transform, name):
      if should_transform:
        return conv1d(t, size, 1, name=name)
      else:
        assert t.get_shape()[-1] == size
        return t

    source_attention = _maybe_transform(source, attention_size,
                                        transform_source, "source_attention")
    target_attention = _maybe_transform(target, attention_size,
                                        transform_target, "target_attention")
    assert attention_size % num_heads == 0
    size_per_head = attention_size // num_heads
    source_attention = tf.reshape(
        source_attention, [batch, source_length, num_heads, size_per_head])
    target_attention = tf.reshape(
        target_attention, [batch, target_length, num_heads, size_per_head])
    # [batch, num_heads, length, size_per_head]
    source_attention = tf.transpose(source_attention, [0, 2, 1, 3])
    target_attention = tf.transpose(target_attention, [0, 2, 1, 3])

    # [batch, num_heads, target_length, source_length]
    attention = tf.matmul(target_attention, source_attention, transpose_b=True)
    attention *= size_per_head**-0.5

    if mask is not None:
      mask = tf.expand_dims(mask, 1)
      mask = (1.0 - mask) * -1e9
      attention += mask
    attention = tf.nn.softmax(attention)
    if eu.should_generate_summaries():
      # Compute a color image summary.
      image = tf.reshape(attention,
                         [batch, num_heads, target_length, source_length])
      image = tf.transpose(image, [0, 2, 3, 1])
      image = tf.pow(image, 0.2)  # for high-dynamic-range
      # Each head will correspond to one of RGB.
      # pad the heads to be a multiple of 3
      extra_heads = -num_heads % 3
      image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, -num_heads % 3]])
      image = tf.reshape(image, [
          batch, target_length, source_length, 3, (num_heads + extra_heads) // 3
      ])
      image = tf.reduce_max(image, 4)
      tf.summary.image("local_attention", image, max_outputs=1)
    # output: [batch, num_heads, target_length, size_per_head]
    output = tf.matmul(attention, source_attention)
    output = tf.transpose(output, [0, 2, 1, 3])
    output = tf.reshape(output, [batch, target_length, attention_size])
    output = _maybe_transform(output, output_size, transform_output,
                              "attention_output")
    return output


def relu_density_logit(x, reduce_dims):
  """logit(density(x)).

  Useful for histograms.

  Args:
    x: a Tensor, typilcally the output of tf.relu
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
    a Tensor with the same shape as inputs
  """
  if (kernel_size != 1 and
      kernel_size != (1, 1) and
      nonpadding_mask is not None):
    while nonpadding_mask.get_shape().ndims < inputs.get_shape().ndims:
      nonpadding_mask = tf.expand_dims(nonpadding_mask, -1)
    return inputs * nonpadding_mask
  else:
    return inputs


def dense_relu_dense(inputs, filter_size, output_size, dropout=0.0,
                     dropout_broadcast_dims=None):
  """Hidden layer with RELU activation followed by linear projection."""
  h = dense(
      inputs, filter_size, use_bias=True, activation=tf.nn.relu, name="conv1")
  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = dense(h, output_size, use_bias=True, name="conv2")
  return o


def conv_relu_conv(inputs,
                   filter_size,
                   output_size,
                   first_kernel_size=3,
                   second_kernel_size=3,
                   padding="SAME",
                   nonpadding_mask=None,
                   dropout=0.0,
                   name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  with tf.variable_scope(name, "conv_relu_conv", [inputs]):
    inputs = maybe_zero_out_padding(
        inputs, first_kernel_size, nonpadding_mask)
    h = tpu_conv1d(inputs, filter_size, first_kernel_size, padding=padding,
                   name="conv1")
    h = tf.nn.relu(h)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
    return tpu_conv1d(h, output_size, second_kernel_size, padding=padding,
                      name="conv2")


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
    inputs = maybe_zero_out_padding(
        inputs, first_kernel_size, nonpadding_mask)
    if inputs.get_shape().ndims == 3:
      is_3d = True
      inputs = tf.expand_dims(inputs, 2)
    else:
      is_3d = False
    h = separable_conv(
        inputs, filter_size, first_kernel_size, activation=tf.nn.relu,
        padding=padding, name="conv1")
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
    if len(labels.shape.as_list()) == 3:  # 2-d labels.
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
    gaussian: If true, use a gaussian distribution for label smoothing

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
  vocab_size = shape_list(logits)[-1]
  with tf.name_scope("padded_cross_entropy", values=[logits, labels]):
    if len(logits.get_shape().as_list()) == 2:
      # Deal with the case where we did not insert extra dimensions due to
      # TPU issues.  No pad-to-same-length happens in this case.
      # TODO(noam): remove this logic once TPU can handle extra dimensions.
      labels = tf.reshape(labels, [-1])
    else:
      logits, labels = pad_with_zeros(logits, labels)
    xent = smoothing_cross_entropy(logits, labels, vocab_size, confidence,
                                   gaussian=gaussian)
    weights = weights_fn(labels)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False):
  """Cross entropy with label smoothing to limit over-confidence.

  Args:
    logits: Tensor of size [batch_size, ?, ?, ?, vocab_size]
    labels: Tensor of size [batch_size, ?, ?, ?]
    vocab_size: Tensor representing the size of the vocabulary.
    confidence: Used to determine on and off values for label smoothing.
      If `gaussian` is true, `confidence` is the variance to the gaussian
      distribution.
    gaussian: Uses a gaussian distribution for label smoothing

  Returns:

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

      normal_dist = tf.distributions.Normal(loc=labels, scale=confidence)
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
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing


def global_pool_1d(inputs, pooling_type="MAX", mask=None):
  """Pool elements across the last dimension.

  Useful to convert a list of vectors into a single vector so as
  to get a representation of a set.

  Args:
    inputs: A tensor of dimensions batch_size x sequence_length x input_dims
      containing the sequences of input vectors.
    pooling_type: the pooling type to use, MAX or AVR
    mask: A tensor of dimensions batch_size x sequence_length containing a
      mask for the inputs with 1's for existing elements, and 0's elsewhere.

  Returns:
    output: A tensor of dimensions batch_size x input_dims
      dimension containing the sequences of transformed vectors.
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
        num_elems = tf.reduce_sum(mask, axis=1, keep_dims=True)
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
    inputs: A tensor of dimensions batch_size x sequence_length x input_dims
      containing the sequences of input vectors.
    pooling_type: Pooling type to use. Currently only supports 'MAX'.

  Returns:
    output: A tensor of dimensions batch_size x sequence_length x input_dims
      dimension containing the running 'totals'.
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
    x: A tensor
  """

  with tf.variable_scope(
      name, default_name="glu_layer", values=[x]):
    depth = shape_list(x)[-1]
    x = tf.layers.dense(x, depth * 2, activation=None)
    x, gating_x = tf.split(x, 2, axis=-1)
    return x * tf.nn.sigmoid(gating_x)


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
    inputs: A tensor of dimensions batch_size x sequence_length x input_dims
      containing the sequences of input vectors.
    context: A tensor of dimensions batch_size x context_dims
      containing a global statistic about the set.
    activation_fn: The activation function to use.
    dropout: Dropout probability.
    name: name.

  Returns:
    output: A tensor of dimensions batch_size x sequence_length x output_dims
      dimension containing the sequences of transformed vectors.
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

  More parameter-efficient verstion of a linear-set-layer with context.

  Args:
    layer_size: Dimension to transform the input vectors to.
    inputs: A tensor of dimensions batch_size x sequence_length x vector
      containing the sequences of input vectors.
    mask: A tensor of dimensions batch_size x sequence_length containing a
      mask for the inputs with 1's for existing elements, and 0's elsewhere.
    sequential: If true, will use a running global pool so each element will
      only depend on those before it. Set true if this layer is being used in
      an output sequence.
    activation_fn: The activation function to use.
    dropout: dropout.
    name: name.

  Returns:
    output: A tensor of dimensions batch_size x sequence_length x vector
      dimension containing the sequences of transformed vectors.
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
      if isinstance(outs[0], list) or isinstance(outs[0], tuple):
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
    a tf.Varaible object.
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
  size_splits = [tf.div(size + i, num_splits) for i in xrange(num_splits)]
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
    product.set_shape(
        self.a.get_shape().as_list()[:-1] + [self.b.get_shape()[0]])
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
  for part in xrange(num_splits):
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
  for part in xrange(num_splits):
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

  if not (isinstance(outputs, tuple) or isinstance(outputs, list)):
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
      func_name="identity_custom_grad%d" % random.randint(1, 10**9),
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
    for i in xrange(num_splits):
      with tf.control_dependencies(ys[-1:]):
        n = layer_norm_compute_python(xs[i], epsilon, scale, bias)
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
        for i in xrange(num_splits):
          with tf.control_dependencies(deps):
            n = layer_norm_compute_python(xs[i], epsilon, scale, bias)
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
  for i in xrange(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def sample_with_temperature(logits, temperature):
  """Either argmax or random sampling.

  Args:
    logits: a Tensor.
    temperature: a float  0.0=argmax 1.0=random

  Returns:
    a Tensor with one fewer dimension than logits.
  """
  if temperature == 0.0:
    return tf.argmax(logits, -1)
  else:
    assert temperature > 0.0
    reshaped_logits = (
        tf.reshape(logits, [-1, shape_list(logits)[-1]]) / temperature)
    choices = tf.multinomial(reshaped_logits, 1)
    choices = tf.reshape(choices,
                         shape_list(logits)[:logits.get_shape().ndims - 1])
    return choices


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones."""
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
    band = tf.matrix_band_part(tf.ones([rows, cols]),
                               tf.cast(num_lower, tf.int64),
                               tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band


def reshape_like_all_dims(a, b):
  """Reshapes a to match the shape of b."""
  ret = tf.reshape(a, tf.shape(b))
  if not tfe_context.in_eager_mode():
    ret.set_shape(b.get_shape())
  return ret


def reduce_by_device(parallelism, data, reduce_fn):
  """Reduces data per device.

  This can be useful, for example, if we want to all-reduce n tensors on k<n
  devices (like during eval when we have only one device).  We call
  reduce_by_device() to first sum the tensors per device, then call our usual
  all-reduce operation to create one sum per device, followed by
  expand_by_device, to create the appropriate number of pointers to these
  results.  See all_reduce_ring() below for an example of how this is used.

  Args:
    parallelism: a expert_utils.Parallelism object
    data: a list of Tensors with length parallelism.n
    reduce_fn: a function taking a list of Tensors.  e.g. tf.add_n

  Returns:
    device_parallelism: a Parallelism object with each device listed only once.
    reduced_data: A list of Tensors, one per device.
  """
  unique_devices = []
  device_to_data = {}
  for dev, datum in zip(parallelism.devices, data):
    if dev not in device_to_data:
      unique_devices.append(dev)
      device_to_data[dev] = [datum]
    else:
      device_to_data[dev].append(datum)
  device_parallelism = eu.Parallelism(unique_devices)
  grouped_data = [device_to_data[dev] for dev in unique_devices]
  return device_parallelism, device_parallelism(reduce_fn, grouped_data)


def expand_by_device(original_parallelism, device_parallelism, data):
  """Opposite of reduce_by_device().

  Args:
    original_parallelism: a expert_utils.Parallelism object.
    device_parallelism: a expert_utils.Parallelism object.
    data: a list of tensors with length device_parallelism.n

  Returns:
    a list of Tensors with length original_parallelism.n
  """
  device_to_datum = {
      device_parallelism.devices[i]: data[i]
      for i in xrange(device_parallelism.n)}
  return [device_to_datum[d] for d in original_parallelism.devices]


def all_reduce_ring(x, parallelism, maybe_reduce=True, use_bfloat16=True):
  """Compute the sum of all Tensors and put the result everywhere.

  Assumes that the devices are connected in a ring.

  Args:
    x: a list of Tensors with length parallelism.n
    parallelism: a expert_utils.Parallelism object.
    maybe_reduce: a boolean - first reduce per device.
    use_bfloat16: a boolean - saves bandwidth but loses precision

  Returns:
    a list of Tensors with length parallelism.n
  """
  if parallelism.n == 1:
    return x

  if maybe_reduce:
    original_parallelism = parallelism
    parallelism, x = reduce_by_device(parallelism, x, tf.add_n)

  if parallelism.n == 1:
    y = x
  else:
    # first shard the input:
    x_flat = parallelism(tf.reshape, x, [[-1]] * parallelism.n)
    # [device, shard]
    x_split = parallelism(approximate_split, x_flat, parallelism.n, 0)
    def _step(source_replica, target_replica, x_split, op="plus_eq"):
      """Helper function - one step of summing or copying.

      If op == "plus_eq", then adds source_replica into target_replica
      If op == "copy", then copies source_replica onto target_replica

      These operations happen for all shards.  The replica numbers are offset
      by the shard numbers to keep all physical links busy.

      Args:
        source_replica: an integer
        target_replica: an integer
        x_split: a list of lists of tensors
        op: a string
      """
      for shard in xrange(parallelism.n):
        source_device = (shard + source_replica) % parallelism.n
        target_device = (shard + target_replica) % parallelism.n
        source = x_split[source_device][shard]
        if use_bfloat16:
          with tf.device(parallelism.devices[source_device]):
            source = tf.to_bfloat16(source)
        with tf.device(parallelism.devices[target_device]):
          source = tf.to_float(source)
          if op == "plus_eq":
            x_split[target_device][shard] += source
          else:
            assert op == "copy"
            x_split[target_device][shard] = tf.identity(source)
    center = parallelism.n // 2
    # accumulate everything towards the center.
    for i in range(center, parallelism.n - 1)[::-1]:
      _step(i + 1, i, x_split, op="plus_eq")
    for i in xrange(center):
      _step(i, i + 1, x_split, op="plus_eq")
    # copy everything away from the center.
    for i in xrange(center, parallelism.n - 1):
      _step(i, i + 1, x_split, op="copy")
    for i in range(center)[::-1]:
      _step(i + 1, i, x_split, op="copy")
    x_concat = parallelism(tf.concat, x_split, 0)
    y = parallelism(reshape_like_all_dims, x_concat, x)
  if maybe_reduce:
    y = expand_by_device(original_parallelism, parallelism, y)
  return y


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

    if not (isinstance(outputs, list) or isinstance(outputs, tuple)):
      outputs = [outputs]
    outputs = list(outputs)
    grads = tf.gradients(outputs, inputs + variables, output_grads)
    grad_inputs = grads[:len(inputs)]
    grad_vars = grads[len(inputs):]
    if is_on_tpu():
      # TODO(noam): remove this hack once XLA does the right thing.
      # Force the gradinets on the inputs to be computed before the variables
      # are updated.  This saves memory by preventing XLA from making an extra
      # copy of the variables.
      grad_vars = force_dependency(grad_vars, grad_inputs)
    return grad_inputs, grad_vars

  @fn_with_custom_grad(grad_fn)
  def fn_with_recompute(*args):
    cached_vs.append(tf.get_variable_scope())
    # TODO(rsepassi): Rm conditional in TF 1.5
    if hasattr(tf.contrib.framework, "current_arg_scope"):
      cached_arg_scope.append(tf.contrib.framework.current_arg_scope())
    else:
      cached_arg_scope.append({})
    return fn(*args)

  return fn_with_recompute(*args)


def force_dependency(xs, ys):
  """Force all of xs to depend on all of ys, using a false data dependency.

  XLA seems to ignore control dependencies.

  Args:
    xs: a list of tensors
    ys: a list of tensors:
  Returns:
    a list of tensors of the same length as xs
  """
  def _first_element(x):
    ndims = x.get_shape().ndims
    return tf.reshape(tf.slice(x, [0] * ndims, [1] * ndims), [])
  my_zero = tf.add_n([_first_element(y) for y  in ys if y is not None]) * 1e-30
  return [x + my_zero for x in xs]


def dense(x, units, **kwargs):
  """Identical to tf.layers.dense, Memory optimization on tpu."""
  fn = lambda x: tf.layers.dense(x, units, **kwargs)
  if is_on_tpu():
    # TODO(noam): remove this hack once XLA does the right thing.
    # Forces the gradinets on the inputs to be computed before the variables
    # are updated.  This saves memory by preventing XLA from making an extra
    # copy of the variables.
    return _recompute_grad(fn, [x])
  else:
    return fn(x)


def mix(x1, x2, steps, is_training,
        min_prob=0.0, max_prob=1.0, mode="lin", simple=False):
  """Mix starting with x2, mixing mixing, going towards x1."""
  if not is_training:
    return x1

  def get_res():
    """Create the result. Separate function to speed it up later (see below)."""
    if mode == "lin":
      alpha_p = inverse_lin_decay(steps)
    else:
      alpha_p = inverse_exp_decay(steps)
    alpha_p = alpha_p * (max_prob - min_prob) + min_prob
    if simple:
      return alpha_p * x1 + (1.0 - alpha_p) * x2
    alpha = tf.random_uniform(shape_list(x1))
    alpha = tf.to_float(tf.less(alpha, alpha_p))
    return alpha * x1 + (1.0 - alpha) * x2

  if max_prob < 1.0:
    return get_res()

  # Prevent sampling after steps is passed to speed it up.
  return tf.cond(tf.less(tf.train.get_global_step(), steps),
                 get_res, lambda: x1)
