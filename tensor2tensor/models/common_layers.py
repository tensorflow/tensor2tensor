# Copyright 2017 Google Inc.
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

import math

# Dependency imports

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.utils import expert_utils as eu

import tensorflow as tf

from tensorflow.python.framework import function

# This is a global setting. When turned off, no @function.Defun is used.
allow_defun = True


def saturating_sigmoid(x):
  """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
  with tf.name_scope("saturating_sigmoid", [x]):
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
  step = tf.to_float(tf.contrib.framework.get_global_step())
  return inv_base**tf.maximum(float(max_step) - step, 0.0)


def standardize_images(x):
  """Image standardization on batches (tf.image.per_image_standardization)."""
  with tf.name_scope("standardize_images", [x]):
    x = tf.to_float(x)
    x_mean = tf.reduce_mean(x, axis=[1, 2, 3], keep_dims=True)
    x_variance = tf.reduce_mean(
        tf.square(x - x_mean), axis=[1, 2, 3], keep_dims=True)
    num_pixels = tf.to_float(tf.shape(x)[1] * tf.shape(x)[2] * 3)
    x = (x - x_mean) / tf.maximum(tf.sqrt(x_variance), tf.rsqrt(num_pixels))
    # TODO(lukaszkaiser): remove hack below, needed for greedy decoding for now.
    if x.shape and len(x.shape) == 4 and x.shape[3] == 1:
      x = tf.concat([x, x, x], axis=3)  # Not used, just a dead tf.cond branch.
    x.set_shape([None, None, None, 3])
    return x


def image_augmentation(images, do_colors=False):
  """Image augmentation: cropping, flipping, and color transforms."""
  images = tf.random_crop(images, [299, 299, 3])
  images = tf.image.random_flip_left_right(images)
  if do_colors:  # More augmentation, but might be slow.
    images = tf.image.random_brightness(images, max_delta=32. / 255.)
    images = tf.image.random_saturation(images, lower=0.5, upper=1.5)
    images = tf.image.random_hue(images, max_delta=0.2)
    images = tf.image.random_contrast(images, lower=0.5, upper=1.5)
  return images


def flatten4d3d(x):
  """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
  xshape = tf.shape(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  # Preserve static shapes when available.
  xshape_static = x.get_shape()
  result.set_shape([xshape_static[0], None, xshape_static[3]])
  return result


def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
  """Embed x of type int64 into dense vectors, reducing to max 4 dimensions."""
  with tf.variable_scope(
      name, default_name="embedding", values=[x], reuse=reuse):
    embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
    # On the backwards pass, we want to convert the gradient from
    # an indexed-slices to a regular tensor before sending it back to the
    # parameter server. This avoids excess computation on the parameter server.
    embedding_var = eu.ConvertGradientToTensor(embedding_var)
    emb_x = tf.gather(embedding_var, x)
    if multiplier != 1.0:
      emb_x *= multiplier
    shape, static_shape = tf.shape(emb_x), emb_x.shape.as_list()
    if not static_shape or len(static_shape) < 5:
      return emb_x
    # If we had extra channel dimensions, assume it's 1, i.e. shape[3] == 1.
    assert len(static_shape) == 5
    return tf.reshape(emb_x, [shape[0], shape[1], shape[2], static_shape[4]])


def shift_left(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
  return shifted_targets


def shift_left_3d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
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
      cur_shape = tf.shape(cur)
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
        cur = tf.cond(
            tf.equal(tf.shape(cur)[2], 1),
            lambda idx=i: deconv1d(cur, idx),
            lambda idx=i: deconv2d(cur, idx))
    return cur


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4.")
  inputs.set_shape([static_shape[0], None, None, static_shape[3]])
  # Add support for left padding.
  if "padding" in kwargs and kwargs["padding"] == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    kwargs["padding"] = "VALID"
  force2d = False  # Special argument we use to force 2d kernels (see below).
  if "force2d" in kwargs:
    force2d = kwargs["force2d"]

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    if "name" in kwargs:
      original_name = kwargs["name"]
      name = kwargs.pop("name") + "_" + name_suffix
    else:
      original_name = None
      name = "conv_" + name_suffix
    original_force2d = None
    if "force2d" in kwargs:
      original_force2d = kwargs.pop("force2d")
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result

  # Manually setting the shape to be unknown in the middle two dimensions so
  # that the `tf.cond` below won't throw an error based on the convolution
  # kernels being too large for the data.
  inputs._shape = tf.TensorShape([static_shape[0], None, None, static_shape[3]])  # pylint: disable=protected-access
  if kernel_size[1] == 1 or force2d:
    # Avoiding the cond below can speed up graph and gradient construction.
    return conv2d_kernel(kernel_size, "single")
  return tf.cond(
      tf.equal(tf.shape(inputs)[2],
               1), lambda: conv2d_kernel((kernel_size[0], 1), "small"),
      lambda: conv2d_kernel(kernel_size, "std"))


def conv(inputs, filters, kernel_size, **kwargs):
  return conv_internal(tf.layers.conv2d, inputs, filters, kernel_size, **kwargs)


def conv1d(inputs, filters, kernel_size, **kwargs):
  return tf.squeeze(
      conv(tf.expand_dims(inputs, 2), filters, (kernel_size, 1), **kwargs), 2)


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
                tf.layers.conv2d(split, filters // separability, kernel_size, **
                                 kwargs))
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


def noam_norm(x, name=None):
  """One version of layer normalization."""
  with tf.name_scope(name, default_name="noam_norm", values=[x]):
    shape = x.get_shape()
    ndims = len(shape)
    return (tf.nn.l2_normalize(x, ndims - 1, epsilon=1.0) *
            tf.sqrt(tf.to_float(shape[-1])))


def residual_function(hparams):
  """Returns a function for combining layer input and layer output.

  The returned function on x (layer input) and y (layer output) computes:
    norm_function(x + t

  Args:
    hparams: model hyperparameters

  Returns:
    a function from x=<layer input> and y=<layer output> to computed output
  """

  def residual_fn(x, y):
    return hparams.norm_function(x + tf.nn.dropout(
        y, 1.0 - hparams.residual_dropout))

  return residual_fn


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
  norm = kwargs.pop("normalizer_fn") if "normalizer_fn" in kwargs else None
  if norm is None and "normalizer_fn" not in kwargs:
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
      if norm is not None:
        cur = norm(cur, name="conv_block_norm_%d" % counter)
    return cur


def conv_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard convolutions."""
  return conv_block_internal(conv, inputs, filters,
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
  with tf.name_scope("pool", [inputs]):
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
            tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
            lambda: tf.constant(2 * (window_size[1] // 2)))
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding_ = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
      inputs = tf.pad(inputs, padding_)
      inputs.set_shape([static_shape[0], None, None, static_shape[3]])
      padding = "VALID"
    window_size_small = (window_size[0], 1)
    strides_small = (strides[0], 1)
    # Manually setting the shape to be unknown in the middle two dimensions so
    # that the `tf.cond` below won't throw an error based on the convolution
    # kernels being too large for the data.
    inputs._shape = tf.TensorShape(  # pylint: disable=protected-access
        [static_shape[0], None, None, static_shape[3]])
    return tf.cond(
        tf.equal(tf.shape(inputs)[2], 1),
        lambda: tf.nn.pool(  # pylint: disable=g-long-lambda
            inputs, window_size_small, pooling_type, padding,
            strides=strides_small),
        lambda: tf.nn.pool(  # pylint: disable=g-long-lambda
            inputs, window_size, pooling_type, padding, strides=strides))


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
    targets_shape, targets_shape_static = tf.shape(targets), targets.get_shape()
    channels = int(targets_shape_static[-1])
    hidden_size = int(x.get_shape()[-1])
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
    shifted_targets = shift_left(flat_targets)
    # Run a SeqCNN large-batch to produce factor outputs out of every target.
    flat_x += tf.zeros_like(shifted_targets)  # Broadcast on axis=1.
    flat_outputs = conv_block(
        tf.concat([flat_x, shifted_targets], axis=3),
        hidden_size,
        dilations_and_kernels,
        padding="LEFT")
    # Reshape back to embedded targets shape.
    outputs = tf.reshape(flat_outputs, [
        tf.shape(targets_emb)[0],
        tf.shape(targets_emb)[1],
        tf.shape(targets_emb)[2], factor * hidden_size
    ])
    # Move depth back to target space.
    if is_2d:
      outputs = tf.depth_to_space(outputs, 2)
    else:
      outputs = tf.reshape(outputs, [
          tf.shape(outputs)[0], block_size * tf.shape(outputs)[1], 1,
          hidden_size
      ])
    # Final reshape before prediction to ensure target size.
    outputs = tf.reshape(outputs, [
        targets_shape[0], targets_shape[1], targets_shape[2], channels,
        hidden_size
    ])
    return tf.layers.dense(outputs, targets_vocab_size)


def moe_layer(data_parallelism,
              ps_devices,
              xs,
              train,
              model_hidden_size,
              expert_hidden_size,
              n1,
              n2,
              loss_coef,
              autoscale=True,
              name=None):
  """A mixture of experts layer.

  Args:
    data_parallelism: a expert_utils.Parallelism object.
    ps_devices: a list of strings
    xs: a list of input tensors.
    train: a boolean scalar.
    model_hidden_size: an integer (input/output size for this layer)
    expert_hidden_size: an integer (size of each expert's hidden layer)
    n1: an integer - number of experts (or # of groups for hierarchical MoE)
    n2: optional integer - size of each group of experts for hierarchical MoE
    loss_coef: a scalar - multiplier on load-balancing losses
    autoscale: a boolean
    name: a string

  Returns:
    ys: a list of tensors:
    extra_training_loss: a scalar
  """
  dp = data_parallelism
  with tf.variable_scope(name, default_name="moe"):
    # Set up the hyperparameters for the gating networks.
    primary_gating_hp = eu.NoisyTopKGatingParams()
    primary_gating_hp.num_experts = n1
    if n2:
      # hierarchical MoE containing moe_n1 groups of moe_n2 experts.
      assert n2 > 1
      secondary_gating_hp = eu.NoisyTopKGatingParams()
      secondary_gating_hp.num_experts = n2
    else:
      # flat mixture of moe_n1 experts.
      secondary_gating_hp = None
    # Set up the hyperparameters for the expert networks.
    # Each expert contains a hidden RELU layer of size filter_size
    expert_hp = eu.FeedForwardExpertParams()
    expert_hp.autoscale = autoscale
    expert_hp.hidden_layer_sizes = [expert_hidden_size]
    # Create the mixture of experts.
    moe = eu.DistributedMixtureOfExperts(primary_gating_hp, secondary_gating_hp,
                                         expert_hp, model_hidden_size,
                                         model_hidden_size, ps_devices, "moe")
    # MoE expects input tensors to be 2d.
    #  Flatten out spatial dimensions.
    xs_2d = dp(tf.reshape, xs, [[-1, model_hidden_size]] * dp.n)
    # Call the MoE
    moe_out_2d, importance, load, _, _ = moe.Eval(
        dp.devices, xs_2d, train, identifiers=None, summaries=True)
    # Reshape the output to the original shape.
    moe_out = dp(tf.reshape, moe_out_2d, dp(tf.shape, xs))
    # These losses encourage equal load on the different experts.
    loss = loss_coef * (eu.CVSquared(importance) + eu.CVSquared(load))
    return moe_out, loss


def simple_attention(target, source, bias=None, summaries=True):
  """A simple attention function.

  Args:
    target: a `Tensor` with shape `[batch, target_timesteps, depth]` or
     `[batch, target_timesteps_1, target_timesteps_2, depth]`
    source: a `Tensor` with shape `[batch, source_timesteps, depth]` or
     `[batch, source_timesteps_1, source_timesteps_2, depth]`
    bias: an optional `Tensor` with shape `[batch, timesteps, 1, 1]` used
     to mask the attention to not attend to padding of input.
    summaries: Boolean, whether to output summaries.

  Returns:
    a `Tensor` with same shape as `target`
  """
  with tf.name_scope("simple_attention", [target, source]):
    target_shape = tf.shape(target)
    source_shape = tf.shape(source)
    target = tf.reshape(target, [
        target_shape[0], target_shape[1] * target_shape[2], target_shape[3]
    ])
    source = tf.reshape(source, [
        source_shape[0], source_shape[1] * source_shape[2], source_shape[3]
    ])
    attention = tf.matmul(target, source, transpose_b=True)
    attention *= tf.rsqrt(tf.to_float(tf.shape(target)[2]))
    if bias is not None:
      attention += tf.expand_dims(tf.squeeze(bias, axis=[2, 3]), axis=1)
    attention = tf.nn.softmax(attention)
    if summaries and not tf.get_variable_scope().reuse:
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
      if dilation_rate > 1:
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


def multiscale_conv_and_attention(x,
                                  padding,
                                  hparams,
                                  source=None,
                                  summaries=True):
  """A common part of t2t layers.

  First, do a linear multiscale convolution
  Second, do attention (if source is not None)

  Applies residuals and normalization on both steps.

  Args:
    x: a Tensor.
    padding: a padding type
    hparams: hyperparameters for model
    source: optional source tensor for attention. (encoder output)
    summaries: Boolean, whether to output summaries.

  Returns:
    a Tensor.
  """
  # TODO(noam): The number of different scales should be a hyperparameter.
  conv_sum = multiscale_conv_sum(
      x,
      hparams.hidden_size, [((hparams.kernel_height**i, hparams.kernel_width**
                              i), (hparams.kernel_height, hparams.kernel_width))
                            for i in xrange(3)],
      "AVG",
      padding=padding)
  # For residuals a rescale if necessary if channels differ.
  if x.get_shape().as_list()[-1] != conv_sum.get_shape().as_list()[-1]:
    x = conv(x, hparams.hidden_size, (1, 1))
  x = noam_norm(x + conv_sum)
  if source is not None:
    x = noam_norm(x + simple_attention(x, source, summaries=summaries))
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


def conv_with_pools_and_attention(x,
                                  padding,
                                  hparams,
                                  source=None,
                                  summaries=True):
  """A common part of t2t layers.

  First, do conv_with_pools
  Second, do attention (if source is not None)

  Applies residuals and normalization on both steps.

  Args:
    x: a Tensor.
    padding: a padding type
    hparams: hyperparameters for model
    source: optional source tensor for attention. (encoder output)
    summaries: Boolean, whether to output summaries.

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
    x = noam_norm(x + simple_attention(x, source, summaries=summaries))
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
  log_timescale_increment = (math.log(max_timescale / min_timescale) /
                             (num_timescales - 1))
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
  length = tf.shape(x)[1]
  depth = tf.shape(x)[3]
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
  return tf.expand_dims(
      tf.matrix_band_part(tf.ones([target_length, source_length]), -1, 0), 0)


def attention_1d_v0(source,
                    target,
                    attention_size,
                    output_size,
                    num_heads,
                    mask=None,
                    transform_source=True,
                    transform_target=True,
                    transform_output=True,
                    summaries=True,
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
    summaries: a boolean
    name: an optional string

  Returns:
    a Tensor of shape [batch, length, output_size]
  """
  with tf.variable_scope(name, default_name="attention", values=[target]):
    source_length = tf.shape(source)[1]
    target_length = tf.shape(target)[1]
    batch = tf.shape(source)[0]

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
    if summaries and not tf.get_variable_scope().reuse:
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


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     summaries=True,
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
    h = conv(
        inputs,
        hidden_size,
        kernel_size,
        activation=tf.nn.relu,
        name="conv1",
        **kwargs)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    if summaries and not tf.get_variable_scope().reuse:
      tf.summary.histogram("hidden_density_logit",
                           relu_density_logit(
                               h, list(range(inputs.shape.ndims - 1))))
    ret = conv(h, output_size, (1, 1), name="conv2", **kwargs)
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
                      train,
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

    # Dropout if training.
    if dropout > 0.0 and train:
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
  with tf.name_scope("pad_to_same_length", [x, y]):
    x_length = tf.shape(x)[axis]
    y_length = tf.shape(y)[axis]
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
  with tf.name_scope("pad_with_zeros", [logits, labels]):
    logits, labels = pad_to_same_length(logits, labels)
    if len(labels.shape.as_list()) == 3:  # 2-d labels.
      logits, labels = pad_to_same_length(logits, labels, axis=2)
    return labels


def weights_nonzero(labels):
  """Assign weight 1.0 to all labels except for padding (id=0)."""
  return tf.to_float(tf.not_equal(labels, 0))


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
  shifted = tf.pad(sentence_num_plus_one, [[0, 0], [2, 0], [0, 0],
                                           [0, 0]])[:, :-2, :, :]
  nonboilerplate = tf.equal(sentence_num_plus_one, shifted)
  ret = tf.to_float(tf.logical_and(nonboilerplate, in_target))
  return ret


def padded_cross_entropy(logits,
                         labels,
                         label_smoothing,
                         weights_fn=weights_nonzero,
                         reduce_sum=True):
  """Compute cross-entropy assuming 0s are padding.

  Computes a loss numerator (the sum of losses), and loss denominator
  (the number of non-padding tokens).

  Args:
    logits: a `Tensor` with shape `[batch, timesteps, vocab_size]`.
    labels: an integer `Tensor` with shape `[batch, timesteps]`.
    label_smoothing: a floating point `Scalar`.
    weights_fn: A function from labels to weights.
    reduce_sum: a Boolean, whether to sum at the end or not.

  Returns:
    loss_numerator: a `Scalar`.  Sum of losses.
    loss_denominator: a `Scalar.  The number of non-padding target tokens.
  """
  confidence = 1.0 - label_smoothing
  vocab_size = tf.shape(logits)[-1]
  with tf.name_scope("padded_cross_entropy", [logits, labels]):
    pad_labels = pad_with_zeros(logits, labels)
    xent = smoothing_cross_entropy(logits, pad_labels, vocab_size, confidence)
    weights = weights_fn(pad_labels)
    if not reduce_sum:
      return xent * weights, weights
    return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)


def smoothing_cross_entropy(logits, labels, vocab_size, confidence):
  """Cross entropy with label smoothing to limit over-confidence."""
  with tf.name_scope("smoothing_cross_entropy", [logits, labels]):
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(confidence * tf.log(confidence) + tf.to_float(
        vocab_size - 1) * low_confidence * tf.log(low_confidence + 1e-20))
    # Soft targets.
    soft_targets = tf.one_hot(
        tf.cast(labels, tf.int32),
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing
