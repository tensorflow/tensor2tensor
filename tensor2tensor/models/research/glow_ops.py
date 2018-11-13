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

"""Various reversible ops for the glow generative model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import numpy as np
import scipy
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
import tensorflow as tf
import tensorflow_probability as tfp

arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope


def default_initializer(std=0.05):
  return tf.random_normal_initializer(0., std)


def get_eps(dist, x):
  """Z = (X - mu) / sigma."""
  return (x - dist.loc) / dist.scale


def set_eps(dist, eps):
  """Z = eps * sigma + mu."""
  return eps * dist.scale + dist.loc


@add_arg_scope
def assign(w, initial_value):
  w = w.assign(initial_value)
  with tf.control_dependencies([w]):
    return w


def get_cond_latents_at_level(cond_latents, level, hparams):
  """Returns a single or list of conditional latents at level 'level'."""
  if cond_latents:
    if hparams.latent_dist_encoder in ["conv_net", "conv3d_net"]:
      return [cond_latent[level] for cond_latent in cond_latents]
    elif hparams.latent_dist_encoder in ["pointwise", "conv_lstm"]:
      return cond_latents[level]


def check_cond_latents(cond_latents, hparams):
  """Shape checking for cond_latents."""
  if cond_latents is None:
    return
  if not isinstance(cond_latents[0], list):
    cond_latents = [cond_latents]
  exp_num_latents = hparams.num_cond_latents
  if hparams.latent_dist_encoder == "conv_net":
    exp_num_latents += int(hparams.cond_first_frame)
  if len(cond_latents) != exp_num_latents:
    raise ValueError("Expected number of cond_latents: %d, got %d" %
                     (exp_num_latents, len(cond_latents)))
  for cond_latent in cond_latents:
    if len(cond_latent) != hparams.n_levels - 1:
      raise ValueError("Expected level_latents to be %d, got %d" %
                       (hparams.n_levels - 1, len(cond_latent)))


@add_arg_scope
def get_variable_ddi(name, shape, initial_value, dtype=tf.float32, init=False,
                     trainable=True):
  """Wrapper for data-dependent initialization."""
  # If init is a tensor bool, w is returned dynamically.
  w = tf.get_variable(name, shape, dtype, None, trainable=trainable)
  if isinstance(init, bool):
    if init:
      return assign(w, initial_value)
    return w
  else:
    return tf.cond(init, lambda: assign(w, initial_value), lambda: w)


@add_arg_scope
def actnorm_3d(name, x, logscale_factor=3.):
  """Applies actnorm to each time-step independently.

  There are a total of 2*n_channels*n_steps parameters learnt.

  Args:
    name: variable scope.
    x: 5-D Tensor, (NTHWC)
    logscale_factor: Increases the learning rate of the scale by
                     logscale_factor.
  Returns:
    x: 5-D Tensor, (NTHWC) with the per-timestep, per-channel normalization.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x = tf.unstack(x, axis=1)
    x_normed = []
    for ind, x_step in enumerate(x):
      x_step, _ = actnorm("actnorm_%d" % ind, x_step,
                          logscale_factor=logscale_factor)
      x_normed.append(x_step)
    return tf.stack(x_normed, axis=1), None


@add_arg_scope
def actnorm(name, x, logscale_factor=3., reverse=False, init=False,
            trainable=True):
  """x_{ij} = s x x_{ij} + b. Per-channel scaling and bias.

  If init is set to True, the scaling and bias are initialized such
  that the mean and variance of the output activations of the first minibatch
  are zero and one respectively.

  Args:
    name: variable scope.
    x: input
    logscale_factor: Used in actnorm_scale. Optimizes f(ls*s') instead of f(s)
                     where s' = s / ls. Helps in faster convergence.
    reverse: forward or reverse operation.
    init: Whether or not to do data-dependent initialization.
    trainable:

  Returns:
    x: output after adding bias and scaling.
    objective: log(sum(s))
  """
  var_arg_scope = arg_scope([get_variable_ddi], trainable=trainable)
  var_scope = tf.variable_scope(name, reuse=tf.AUTO_REUSE)

  with var_scope, var_arg_scope:
    if not reverse:
      x = actnorm_center(name + "_center", x, reverse, init=init)
      x, objective = actnorm_scale(
          name + "_scale", x, logscale_factor=logscale_factor,
          reverse=reverse, init=init)
    else:
      x, objective = actnorm_scale(
          name + "_scale", x, logscale_factor=logscale_factor,
          reverse=reverse, init=init)
      x = actnorm_center(name + "_center", x, reverse, init=init)
    return x, objective


@add_arg_scope
def actnorm_center(name, x, reverse=False, init=False):
  """Add a bias to x.

  Initialize such that the output of the first minibatch is zero centered
  per channel.

  Args:
    name: scope
    x: 2-D or 4-D Tensor.
    reverse: Forward or backward operation.
    init: data-dependent initialization.

  Returns:
    x_center: (x + b), if reverse is True and (x - b) otherwise.
  """
  shape = common_layers.shape_list(x)
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    assert len(shape) == 2 or len(shape) == 4
    if len(shape) == 2:
      x_mean = tf.reduce_mean(x, [0], keepdims=True)
      b = get_variable_ddi("b", (1, shape[1]), initial_value=-x_mean,
                           init=init)
    elif len(shape) == 4:
      x_mean = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
      b = get_variable_ddi(
          "b", (1, 1, 1, shape[3]), initial_value=-x_mean, init=init)

    if not reverse:
      x += b
    else:
      x -= b
    return x


@add_arg_scope
def actnorm_scale(name, x, logscale_factor=3., reverse=False, init=False):
  """Per-channel scaling of x."""
  x_shape = common_layers.shape_list(x)
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

    # Variance initialization logic.
    assert len(x_shape) == 2 or len(x_shape) == 4
    if len(x_shape) == 2:
      x_var = tf.reduce_mean(x**2, [0], keepdims=True)
      logdet_factor = 1
      var_shape = (1, x_shape[1])
    elif len(x_shape) == 4:
      x_var = tf.reduce_mean(x**2, [0, 1, 2], keepdims=True)
      logdet_factor = x_shape[1]*x_shape[2]
      var_shape = (1, 1, 1, x_shape[3])

    init_value = tf.log(1.0 / (tf.sqrt(x_var) + 1e-6)) / logscale_factor
    logs = get_variable_ddi("logs", var_shape, initial_value=init_value,
                            init=init)
    logs = logs * logscale_factor

    # Function and reverse function.
    if not reverse:
      x = x * tf.exp(logs)
    else:
      x = x * tf.exp(-logs)

    # Objective calculation, h * w * sum(log|s|)
    dlogdet = tf.reduce_sum(logs) * logdet_factor
    if reverse:
      dlogdet *= -1
    return x, dlogdet


@add_arg_scope
def invertible_1x1_conv(name, x, reverse=False):
  """1X1 convolution on x.

  The 1X1 convolution is parametrized as P*L*(U + sign(s)*exp(log(s))) where
  1. P is a permutation matrix.
  2. L is a lower triangular matrix with diagonal entries unity.
  3. U is a upper triangular matrix where the diagonal entries zero.
  4. s is a vector.

  sign(s) and P are fixed and the remaining are optimized. P, L, U and s are
  initialized by the PLU decomposition of a random rotation matrix.

  Args:
    name: scope
    x: Input Tensor.
    reverse: whether the pass is from z -> x or x -> z.

  Returns:
    x_conv: x after a 1X1 convolution is applied on x.
    objective: sum(log(s))
  """
  _, height, width, channels = common_layers.shape_list(x)
  w_shape = [channels, channels]

  # Random rotation-matrix Q
  random_matrix = np.random.rand(channels, channels)
  np_w = scipy.linalg.qr(random_matrix)[0].astype("float32")

  # Initialize P,L,U and s from the LU decomposition of a random rotation matrix
  np_p, np_l, np_u = scipy.linalg.lu(np_w)
  np_s = np.diag(np_u)
  np_sign_s = np.sign(np_s)
  np_log_s = np.log(np.abs(np_s))
  np_u = np.triu(np_u, k=1)

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    p = tf.get_variable("P", initializer=np_p, trainable=False)
    l = tf.get_variable("L", initializer=np_l)
    sign_s = tf.get_variable(
        "sign_S", initializer=np_sign_s, trainable=False)
    log_s = tf.get_variable("log_S", initializer=np_log_s)
    u = tf.get_variable("U", initializer=np_u)

    # W = P * L * (U + sign_s * exp(log_s))
    l_mask = np.tril(np.ones([channels, channels], dtype=np.float32), -1)
    l = l * l_mask + tf.eye(channels, channels)
    u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
    w = tf.matmul(p, tf.matmul(l, u))

    objective = tf.reduce_sum(log_s) * height * width
    if not reverse:
      w = tf.reshape(w, [1, 1] + w_shape)
      x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format="NHWC")
    else:
      u_inv = tf.matrix_inverse(u)
      l_inv = tf.matrix_inverse(l)
      p_inv = tf.matrix_inverse(p)
      w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
      w_inv = tf.reshape(w_inv, [1, 1]+w_shape)
      x = tf.nn.conv2d(
          x, w_inv, [1, 1, 1, 1], "SAME", data_format="NHWC")
      objective *= -1
  return x, objective


def add_edge_bias(x, filter_size):
  """Pad x and concatenates an edge bias across the depth of x.

  The edge bias can be thought of as a binary feature which is unity when
  the filter is being convolved over an edge and zero otherwise.

  Args:
    x: Input tensor, shape (NHWC)
    filter_size: filter_size to determine padding.
  Returns:
    x_pad: Input tensor, shape (NHW(c+1))
  """
  x_shape = common_layers.shape_list(x)
  if filter_size[0] == 1 and filter_size[1] == 1:
    return x
  a = (filter_size[0] - 1) // 2  # vertical padding size
  b = (filter_size[1] - 1) // 2  # horizontal padding size
  padding = [[0, 0], [a, a], [b, b], [0, 0]]
  x_bias = tf.zeros(x_shape[:-1] + [1])

  x = tf.pad(x, padding)
  x_pad = tf.pad(x_bias, padding, constant_values=1)
  return tf.concat([x, x_pad], axis=3)


def time_pad(x, filter_size):
  """Pad left across time and pad valid across the spatial components.

  Args:
    x: 5-D Tensor, (NTHWC)
    filter_size: list of ints
  Returns:
    x_pad: 5-D Tensor.
  """
  if filter_size == [1, 1, 1]:
    return x
  a = (filter_size[1] - 1) // 2  # vertical padding size
  b = (filter_size[2] - 1) // 2  # horizontal padding size
  c = filter_size[0] - 1
  padding = [[0, 0], [c, 0], [a, a], [b, b], [0, 0]]
  return tf.pad(x, padding)


@add_arg_scope
def conv(name, x, output_channels, filter_size=None, stride=None,
         logscale_factor=3.0, apply_actnorm=True, conv_init="default"):
  """Convolutional layer with edge bias padding and optional actnorm.

  If x is 5-dimensional, actnorm is applied independently across every
  time-step.

  Args:
    name: variable scope.
    x: 4-D Tensor or 5-D Tensor of shape NHWC or NTHWC
    output_channels: Number of output channels.
    filter_size: list of ints, if None [3, 3] and [2, 3, 3] are defaults for
                 4-D and 5-D input tensors respectively.
    stride: list of ints, default stride: 1
    logscale_factor: see actnorm for parameter meaning.
    apply_actnorm: if apply_actnorm the activations of the first minibatch
                   have zero mean and unit variance. Else, there is no scaling
                   applied.
    conv_init: default or zeros. default is a normal distribution with 0.05 std.
  Returns:
    x: actnorm(conv2d(x))
  Raises:
    ValueError: if init is set to "zeros" and apply_actnorm is set to True.
  """
  if conv_init == "zeros" and apply_actnorm:
    raise ValueError("apply_actnorm is unstable when init is set to zeros.")

  x_shape = common_layers.shape_list(x)
  is_2d = len(x_shape) == 4

  # set filter_size, stride and in_channels
  if is_2d:
    if filter_size is None:
      filter_size = [3, 3]
    if stride is None:
      stride = [1, 1]
    actnorm_func = actnorm
    x = add_edge_bias(x, filter_size=filter_size)
    conv_filter = tf.nn.conv2d
  else:
    if filter_size is None:
      filter_size = [2, 3, 3]
    if stride is None:
      stride = [1, 1, 1]
    actnorm_func = actnorm_3d
    x = time_pad(x, filter_size=filter_size)
    conv_filter = tf.nn.conv3d

  in_channels = common_layers.shape_list(x)[-1]
  filter_shape = filter_size + [in_channels, output_channels]
  stride_shape = [1] + stride + [1]

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

    if conv_init == "default":
      initializer = default_initializer()
    elif conv_init == "zeros":
      initializer = tf.zeros_initializer()

    w = tf.get_variable("W", filter_shape, tf.float32, initializer=initializer)
    x = conv_filter(x, w, stride_shape, padding="VALID")
    if apply_actnorm:
      x, _ = actnorm_func("actnorm", x, logscale_factor=logscale_factor)
    else:
      x += tf.get_variable("b", [1, 1, 1, output_channels],
                           initializer=tf.zeros_initializer())
      logs = tf.get_variable("logs", [1, output_channels],
                             initializer=tf.zeros_initializer())
      x *= tf.exp(logs * logscale_factor)
    return x


@add_arg_scope
def conv_block(name, x, mid_channels, time_filter=2):
  """2 layer conv block used in the affine coupling layer.

  Args:
    name: variable scope.
    x: 4-D or 5-D Tensor.
    mid_channels: Output channels of the second layer.
    time_filter: Filter across time to capture context.
  Returns:
    x: 4-D Tensor: Output activations.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

    x_shape = common_layers.shape_list(x)
    is_2d = len(x_shape) == 4
    if is_2d:
      first_filter = [3, 3]
      second_filter = [1, 1]
    else:
      first_filter = [time_filter, 3, 3]
      second_filter = [1, 1, 1]

    # Edge Padding + conv2d + actnorm + relu:
    # [output: 512 channels]
    x = conv("1_1", x, output_channels=mid_channels, filter_size=first_filter)
    x = tf.nn.relu(x)

    # Padding + conv2d + actnorm + relu
    # [input, output: 512 channels]
    x = conv("1_2", x, output_channels=mid_channels, filter_size=second_filter)
    x = tf.nn.relu(x)
    return x


@add_arg_scope
def affine_coupling_network(name, x, mid_channels, output_channels):
  """3-layer conv2d.

  Args:
    name:
    x:
    mid_channels: Number of output channels of the first layer.
    output_channels: Number of output channels.

  Returns:
    output:
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

    x = conv_block("conv_block", x, mid_channels=mid_channels)

    # Final layer.
    x = conv("zeros", x, filter_size=[3, 3], stride=[1, 1],
             output_channels=output_channels, apply_actnorm=False,
             conv_init="zeros")
  return x


@add_arg_scope
def affine_coupling(name, x, mid_channels=512, reverse=False):
  """Reversible affine coupling layer.

  Args:
    name:
    x:
    mid_channels: intermediate
    reverse: Forward or reverse operation.
  Returns:
    output:
    objective:
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x_shape = common_layers.shape_list(x)
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

    # scale, shift = NN(x1)
    # If reverse:
    # z2 = scale * (x2 + shift)
    # Else:
    # z2 = (x2 / scale) - shift
    z1 = x1
    log_scale_and_shift = affine_coupling_network(
        "nn", x1, mid_channels, x_shape[-1])
    shift = log_scale_and_shift[:, :, :, 0::2]
    scale = tf.nn.sigmoid(log_scale_and_shift[:, :, :, 1::2] + 2.0)
    if not reverse:
      z2 = (x2 + shift) * scale
    else:
      z2 = x2 / scale - shift

    objective = tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
    if reverse:
      objective *= -1
    return tf.concat([z1, z2], axis=3), objective


@add_arg_scope
def squeeze(name, x, factor=2, reverse=True):
  """Block-wise spatial squeezing of x to increase the number of channels.

  Args:
    name: Used for variable scoping.
    x: 4-D Tensor of shape (batch_size X H X W X C)
    factor: Factor by which the spatial dimensions should be squeezed.
    reverse: Squueze or unsqueeze operation.

  Returns:
    x: 4-D Tensor of shape (batch_size X (H//factor) X (W//factor) X
       (cXfactor^2). If reverse is True, then it is factor = (1 / factor)
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    shape = common_layers.shape_list(x)
    if factor == 1:
      return x
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])

    if not reverse:
      assert height % factor == 0 and width % factor == 0
      x = tf.reshape(x, [-1, height//factor, factor,
                         width//factor, factor, n_channels])
      x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
      x = tf.reshape(x, [-1, height//factor, width //
                         factor, n_channels*factor*factor])
    else:
      x = tf.reshape(
          x, (-1, height, width, int(n_channels/factor**2), factor, factor))
      x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
      x = tf.reshape(x, (-1, int(height*factor),
                         int(width*factor), int(n_channels/factor**2)))
    return x


@add_arg_scope
def temporal_latent_to_dist(name, x, hparams, output_channels=None):
  """Network that maps a time-indexed list of 3-D latents to a gaussian.

  Args:
    name: variable scope.
    x: List of 4-D Tensors indexed by time, (NHWC)
    hparams: tf.contrib.training.Hparams.
    output_channels: int, Number of channels of the output gaussian mean.
  Returns:
    dist: tfp.distributions.Normal
  """
  res_channels = common_layers.shape_list(x)[-1]
  if output_channels is None:
    output_channels = res_channels
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    h = x
    for i in range(hparams.latent_encoder_depth):
      h1 = conv_block("conv3d_1_%d" % i, h, time_filter=2,
                      mid_channels=hparams.latent_encoder_width)
      h2 = conv("conv3d_zeros_%d" % i, h1, apply_actnorm=False,
                output_channels=res_channels, conv_init="zeros",
                filter_size=[2, 3, 3])
      h += h2

    # take last activation that should capture all context since padding is
    # on left.
    h = h[:, -1, :, :, :]
    h = conv("res_final", h, apply_actnorm=False, conv_init="zeros",
             output_channels=2*output_channels, filter_size=[1, 1])
    mean, log_scale = h[:, :, :, 0::2], h[:, :, :, 1::2]
  return tfp.distributions.Normal(mean, tf.exp(log_scale))


@add_arg_scope
def single_conv_dist(name, x, output_channels=None):
  """A 3x3 convolution mapping x to a standard normal distribution at init.

  Args:
    name: variable scope.
    x: 4-D Tensor.
    output_channels: number of channels of the mean and std.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x_shape = common_layers.shape_list(x)
    if output_channels is None:
      output_channels = x_shape[-1]
    mean_log_scale = conv("conv2d", x, output_channels=2*output_channels,
                          conv_init="zeros", apply_actnorm=False)
    mean = mean_log_scale[:, :, :, 0::2]
    log_scale = mean_log_scale[:, :, :, 1::2]
    return tf.distributions.Normal(mean, tf.exp(log_scale))


@add_arg_scope
def latent_to_dist(name, x, hparams, output_channels=None):
  """Map latent to the mean and log-scale of a Gaussian.

  Args:
    name: variable scope.
    x: 4-D Tensor of shape (NHWC)
    hparams: tf.contrib.training.HParams.
      latent_architecture - can be "single_conv", "glow_nn" or "glow_resnet",
                            default = single_conv
      latent_encoder_depth - int, depth of architecture, valid if
                             latent_architecture is "glow_nn" or "glow_resnet".
      latent_pre_output_channels - 512, valid only when latent_architecture
                                   is "glow_nn".
      latent_encoder_width - 512, maximum width of the network
    output_channels: int, number of output channels of the mean (and std).
                     if not provided, set it to be the output channels of x.
  Returns:
    dist: instance of tfp.distributions.Normal
  Raises:
    ValueError: If architecture not in ["single_conv", "glow_nn"]
  """
  architecture = hparams.get("latent_architecture", "single_conv")
  depth = hparams.get("latent_encoder_depth", 1)
  pre_output_channels = hparams.get("latent_pre_output_channels", 512)
  width = hparams.get("latent_encoder_width", 512)

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    x_shape = common_layers.shape_list(x)
    if output_channels is None:
      output_channels = x_shape[-1]
    if architecture == "single_conv":
      return single_conv_dist("single_conv", x, output_channels)
    if architecture == "glow_nn":
      mean_log_scale = x
      for layer in range(1, depth + 1):
        mid_channels = pre_output_channels // 2**(depth - layer)
        mean_log_scale = conv_block("glow_nn_%d" % layer, mean_log_scale,
                                    mid_channels=mid_channels)
      mean_log_scale = conv("glow_nn_zeros", mean_log_scale,
                            filter_size=[3, 3], stride=[1, 1],
                            output_channels=2*output_channels,
                            apply_actnorm=False, conv_init="zeros")
    elif architecture == "glow_resnet":
      h = x
      for layer in range(depth):
        h2 = conv_block("glow_res_%d" % layer, h, mid_channels=width)
        h3 = conv("glow_res_zeros_%d" % layer, h2, conv_init="zeros",
                  output_channels=x_shape[-1], apply_actnorm=False)
        h += h3
      mean_log_scale = conv("glow_res_final", h, conv_init="zeros",
                            output_channels=2*output_channels,
                            apply_actnorm=False)
    else:
      raise ValueError("expected architecture to be single_conv or glow_nn "
                       "got %s" % architecture)

    mean = mean_log_scale[:, :, :, 0::2]
    log_scale = mean_log_scale[:, :, :, 1::2]
    return tfp.distributions.Normal(mean, tf.exp(log_scale))


@add_arg_scope
def merge_level_and_latent_dist(level_dist, latent_dist,
                                merge_std="prev_level"):
  """Merge level_dist and latent_dist.

  new_dist ~ N(level_dist.mean + latent_dis.mean, std) where std is determined
  according to merge_std.

  Args:
    level_dist: instance of tfp.distributions.Normal
    latent_dist: instance of tfp.distributions.Normal
    merge_std: can be "prev_level", "prev_step" or "normal".
  Returns:
    merged_dist: instance of tfp.distributions.Normal
  """
  level_mean, level_std = level_dist.loc, level_dist.scale
  latent_mean, latent_std = latent_dist.loc, latent_dist.scale
  new_mean = level_mean + latent_mean
  if merge_std == "normal":
    z_shape = common_layers.shape_list(latent_mean)
    log_scale = tf.get_variable(
        "merge_std", shape=z_shape, dtype=tf.float32,
        initializer=tf.zeros_initializer(), trainable=False)
    scale = tf.exp(log_scale * 3.0)
  elif merge_std == "prev_level":
    scale = level_std
  elif merge_std == "prev_step":
    scale = latent_std
  return tfp.distributions.Normal(loc=new_mean, scale=scale)


@add_arg_scope
def level_cond_prior(prior_dist, z, latent, hparams, state):
  """Returns a conditional prior for each level.

  Args:
    prior_dist: Distribution conditioned on the previous levels.
    z: Tensor, output of the previous levels.
    latent: Tensor or a list of tensors to condition the latent_distribution.
    hparams: next_frame_glow hparams.
    state: Current LSTM state. Used only if hparams.latent_dist_encoder is
           a lstm.
  Raises:
    ValueError: If hparams.latent_dist_encoder is "pointwise" and if the shape
                of latent is different from z.
  """
  latent_dist_encoder = hparams.get("latent_dist_encoder", None)
  latent_skip = hparams.get("latent_skip", False)
  if latent_dist_encoder == "pointwise":
    last_latent = latent
    merge_std = hparams.level_scale
    latent_shape = common_layers.shape_list(latent)
    z_shape = common_layers.shape_list(z)
    if latent_shape != z_shape:
      raise ValueError("Expected latent_shape to be %s, got %s" %
                       (latent_shape, z_shape))
    latent_dist = scale_gaussian_prior(
        "latent_prior", latent, logscale_factor=3.0)
    cond_dist = merge_level_and_latent_dist(prior_dist, latent_dist,
                                            merge_std=merge_std)

  elif latent_dist_encoder == "conv_net":
    output_channels = common_layers.shape_list(z)[-1]
    last_latent = latent[-1]
    latent_stack = tf.concat([prior_dist.loc] + latent, axis=-1)
    cond_dist = latent_to_dist(
        "latent_stack", latent_stack, hparams=hparams,
        output_channels=output_channels)

  elif latent_dist_encoder == "conv3d_net":
    last_latent = latent[-1]
    output_channels = common_layers.shape_list(last_latent)[-1]
    num_steps = len(latent)

    # Stack across time.
    cond_latents = tf.stack(latent, axis=1)

    # Concat latents from previous levels across channels.
    prev_latents = tf.tile(tf.expand_dims(prior_dist.loc, axis=1),
                           [1, num_steps, 1, 1, 1])
    cond_latents = tf.concat((cond_latents, prev_latents), axis=-1)
    cond_dist = temporal_latent_to_dist(
        "latent_stack", cond_latents, hparams, output_channels=output_channels)

  elif latent_dist_encoder == "conv_lstm":
    last_latent = latent
    output_channels = common_layers.shape_list(z)[-1]
    latent_stack = tf.concat((prior_dist.loc, latent), axis=-1)
    _, state = common_video.conv_lstm_2d(
        latent_stack, state, hparams.latent_encoder_width, kernel_size=3,
        name="conv_lstm")

    cond_dist = single_conv_dist(
        "state_to_dist", state.h, output_channels=output_channels)
  if latent_skip:
    new_mean = cond_dist.loc + last_latent
    cond_dist = tfp.distributions.Normal(new_mean, cond_dist.scale)
  return cond_dist.loc, cond_dist.scale, state


@add_arg_scope
def compute_prior(name, z, latent, hparams, condition=False, state=None):
  """Distribution on z_t conditioned on z_{t-1} and latent.

  Args:
    name: variable scope.
    z: 4-D Tensor.
    latent: optional,
            if hparams.latent_dist_encoder == "pointwise", this is a list
            of 4-D Tensors of length hparams.num_cond_latents.
            else, this is just a 4-D Tensor
            The first-three dimensions of the latent should be the same as z.
    hparams: next_frame_glow_hparams.
    condition: Whether or not to condition the distribution on latent.
    state: tf.contrib.rnn.LSTMStateTuple.
           the current state of a LSTM used to model the distribution. Used
           only if hparams.latent_dist_encoder = "conv_lstm".
  Returns:
    prior_dist: instance of tfp.distributions.Normal
    state: Returns updated state.
  Raises:
    ValueError: If hparams.latent_dist_encoder is "pointwise" and if the shape
                of latent is different from z.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    if isinstance(condition, bool):
      condition = tf.constant(condition, dtype=tf.bool)
    prior_dist = single_conv_dist("level_prior", z)
    prior_mean, prior_scale = prior_dist.loc, prior_dist.scale

    if latent is None:
      mean, scale = prior_mean, prior_scale
    else:
      cond_mean, cond_scale, state = level_cond_prior(
          prior_dist, z, latent, hparams, state)
      mean, scale = tf.cond(
          condition, lambda: (cond_mean, cond_scale),
          lambda: (prior_mean, prior_scale))
    dist = tfp.distributions.Normal(mean, scale)
    return dist, state


@add_arg_scope
def split(name, x, reverse=False, eps=None, eps_std=None, cond_latents=None,
          hparams=None, state=None, condition=False):
  """Splits / concatenates x into x1 and x2 across number of channels.

  For the forward pass, x2 is assumed be gaussian,
  i.e P(x2 | x1) ~ N(mu, sigma) where mu and sigma are the outputs of
  a network conditioned on x1 and optionally on cond_latents.
  For the reverse pass, x2 is determined from mu(x1) and sigma(x1).
  This is deterministic/stochastic depending on whether eps is provided.

  Args:
    name: variable scope.
    x: 4-D Tensor, shape (NHWC).
    reverse: Forward or reverse pass.
    eps: If eps is provided, x2 is set to be
    eps_std: Sample x2 with the provided eps_std.
    cond_latents: optionally condition x2 on cond_latents.
    hparams: next_frame_glow hparams.
    state: tf.contrib.rnn.LSTMStateTuple. Current state of the LSTM over z_2.
           Used only when hparams.latent_dist_encoder == "conv_lstm"
    condition: bool, Whether or not to condition the distribution on
               cond_latents.

  Returns:
  Raises:
    ValueError: If latent is provided and shape is not equal to NHW(C/2)
                where (NHWC) is the size of x.
  """
  # TODO(mechcoder) Change the return type to be a dict.
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    if not reverse:
      x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)

      # objective: P(x2|x1) ~N(x2 ; NN(x1))
      prior_dist, state = compute_prior(
          "prior_on_z2", x1, cond_latents, hparams, condition, state=state)
      logpb = tf.reduce_sum(prior_dist.log_prob(x2), axis=[1, 2, 3])
      eps = get_eps(prior_dist, x2)
      return x1, logpb, eps, x2, state
    else:
      prior_dist, state = compute_prior(
          "prior_on_z2", x, cond_latents, hparams, condition, state=state)
      if eps is not None:
        x2 = set_eps(prior_dist, eps)
      elif eps_std is not None:
        x2 = eps_std * tf.random_normal(common_layers.shape_list(x))
      else:
        x2 = prior_dist.sample()
      return tf.concat([x, x2], 3), x2, state


@add_arg_scope
def revnet_step(name, x, hparams, reverse=True):
  """One step of glow generative flow.

  Actnorm + invertible 1X1 conv + affine_coupling.

  Args:
    name: used for variable scope.
    x: input
    hparams: affine_coupling_width is the only hparam that is being used in
             this function.
    reverse: forward or reverse pass.
  Returns:
    z: Output of one step of reversible flow.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    ops = [
        partial(actnorm, name="actnorm", reverse=reverse),
        partial(invertible_1x1_conv, name="invertible", reverse=reverse),
        partial(affine_coupling, name="affine", reverse=reverse,
                mid_channels=hparams.affine_coupling_width)]

    if reverse:
      ops = ops[::-1]

    objective = 0.0
    for op in ops:
      x, curr_obj = op(x=x)
      objective += curr_obj
    return x, objective


def revnet(name, x, hparams, reverse=True):
  """'hparams.depth' steps of generative flow."""
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    steps = np.arange(hparams.depth)
    if reverse:
      steps = steps[::-1]

    objective = 0.0
    for step in steps:
      x, curr_obj = revnet_step(
          "revnet_step_%d" % step, x, hparams, reverse=reverse)
      objective += curr_obj
    return x, objective


@add_arg_scope
def scale_gaussian_prior(name, z, logscale_factor=3.0, trainable=True):
  """Returns N(s^i * z^i, std^i) where s^i and std^i are pre-component.

  s^i is a learnable parameter with identity initialization.
  std^i is optionally learnable with identity initialization.

  Args:
    name: variable scope.
    z: input_tensor
    logscale_factor: equivalent to scaling up the learning_rate by a factor
                     of logscale_factor.
    trainable: Whether or not std^i is learnt.
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    z_shape = common_layers.shape_list(z)
    latent_multiplier = tf.get_variable(
        "latent_multiplier", shape=z_shape, dtype=tf.float32,
        initializer=tf.ones_initializer())
    log_scale = tf.get_variable(
        "log_scale_latent", shape=z_shape, dtype=tf.float32,
        initializer=tf.zeros_initializer(), trainable=trainable)
    log_scale = log_scale * logscale_factor
    return tfp.distributions.Normal(
        loc=latent_multiplier * z, scale=tf.exp(log_scale))


@add_arg_scope
def top_prior(name, z_shape, learn_prior="normal"):
  """Unconditional prior distribution.

  Args:
    name: variable scope
    z_shape: Shape of the mean / scale of the prior distribution.
    learn_prior: Possible options are "normal" and "single_conv".
                 If set to "single_conv", the gaussian is parametrized by a
                 single convolutional layer whose input are an array of zeros
                 and initialized such that the mean and std are zero and one.
                 If set to "normal", the prior is just a Gaussian with zero
                 mean and unit variance.
  Returns:
    objective: 1-D Tensor shape=(batch_size,) summed across spatial components.
  Raises:
    ValueError: If learn_prior not in "normal" or "single_conv"
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    h = tf.zeros(z_shape, dtype=tf.float32)
    if learn_prior == "normal":
      prior_dist = tfp.distributions.Normal(h, tf.exp(h))
    elif learn_prior == "single_conv":
      prior_dist = single_conv_dist("top_learn_prior", h)
    else:
      raise ValueError("Expected learn_prior to be normal or single_conv "
                       "got %s" % learn_prior)
    return prior_dist


def uniform_binning_correction(x, n_bits=8):
  """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

  Args:
    x: 4-D Tensor of shape (NHWC)
    n_bits: optional.
  Returns:
    x: x ~ U(x, x + 1.0 / 256)
    objective: Equivalent to -q(x)*log(q(x)).
  """
  n_bins = 2**n_bits
  batch_size, height, width, n_channels = common_layers.shape_list(x)
  hwc = float(height * width * n_channels)

  x = x + tf.random_uniform(
      shape=(batch_size, height, width, n_channels),
      minval=0.0, maxval=1.0/n_bins)
  objective = -np.log(n_bins) * hwc * tf.ones(batch_size)
  return x, objective


@add_arg_scope
def encoder_decoder(name, x, hparams, eps=None, reverse=False,
                    cond_latents=None, condition=False, states=None):
  """Glow encoder-decoder. n_levels of (Squeeze + Flow + Split.) operations."""
  # TODO(mechcoder) Change return_type to a dict to be backward compatible.
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

    if states and len(states) != hparams.n_levels - 1:
      raise ValueError("Expected length of states to be %d, got %d" %
                       (hparams.n_levels - 1, len(states)))
    if states is None:
      states = [None] * (hparams.n_levels - 1)
    if eps and len(eps) != hparams.n_levels - 1:
      raise ValueError("Expected length of eps to be %d, got %d" %
                       (hparams.n_levels - 1, len(eps)))
    if eps is None:
      eps = [None] * (hparams.n_levels - 1)
    check_cond_latents(cond_latents, hparams)

    objective = 0.0
    all_eps = []
    all_latents = []
    new_states = []

    if not reverse:
      # Squeeze + Flow + Split
      for level in range(hparams.n_levels):
        x = squeeze("squeeze_%d" % level, x, factor=2, reverse=False)

        x, obj = revnet("revnet_%d" % level, x, hparams, reverse=False)
        objective += obj

        if level < hparams.n_levels - 1:
          curr_cond_latents = get_cond_latents_at_level(
              cond_latents, level, hparams)
          x, obj, eps, z, state = split("split_%d" % level, x, reverse=False,
                                        cond_latents=curr_cond_latents,
                                        condition=condition,
                                        hparams=hparams, state=states[level])
          objective += obj
          all_eps.append(eps)
          all_latents.append(z)
          new_states.append(state)

      return x, objective, all_eps, all_latents, new_states

    else:
      for level in reversed(range(hparams.n_levels)):
        if level < hparams.n_levels - 1:

          curr_cond_latents = get_cond_latents_at_level(
              cond_latents, level, hparams)

          x, latent, state = split("split_%d" % level, x, eps=eps[level],
                                   reverse=True, cond_latents=curr_cond_latents,
                                   condition=condition, hparams=hparams,
                                   state=states[level])
          new_states.append(state)
          all_latents.append(latent)

        x, obj = revnet(
            "revnet_%d" % level, x, hparams=hparams, reverse=True)
        objective += obj
        x = squeeze("squeeze_%d" % level, x, reverse=True)
      return x, objective, all_latents[::-1], new_states[::-1]
