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

"""BlueNet: and out of the blue network to experiment with shake-shake."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

# var:          1d tensor, raw weights for each choice
# tempered_var: raw weights with temperature applied
# inv_t:        inverse of the temperature to use when normalizing `var`
# normalized:   same shape as var, but where each item is between 0 and 1, and
#               the sum is 1
SelectionWeights = collections.namedtuple(
    "SelectionWeights", ["var", "tempered_var", "inv_t", "normalized"])


def create_selection_weights(name,
                             type_,
                             shape,
                             inv_t=1,
                             initializer=tf.zeros_initializer(),
                             regularizer=None,
                             names=None):
  """Create a SelectionWeights tuple.

  Args:
    name: Name for the underlying variable containing the unnormalized weights.
    type_: "softmax" or "sigmoid" or ("softmax_topk", k) where k is an int.
    shape: Shape for the variable.
    inv_t: Inverse of the temperature to use in normalization.
    initializer: Initializer for the variable, passed to `tf.get_variable`.
    regularizer: Regularizer for the variable. A callable which accepts
      `tempered_var` and `normalized`.
    names: Name of each selection.

  Returns:
    The created SelectionWeights tuple.

  Raises:
    ValueError: if type_ is not in the supported range.
  """
  var = tf.get_variable(name, shape, initializer=initializer)

  if callable(inv_t):
    inv_t = inv_t(var)
  if inv_t == 1:
    tempered_var = var
  else:
    tempered_var = var * inv_t

  if type_ == "softmax":
    weights = tf.nn.softmax(tempered_var)
  elif type_ == "sigmoid":
    weights = tf.nn.sigmoid(tempered_var)
  elif isinstance(type_, (list, tuple)) and type_[0] == "softmax_topk":
    assert len(shape) == 1
    # TODO(rshin): Change this to select without replacement?
    selection = tf.multinomial(tf.expand_dims(var, axis=0), 4)
    selection = tf.squeeze(selection, axis=0)  # [k] selected classes.
    to_run = tf.one_hot(selection, shape[0])  # [k x nmodules] one-hot.
    # [nmodules], 0=not run, 1=run.
    to_run = tf.minimum(tf.reduce_sum(to_run, axis=0), 1)
    weights = tf.nn.softmax(tempered_var - 1e9 * (1.0 - to_run))
  else:
    raise ValueError("Unknown type: %s" % type_)

  if regularizer is not None:
    loss = regularizer(tempered_var, weights)
    if loss is not None:
      tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)

  if names is not None:
    tf.get_collection_ref("selection_weight_names/" + var.name).extend(
        names.flatten() if isinstance(names, np.ndarray) else names)
    tf.add_to_collection("selection_weight_names_tensor/" + var.name,
                         tf.constant(names))

  return SelectionWeights(
      var=var, tempered_var=tempered_var, inv_t=inv_t, normalized=weights)


def kernel_premultiplier(max_kernel_size, kernel_sizes, input_channels,
                         kernel_selection_weights, channel_selection_weights):
  """Get weights to multiply the kernel with, before convolving.

  Args:
    max_kernel_size: (int, int) tuple giving the largest kernel size.
    kernel_sizes: A list of (height, width) pairs of integers, containing
      different kernel sizes to use.
    input_channels: A list of (begin, end) pairs of integers, which describe
      which channels in the input to use.
    kernel_selection_weights: SelectionWeights object to use for choosing
      among kernel sizes.
    channel_selection_weights: SelectionWeights object to use for choosing
      among which input channels to use.

  Returns:
    The multiplier.
  """
  kernel_weights = []
  for kernel_i, (h, w) in enumerate(kernel_sizes):
    top = (max_kernel_size[0] - h) // 2
    bot = max_kernel_size[0] - h - top
    left = (max_kernel_size[1] - w) // 2
    right = max_kernel_size[1] - w - left
    kernel_weight = tf.fill((h, w),
                            kernel_selection_weights.normalized[kernel_i])
    if top != 0 or bot != 0 or left != 0 or right != 0:
      kernel_weight = tf.pad(kernel_weight, [[top, bot], [left, right]])
    kernel_weights.append(kernel_weight)
  kernel_weight = tf.add_n(kernel_weights)

  channel_weights = []
  min_channel = np.min(input_channels)
  max_channel = np.max(input_channels)
  for channel_i, (begin, end) in enumerate(input_channels):
    channel_weight = tf.pad(
        tf.fill((end - begin,),
                channel_selection_weights.normalized[channel_i]),
        [[begin - min_channel, max_channel - end]])
    channel_weights.append(channel_weight)
  channel_weight = tf.add_n(channel_weights)

  multiplier = (tf.reshape(kernel_weight, max_kernel_size +
                           (1, 1)) * tf.reshape(channel_weight, (1, 1, -1, 1)))
  return multiplier


def make_subseparable_kernel(kernel_size, input_channels, filters, separability,
                             kernel_initializer, kernel_regularizer):
  """Make a kernel to do subseparable convolution wiht  `tf.nn.conv2d`.

  Args:
    kernel_size: (height, width) tuple.
    input_channels: Number of input channels.
    filters: Number of output channels.
    separability: Integer denoting separability.
    kernel_initializer: Initializer to use for the kernel.
    kernel_regularizer: Regularizer to use for the kernel.

  Returns:
    A 4D tensor.
  """
  if separability == 1:
    # Non-separable convolution
    return tf.get_variable(
        "kernel",
        kernel_size + (input_channels, filters),
        initializer=kernel_initializer,
        regularizer=kernel_regularizer)

  elif separability == 0 or separability == -1:
    # Separable convolution
    # TODO(rshin): Check initialization is as expected, as these are not 4D.
    depthwise_kernel = tf.get_variable(
        "depthwise_kernel",
        kernel_size + (input_channels,),
        initializer=kernel_initializer,
        regularizer=kernel_regularizer)

    pointwise_kernel = tf.get_variable(
        "pointwise_kernel", (input_channels, filters),
        initializer=kernel_initializer,
        regularizer=kernel_regularizer)

    expanded_depthwise_kernel = tf.transpose(
        tf.scatter_nd(
            indices=tf.tile(
                tf.expand_dims(tf.range(0, input_channels), axis=1), [1, 2]),
            updates=tf.transpose(depthwise_kernel, (2, 0, 1)),
            shape=(input_channels, input_channels) + kernel_size), (2, 3, 0, 1))

    return tf.reshape(
        tf.matmul(
            tf.reshape(expanded_depthwise_kernel, (-1, input_channels)),
            pointwise_kernel), kernel_size + (input_channels, filters))

  elif separability >= 2:
    assert filters % separability == 0, (filters, separability)
    assert input_channels % separability == 0, (filters, separability)

    raise NotImplementedError

  elif separability <= -2:
    separability *= -1
    assert filters % separability == 0, (filters, separability)
    assert input_channels % separability == 0, (filters, separability)

    raise NotImplementedError


def multi_subseparable_conv(inputs,
                            filters,
                            kernel_sizes,
                            input_channels,
                            separabilities,
                            kernel_selection_weights=None,
                            channel_selection_weights=None,
                            separability_selection_weights=None,
                            kernel_selection_weights_params=None,
                            channel_selection_weights_params=None,
                            separability_selection_weights_params=None,
                            kernel_initializer=None,
                            kernel_regularizer=None,
                            scope=None):
  """Simultaneously compute different kinds of convolutions on subsets of input.

  Args:
    inputs: 4D tensor containing the input, in NHWC format.
    filters: Integer, number of output channels.
    kernel_sizes: A list of (height, width) pairs of integers, containing
      different kernel sizes to use.
    input_channels: A list of (begin, end) pairs of integers, which describe
      which channels in the input to use.
    separabilities: An integer or a list, how separable are the convolutions.
    kernel_selection_weights: SelectionWeights object to use for choosing
      among kernel sizes.
    channel_selection_weights: SelectionWeights object to use for choosing
      among which input channels to use.
    separability_selection_weights: SelectionWeights object to use for choosing
      separability.
    kernel_selection_weights_params: dict with up to three keys
      - initializer
      - regularizer
      - inv_t
    channel_selection_weights_params: dict with up to three keys
      - initializer
      - regularizer
      - inv_t
    separability_selection_weights_params: dict with up to three keys
      - initializer
      - regularizer
      - inv_t
    kernel_initializer: Initializer to use for kernels.
    kernel_regularizer: Regularizer to use for kernels.
    scope: the scope to use.

  Returns:
    Result of convolution.
  """
  kernel_selection_weights_params = kernel_selection_weights_params or {}
  channel_selection_weights_params = channel_selection_weights_params or {}
  if separability_selection_weights_params is None:
    separability_selection_weights_params = {}

  # Get input image size.
  input_shape = inputs.get_shape().as_list()
  assert len(input_shape) == 4
  in_channels = input_shape[3]
  assert in_channels is not None

  max_kernel_size = tuple(np.max(kernel_sizes, axis=0))
  max_num_channels = np.max(input_channels) - np.min(input_channels)

  with tf.variable_scope(scope or "selection_weights"):
    if kernel_selection_weights is None:
      kernel_selection_weights = create_selection_weights(
          "kernels",
          "softmax", (len(kernel_sizes),),
          names=["kernel_h{}_w{}".format(h, w) for h, w in kernel_sizes],
          **kernel_selection_weights_params)

    if channel_selection_weights is None:
      channel_selection_weights = create_selection_weights(
          "channels",
          "softmax", (len(input_channels),),
          names=["channels_{}_{}".format(c1, c2) for c1, c2 in input_channels],
          **channel_selection_weights_params)

    if separability_selection_weights is None:
      separability_selection_weights = create_selection_weights(
          "separability",
          "softmax", (len(separabilities),),
          names=["separability_{}".format(s) for s in separabilities],
          **separability_selection_weights_params)

  kernels = []
  for separability in separabilities:
    with tf.variable_scope("separablity_{}".format(separability)):
      kernel = make_subseparable_kernel(max_kernel_size, max_num_channels,
                                        filters, separability,
                                        kernel_initializer, kernel_regularizer)

      premultiplier = kernel_premultiplier(
          max_kernel_size, kernel_sizes, input_channels,
          kernel_selection_weights, channel_selection_weights)

      kernels.append(kernel * premultiplier)

  kernel = tf.add_n([
      separability_selection_weights.normalized[i] * k
      for i, k in enumerate(kernels)
  ])

  if np.min(input_channels) != 0 or np.max(input_channels) != in_channels:
    inputs = inputs[:, :, :, np.min(input_channels):np.max(input_channels)]

  return tf.nn.conv2d(
      inputs,
      filter=kernel,
      strides=[1, 1, 1, 1],
      padding="SAME",
      data_format="NHWC",
      name="conv2d")


def conv_module(kw, kh, sep, div):

  def convfn(x, hparams):
    return common_layers.subseparable_conv(
        x,
        hparams.hidden_size // div, (kw, kh),
        padding="SAME",
        separability=sep,
        name="conv_%d%d_sep%d_div%d" % (kw, kh, sep, div))

  return convfn


def multi_conv_module(kernel_sizes, seps):

  def convfn(x, hparams):
    return multi_subseparable_conv(x, hparams.hidden_size, kernel_sizes,
                                   [(0, hparams.hidden_size)], seps)

  return convfn


def layernorm_module(x, hparams):
  return common_layers.layer_norm(x, hparams.hidden_size, name="layer_norm")


def noamnorm_module(x, hparams):
  del hparams  # Unused.
  return common_layers.noam_norm(x)


def identity_module(x, hparams):
  del hparams  # Unused.
  return x


def first_binary_module(x, y, hparams):
  del y, hparams  # Unused.
  return x


def second_binary_module(x, y, hparams):
  del x, hparams  # Unused.
  return y


def sum_binary_module(x, y, hparams):
  del hparams  # Unused.
  return x + y


def shakeshake_binary_module(x, y, hparams):
  del hparams  # Unused.
  return common_layers.shakeshake2(x, y)


def run_binary_modules(modules, cur1, cur2, hparams):
  """Run binary modules."""
  selection_weights = create_selection_weights(
      "selection",
      "softmax",
      shape=[len(modules)],
      inv_t=100.0 * common_layers.inverse_exp_decay(
          hparams.anneal_until, min_value=0.01))
  all_res = [modules[n](cur1, cur2, hparams) for n in xrange(len(modules))]
  all_res = tf.concat([tf.expand_dims(r, axis=0) for r in all_res], axis=0)
  res = all_res * tf.reshape(selection_weights.normalized, [-1, 1, 1, 1, 1])
  return tf.reduce_sum(res, axis=0)


def run_unary_modules_basic(modules, cur, hparams):
  """Run unary modules."""
  selection_weights = create_selection_weights(
      "selection",
      "softmax",
      shape=[len(modules)],
      inv_t=100.0 * common_layers.inverse_exp_decay(
          hparams.anneal_until, min_value=0.01))
  all_res = [modules[n](cur, hparams) for n in xrange(len(modules))]
  all_res = tf.concat([tf.expand_dims(r, axis=0) for r in all_res], axis=0)
  res = all_res * tf.reshape(selection_weights.normalized, [-1, 1, 1, 1, 1])
  return tf.reduce_sum(res, axis=0)


def run_unary_modules_sample(modules, cur, hparams, k):
  """Run modules, sampling k."""
  selection_weights = create_selection_weights(
      "selection", ("softmax_topk", k),
      shape=[len(modules)],
      inv_t=100.0 * common_layers.inverse_exp_decay(
          hparams.anneal_until, min_value=0.01))
  all_res = [
      tf.cond(
          tf.less(selection_weights.normalized[n], 1e-6),
          lambda: tf.zeros_like(cur),
          lambda i=n: modules[i](cur, hparams)) for n in xrange(len(modules))
  ]
  all_res = tf.concat([tf.expand_dims(r, axis=0) for r in all_res], axis=0)
  res = all_res * tf.reshape(selection_weights.normalized, [-1, 1, 1, 1, 1])
  return tf.reduce_sum(res, axis=0)


def run_unary_modules(modules, cur, hparams):
  if len(modules) < 8:
    return run_unary_modules_basic(modules, cur, hparams)
  return run_unary_modules_sample(modules, cur, hparams, 4)


def batch_deviation(x):
  """Average deviation of the batch."""
  x_mean = tf.reduce_mean(x, axis=[0], keep_dims=True)
  x_variance = tf.reduce_mean(tf.square(x - x_mean), axis=[0], keep_dims=True)
  return tf.reduce_mean(tf.sqrt(x_variance))


@registry.register_model
class BlueNet(t2t_model.T2TModel):

  def body(self, features):
    hparams = self._hparams
    # TODO(rshin): Give identity_module lower weight by default.
    multi_conv = multi_conv_module(
        kernel_sizes=[(3, 3), (5, 5), (7, 7)], seps=[0, 1])
    conv_modules = [multi_conv, identity_module]
    activation_modules = [
        identity_module, lambda x, _: tf.nn.relu(x), lambda x, _: tf.nn.elu(x),
        lambda x, _: tf.tanh(x)
    ]
    norm_modules = [identity_module, layernorm_module, noamnorm_module]
    binary_modules = [
        first_binary_module, second_binary_module, sum_binary_module,
        shakeshake_binary_module
    ]
    inputs = features["inputs"]

    def run_unary(x, name):
      """A single step of unary modules."""
      x_shape = x.get_shape()
      with tf.variable_scope(name):
        with tf.variable_scope("norm"):
          x = run_unary_modules(norm_modules, x, hparams)
          x.set_shape(x_shape)
        with tf.variable_scope("activation"):
          x = run_unary_modules(activation_modules, x, hparams)
          x.set_shape(x_shape)
        with tf.variable_scope("conv"):
          x = run_unary_modules(conv_modules, x, hparams)
          x.set_shape(x_shape)
      return tf.nn.dropout(x, 1.0 - hparams.dropout), batch_deviation(x)

    cur1, cur2, cur3, extra_loss = inputs, inputs, inputs, 0.0
    cur_shape = inputs.get_shape()
    for i in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % i):
        cur1, loss1 = run_unary(cur1, "unary1")
        cur2, loss2 = run_unary(cur2, "unary2")
        cur3, loss3 = run_unary(cur2, "unary3")
        extra_loss += (loss1 + loss2 + loss3) / float(hparams.num_hidden_layers)
        with tf.variable_scope("binary1"):
          next1 = run_binary_modules(binary_modules, cur1, cur2, hparams)
          next1.set_shape(cur_shape)
        with tf.variable_scope("binary2"):
          next2 = run_binary_modules(binary_modules, cur1, cur3, hparams)
          next2.set_shape(cur_shape)
        with tf.variable_scope("binary3"):
          next3 = run_binary_modules(binary_modules, cur2, cur3, hparams)
          next3.set_shape(cur_shape)
        cur1, cur2, cur3 = next1, next2, next3

    anneal = common_layers.inverse_exp_decay(hparams.anneal_until)
    extra_loss *= hparams.batch_deviation_loss_factor * anneal
    return cur1, extra_loss


@registry.register_hparams
def bluenet_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 4096
  hparams.hidden_size = 256
  hparams.dropout = 0.2
  hparams.symbol_dropout = 0.5
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 8
  hparams.kernel_height = 3
  hparams.kernel_width = 3
  hparams.learning_rate_decay_scheme = "exp10k"
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 3.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  hparams.add_hparam("anneal_until", 40000)
  hparams.add_hparam("batch_deviation_loss_factor", 5.0)
  return hparams


@registry.register_hparams
def bluenet_tiny():
  hparams = bluenet_base()
  hparams.batch_size = 1024
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 4
  hparams.learning_rate_decay_scheme = "none"
  return hparams
