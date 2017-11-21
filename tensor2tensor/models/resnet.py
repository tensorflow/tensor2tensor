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

"""Resnets."""
# Copied from cloud_tpu/models/resnet_garden and modified

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

# TODO(rsepassi): make hparams
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def bottleneck_block(inputs, filters, is_training, projection_shortcut, strides,
                     data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height, width].
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: channels_{first, last}

  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  out = inputs
  out = batch_norm_relu(out, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(out)

  do_bn_relus = [False, True, True]
  kernel_sizes = [1, 3, 1]
  layer_strides = [1, strides, 1]
  filter_sizes = [filters, filters, 4 * filters]

  for do_bn_relu, kernel_size, layer_stride, filter_size in zip(
      do_bn_relus, kernel_sizes, layer_strides, filter_sizes):
    if do_bn_relu:
      out = batch_norm_relu(out, is_training, data_format)
    out = conv2d_fixed_padding(
        inputs=out,
        filters=filter_size,
        kernel_size=kernel_size,
        strides=layer_stride,
        data_format=data_format)

  return out + shortcut


def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost.
  out = tf.layers.batch_normalization(
      inputs=inputs,
      axis=1 if data_format == "channels_first" else 3,
      momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training,
      fused=True)
  out = tf.nn.relu(out)
  return out


def block_layer(inputs, filters, block_fn, blocks, strides, is_training,
                data_format, name):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height, width].
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    data_format: channels_{first, last}
    name: A string name for the tensor output of the block layer.

  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

  return tf.identity(inputs, name)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A 4D tensor layed out according to data_format
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: channels_{first, last}

  Returns:
    A tensor of size [batch, channels, height_out, width_out] with the
      input either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  spatial_pads = [[pad_beg, pad_end], [pad_beg, pad_end]]
  if data_format == "channels_first":
    pads = [[0, 0], [0, 0]] + spatial_pads
  else:
    assert data_format == "channels_last"
    pads = [[0, 0]] + spatial_pads + [[0, 0]]
  padded_inputs = tf.pad(inputs, pads)
  return padded_inputs


def conv2d_fixed_padding(**kwargs):
  """conv2d with fixed_padding, based only on kernel_size."""
  strides = kwargs["strides"]
  if strides > 1:
    kwargs["inputs"] = fixed_padding(kwargs["inputs"], kwargs["kernel_size"],
                                     kwargs["data_format"])

  defaults = {
      "padding": ("SAME" if strides == 1 else "VALID"),
      "use_bias": False,
      "kernel_initializer": tf.variance_scaling_initializer(),
  }
  defaults.update(kwargs)

  return tf.layers.conv2d(**defaults)


def resnet50(inputs, hparams):
  """Resnet50."""
  is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
  block_fn = bottleneck_block

  out = inputs
  data_format = "channels_first" if hparams.use_nchw else "channels_last"
  if hparams.use_nchw:
    # Convert from channels_last (NHWC) to channels_first (NCHW). This provides
    # a large performance boost on GPU.
    out = tf.transpose(inputs, [0, 3, 1, 2])

  out = conv2d_fixed_padding(
      inputs=out, filters=64, kernel_size=7, strides=2, data_format=data_format)
  out = tf.identity(out, "initial_conv")
  out = tf.layers.max_pooling2d(
      inputs=out,
      pool_size=3,
      strides=2,
      padding="SAME",
      data_format=data_format)
  out = tf.identity(out, "initial_max_pool")

  for i, (num_filters, stride, block_size) in enumerate(
      zip(hparams.num_filters, hparams.strides, hparams.layer_sizes)):
    out = block_layer(
        inputs=out,
        filters=num_filters,
        block_fn=block_fn,
        blocks=block_size,
        strides=stride,
        is_training=is_training,
        data_format=data_format,
        name="block_layer_%d" % i)

  out = batch_norm_relu(out, is_training, data_format)
  out = tf.layers.average_pooling2d(
      inputs=out,
      pool_size=7,
      strides=1,
      padding="VALID",
      data_format=data_format)
  out = tf.identity(out, "final_avg_pool")

  if hparams.use_nchw:
    # Back to NHWC
    out = tf.transpose(out, [0, 2, 3, 1])
  return out


@registry.register_model
class Resnet50(t2t_model.T2TModel):

  def model_fn_body(self, features):
    return resnet50(features["inputs"], self.hparams)


@registry.register_hparams
def resnet_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.add_hparam("layer_sizes", [3, 4, 6, 3])
  hparams.add_hparam("use_nchw", True)
  hparams.add_hparam("num_filters", [64, 128, 256, 512])
  hparams.add_hparam("strides", [1, 2, 2, 2])
  hparams.tpu_batch_size_per_shard = 48
  return hparams
