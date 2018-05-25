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
"""Resnets."""
# Copied from cloud_tpu/models/resnet/resnet_model.py and modified

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs,
                    is_training,
                    relu=True,
                    init_zero=False,
                    data_format="channels_first"):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if init_zero:
    gamma_initializer = tf.zeros_initializer()
  else:
    gamma_initializer = tf.ones_initializer()

  if data_format == "channels_first":
    axis = 1
  else:
    axis = 3

  inputs = tf.layers.batch_normalization(
      inputs=inputs,
      axis=axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      center=True,
      scale=True,
      training=is_training,
      fused=True,
      gamma_initializer=gamma_initializer)

  if relu:
    inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format="channels_first"):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == "channels_first":
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format="channels_first"):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return tf.layers.conv2d(
      inputs=inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=("SAME" if strides == 1 else "VALID"),
      use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def residual_block(inputs,
                   filters,
                   is_training,
                   projection_shortcut,
                   strides,
                   final_block,
                   data_format="channels_first"):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    projection_shortcut: `function` to use for projection shortcuts (typically
        a 1x1 convolution to match the filter dimensions). If None, no
        projection is used and the input is passed as unchanged through the
        shortcut connection.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    final_block: unused parameter to keep the same function signature as
        `bottleneck_block`.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  del final_block
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs,
                     filters,
                     is_training,
                     projection_shortcut,
                     strides,
                     final_block,
                     data_format="channels_first"):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    projection_shortcut: `function` to use for projection shortcuts (typically
        a 1x1 convolution to match the filter dimensions). If None, no
        projection is used and the input is passed as unchanged through the
        shortcut connection.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    final_block: `bool` set to True if it is this the final block in the group.
        This is changes the behavior of batch normalization initialization for
        the final batch norm in a block.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  # TODO(chrisying): this block is technically the post-activation resnet-v1
  # bottleneck unit. Test with v2 (pre-activation) and replace if there is no
  # difference for consistency.
  shortcut = inputs
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters,
      kernel_size=3,
      strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=4 * filters,
      kernel_size=1,
      strides=1,
      data_format=data_format)
  inputs = batch_norm_relu(
      inputs,
      is_training,
      relu=False,
      init_zero=final_block,
      data_format=data_format)

  return tf.nn.relu(inputs + shortcut)


def block_layer(inputs,
                filters,
                block_fn,
                blocks,
                strides,
                is_training,
                name,
                data_format="channels_first"):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters_out,
        kernel_size=1,
        strides=strides,
        data_format=data_format)
    return batch_norm_relu(
        inputs, is_training, relu=False, data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    False, data_format)

  for i in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1, (i + 1 == blocks),
                      data_format)

  return tf.identity(inputs, name)


def resnet_v2(inputs,
              block_fn,
              layers,
              filters,
              data_format="channels_first",
              is_training=False):
  """Resnet model.

  Args:
    inputs: `Tensor` images.
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 3 or 4 `int`s denoting the number of blocks to include in
      each of the 3 or 4 block groups. Each group consists of blocks that take
      inputs of the same resolution.
    filters: list of 4 or 5 `int`s denoting the number of filter to include in
      block.
    data_format: `str`, "channels_first" `[batch, channels, height,
        width]` or "channels_last" `[batch, height, width, channels]`.
    is_training: bool, build in training mode or not.

  Returns:
    Pre-logit activations.
  """
  inputs = conv2d_fixed_padding(
      inputs=inputs,
      filters=filters[0],
      kernel_size=7,
      strides=2,
      data_format=data_format)
  inputs = tf.identity(inputs, "initial_conv")
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)

  inputs = tf.layers.max_pooling2d(
      inputs=inputs,
      pool_size=3,
      strides=2,
      padding="SAME",
      data_format=data_format)
  inputs = tf.identity(inputs, "initial_max_pool")

  inputs = block_layer(
      inputs=inputs,
      filters=filters[1],
      block_fn=block_fn,
      blocks=layers[0],
      strides=1,
      is_training=is_training,
      name="block_layer1",
      data_format=data_format)
  inputs = block_layer(
      inputs=inputs,
      filters=filters[2],
      block_fn=block_fn,
      blocks=layers[1],
      strides=2,
      is_training=is_training,
      name="block_layer2",
      data_format=data_format)
  inputs = block_layer(
      inputs=inputs,
      filters=filters[3],
      block_fn=block_fn,
      blocks=layers[2],
      strides=2,
      is_training=is_training,
      name="block_layer3",
      data_format=data_format)
  if filters[4]:
    inputs = block_layer(
        inputs=inputs,
        filters=filters[4],
        block_fn=block_fn,
        blocks=layers[3],
        strides=2,
        is_training=is_training,
        name="block_layer4",
        data_format=data_format)

  return inputs


@registry.register_model
class Resnet(t2t_model.T2TModel):
  """Residual Network."""

  def body(self, features):
    hp = self.hparams
    block_fns = {
        "residual": residual_block,
        "bottleneck": bottleneck_block,
    }
    assert hp.block_fn in block_fns

    inputs = features["inputs"]

    data_format = "channels_last"
    if hp.use_nchw:
      # Convert from channels_last (NHWC) to channels_first (NCHW). This
      # provides a large performance boost on GPU.
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
      data_format = "channels_first"

    out = resnet_v2(
        inputs,
        block_fns[hp.block_fn],
        hp.layer_sizes,
        hp.filter_sizes,
        data_format,
        is_training=hp.mode == tf.estimator.ModeKeys.TRAIN)

    if hp.use_nchw:
      out = tf.transpose(out, [0, 2, 3, 1])

    return out

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Predict."""
    del decode_length, beam_size, top_beams, alpha, use_tpu
    assert features is not None
    logits, _ = self(features)  # pylint: disable=not-callable
    assert len(logits.get_shape()) == 5
    logits = tf.squeeze(logits, [1, 2, 3])
    log_probs = common_layers.log_prob_from_logits(logits)
    predictions, scores = common_layers.argmax_with_score(log_probs)
    return {
        "outputs": predictions,
        "scores": scores,
    }


def resnet_base():
  """Set of hyperparameters."""
  # For imagenet on TPU:
  # Set train_steps=120000
  # Set eval_steps=48

  # Base
  hparams = common_hparams.basic_params1()

  # Model-specific parameters
  hparams.add_hparam("layer_sizes", [3, 4, 6, 3])
  hparams.add_hparam("filter_sizes", [64, 64, 128, 256, 512])
  hparams.add_hparam("block_fn", "bottleneck")
  hparams.add_hparam("use_nchw", True)

  # Variable init
  hparams.initializer = "normal_unit_scaling"
  hparams.initializer_gain = 2.

  # Optimization
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  hparams.optimizer_momentum_nesterov = True
  hparams.weight_decay = 1e-4
  hparams.clip_grad_norm = 0.0
  # (base_lr=0.1) * (batch_size=128*8 (on TPU, or 8 GPUs)=1024) / (256.)
  hparams.learning_rate = 0.4
  hparams.learning_rate_decay_scheme = "cosine"
  # For image_imagenet224, 120k training steps, which effectively makes this a
  # cosine decay (i.e. no cycles).
  hparams.learning_rate_cosine_cycle_steps = 120000

  hparams.batch_size = 128
  return hparams


@registry.register_hparams
def resnet_50():
  hp = resnet_base()
  return hp


@registry.register_hparams
def resnet_18():
  hp = resnet_base()
  hp.block_fn = "residual"
  hp.layer_sizes = [2, 2, 2, 2]
  return hp


@registry.register_hparams
def resnet_imagenet_34():
  """Set of hyperparameters."""
  hp = resnet_base()
  hp.block_fn = "residual"
  hp.layer_sizes = [2, 4, 8, 2]

  return hp


@registry.register_hparams
def resnet_imagenet_102():
  hp = resnet_imagenet_34()
  hp.layer_sizes = [3, 8, 36, 3]
  return hp


@registry.register_hparams
def resnet_cifar_15():
  """Set of hyperparameters."""
  hp = resnet_base()
  hp.block_fn = "residual"
  hp.layer_sizes = [2, 2, 2]
  hp.filter_sizes = [16, 16, 32, 64, None]

  hp.learning_rate = 0.1 * 128. * 8. / 256.
  hp.learning_rate_decay_scheme = "piecewise"
  hp.add_hparam("learning_rate_boundaries", [40000, 60000, 80000])
  hp.add_hparam("learning_rate_multiples", [0.1, 0.01, 0.001])

  return hp


@registry.register_hparams
def resnet_cifar_32():
  hp = resnet_cifar_15()
  hp.layer_sizes = [5, 5, 5]
  return hp


@registry.register_hparams
def resnet_34():
  hp = resnet_base()
  hp.block_fn = "residual"
  return hp


@registry.register_hparams
def resnet_101():
  hp = resnet_base()
  hp.layer_sizes = [3, 4, 23, 3]
  return hp


@registry.register_hparams
def resnet_152():
  hp = resnet_base()
  hp.layer_sizes = [3, 8, 36, 3]
  return hp


@registry.register_hparams
def resnet_200():
  hp = resnet_base()
  hp.layer_sizes = [3, 24, 36, 3]
  return hp
