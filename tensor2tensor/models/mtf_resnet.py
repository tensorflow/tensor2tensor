# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""ResNet model with model and data parallelism using MTF.

Integration of Mesh tensorflow with ResNet to do model parallelism.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import mesh_tensorflow as mtf

from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import mtf_model
from tensor2tensor.utils import registry
import tensorflow as tf


BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(inputs, is_training, relu=True):
  """Block of batch norm and relu."""
  inputs = mtf.layers.batch_norm(
      inputs,
      is_training,
      BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      init_zero=(not relu))
  if relu:
    inputs = mtf.relu(inputs)
  return inputs


def bottleneck_block(inputs,
                     filters,
                     is_training,
                     strides,
                     projection_shortcut=None,
                     row_blocks_dim=None,
                     col_blocks_dim=None):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: a `mtf.Tensor` of shape
        `[batch_dim, row_blocks, col_blocks, rows, cols, in_channels]`.
    filters: `int` number of filters for the first two convolutions. Note
        that the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training mode.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    projection_shortcut: `function` to use for projection shortcuts (typically
        a 1x1 convolution to match the filter dimensions). If None, no
        projection is used and the input is passed as unchanged through the
        shortcut connection.
    row_blocks_dim: a mtf.Dimension, row dimension which is
        spatially partitioned along mesh axis
    col_blocks_dim: a mtf.Dimension, row dimension which is
        spatially partitioned along mesh axis

  Returns:
    The output `Tensor` of the block.
  """
  shortcut = inputs

  filter_h_dim = mtf.Dimension("filter_height", 3)
  filter_w_dim = mtf.Dimension("filter_width", 3)
  one_h_dim = mtf.Dimension("filter_height", 1)
  one_w_dim = mtf.Dimension("filter_width", 1)

  if projection_shortcut is not None:
    filters_dim = mtf.Dimension("filtersp", filters)
    kernel = mtf.get_variable(
        inputs.mesh, "kernel", mtf.Shape(
            [one_h_dim, one_w_dim, inputs.shape.dims[-1], filters_dim]))
    shortcut = projection_shortcut(inputs, kernel)

  # First conv block
  filters1_dim = mtf.Dimension("filters1", filters)
  kernel1 = mtf.get_variable(
      inputs.mesh, "kernel1", mtf.Shape(
          [one_h_dim, one_w_dim, inputs.shape.dims[-1], filters1_dim]))
  inputs = mtf.conv2d_with_blocks(
      inputs,
      kernel1,
      strides=[1, 1, 1, 1],
      padding="SAME",
      h_blocks_dim=None, w_blocks_dim=col_blocks_dim)

  # TODO(nikip): Add Dropout?
  inputs = batch_norm_relu(inputs, is_training)

  # Second conv block
  filters2_dim = mtf.Dimension("filters2", 4*filters)
  kernel2 = mtf.get_variable(
      inputs.mesh, "kernel2", mtf.Shape(
          [filter_h_dim, filter_w_dim, filters1_dim, filters2_dim]))
  inputs = mtf.conv2d_with_blocks(
      inputs,
      kernel2,
      strides=[1, 1, 1, 1],
      padding="SAME",
      h_blocks_dim=row_blocks_dim, w_blocks_dim=col_blocks_dim)

  inputs = batch_norm_relu(inputs, is_training)

  # Third wide conv filter block
  filters3_dim = mtf.Dimension("filters3", filters)
  filters3_kernel = mtf.get_variable(
      inputs.mesh, "wide_kernel", mtf.Shape(
          [one_h_dim, one_w_dim, filters2_dim, filters3_dim]))
  inputs = mtf.conv2d_with_blocks(
      inputs,
      filters3_kernel,
      strides,
      padding="SAME",
      h_blocks_dim=None, w_blocks_dim=col_blocks_dim)

  # TODO(nikip): Althought the original resnet code has this batch norm, in our
  # setup this is causing no gradients to be passed. Investigate further.
  # inputs = batch_norm_relu(inputs, is_training, relu=True)

  # TODO(nikip): Maybe add residual with a projection?
  return mtf.relu(
      shortcut + mtf.rename_dimension(
          inputs, inputs.shape.dims[-1].name, shortcut.shape.dims[-1].name))


def block_layer(inputs,
                filters,
                blocks,
                strides,
                is_training,
                name,
                row_blocks_dim=None,
                col_blocks_dim=None):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    row_blocks_dim: a mtf.Dimension, row dimension which is
        spatially partitioned along mesh axis
    col_blocks_dim: a mtf.Dimension, row dimension which is
        spatially partitioned along mesh axis

  Returns:
    The output `Tensor` of the block layer.
  """
  with tf.variable_scope(name, default_name="block_layer"):
    # Only the first block per block_layer uses projection_shortcut and strides
    def projection_shortcut(inputs, kernel):
      """Project identity branch."""
      inputs = mtf.conv2d_with_blocks(
          inputs,
          kernel,
          strides=strides,
          padding="SAME",
          h_blocks_dim=None, w_blocks_dim=col_blocks_dim)
      return batch_norm_relu(
          inputs, is_training, relu=False)

    inputs = bottleneck_block(
        inputs,
        filters,
        is_training,
        strides=strides,
        projection_shortcut=projection_shortcut,
        row_blocks_dim=row_blocks_dim,
        col_blocks_dim=col_blocks_dim)

    for i in range(1, blocks):
      with tf.variable_scope("bottleneck_%d" % i):
        inputs = bottleneck_block(
            inputs,
            filters,
            is_training,
            strides=[1, 1, 1, 1],
            projection_shortcut=None,
            row_blocks_dim=row_blocks_dim,
            col_blocks_dim=col_blocks_dim)

    return inputs


@registry.register_model
class MtfResNet(mtf_model.MtfModel):
  """ResNet in mesh_tensorflow."""

  def set_activation_type(self):
    hparams = self._hparams
    if hparams.activation_dtype == "float32":
      activation_dtype = tf.float32
    elif hparams.activation_dtype == "float16":
      activation_dtype = tf.float16
    elif hparams.activation_dtype == "bfloat16":
      activation_dtype = tf.bfloat16
    else:
      raise ValueError(
          "unknown hparams.activation_dtype %s" % hparams.activation_dtype)
    return activation_dtype

  def mtf_model_fn(self, features, mesh):
    features = copy.copy(features)
    tf.logging.info("features = %s" % features)
    hparams = self._hparams
    activation_dtype = self.set_activation_type()
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN

    # Declare all the dimensions
    batch_dim = mtf.Dimension("batch", hparams.batch_size)
    hidden_dim = mtf.Dimension("hidden", hparams.hidden_size)
    filter_h_dim = mtf.Dimension("filter_height", 7)
    filter_w_dim = mtf.Dimension("filter_width", 7)
    filters = mtf.Dimension("filters", hparams.filter_sizes[0])
    rows_dim = mtf.Dimension("rows_size", hparams.rows_size)
    cols_dim = mtf.Dimension("cols_size", hparams.cols_size)
    row_blocks_dim = mtf.Dimension("row_blocks", hparams.row_blocks)
    col_blocks_dim = mtf.Dimension("col_blocks", hparams.col_blocks)
    classes_dim = mtf.Dimension("classes", 10)
    channels_dim = mtf.Dimension("channels", 3)
    one_channel_dim = mtf.Dimension("one_channel", 1)

    inputs = features["inputs"]
    x = mtf.import_tf_tensor(
        mesh, tf.reshape(inputs, [
            hparams.batch_size,
            hparams.row_blocks,
            hparams.rows_size // hparams.row_blocks,
            hparams.col_blocks,
            hparams.num_channels*hparams.cols_size // hparams.col_blocks,
            hparams.num_channels]),
        mtf.Shape(
            [batch_dim, row_blocks_dim, rows_dim,
             col_blocks_dim, cols_dim, channels_dim]))
    x = mtf.transpose(x, [batch_dim, row_blocks_dim, col_blocks_dim,
                          rows_dim, cols_dim, channels_dim])

    x = mtf.to_float(x)
    initial_filters = mtf.get_variable(
        mesh, "init_filters",
        mtf.Shape([filter_h_dim, filter_w_dim, channels_dim, filters]))
    x = mtf.conv2d_with_blocks(
        x,
        initial_filters,
        strides=[1, 1, 1, 1],
        padding="SAME",
        h_blocks_dim=None, w_blocks_dim=col_blocks_dim)

    x = batch_norm_relu(x, is_training)

    # Conv blocks
    # [block - strided block layer - strided block layer] x n
    for layer in range(hparams.num_layers):
      layer_name = "block_layer_%d" % layer
      with tf.variable_scope(layer_name):
        # Residual block layer
        x = block_layer(
            inputs=x,
            filters=hparams.filter_sizes[0],
            blocks=hparams.layer_sizes[0],
            strides=[1, 1, 1, 1],
            is_training=is_training,
            name="block_layer1",
            row_blocks_dim=None,
            col_blocks_dim=None)
        x = block_layer(
            inputs=x,
            filters=hparams.filter_sizes[1],
            blocks=hparams.layer_sizes[1],
            strides=[1, 1, 1, 1],
            is_training=is_training,
            name="block_layer2",
            row_blocks_dim=None,
            col_blocks_dim=None)
        x = block_layer(
            inputs=x,
            filters=hparams.filter_sizes[2],
            blocks=hparams.layer_sizes[2],
            strides=[1, 1, 1, 1],
            is_training=is_training,
            name="block_layer3",
            row_blocks_dim=None,
            col_blocks_dim=None)

    # Calculate the logits and loss.
    out = x
    outputs = mtf.layers.dense(
        out, hidden_dim,
        reduced_dims=out.shape.dims[-5:],
        activation=mtf.relu, name="dense")

    # We assume fixed vocab size for targets
    labels = tf.squeeze(tf.to_int32(features["targets"]), [2, 3])
    labels = mtf.import_tf_tensor(
        mesh, tf.reshape(labels, [hparams.batch_size]), mtf.Shape([batch_dim]))

    logits = mtf.layers.dense(outputs, classes_dim, name="logits")
    soft_targets = mtf.one_hot(labels, classes_dim, dtype=activation_dtype)
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, soft_targets, classes_dim)

    # Reshape logits so it doesn't break inside t2t.
    logits = mtf.reshape(
        logits,
        mtf.Shape([batch_dim, one_channel_dim, classes_dim]))
    loss = mtf.reduce_mean(loss)
    return logits, loss


@registry.register_hparams
def mtf_resnet_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.no_data_parallelism = True
  hparams.use_fixed_batch_size = True
  hparams.batch_size = 32
  hparams.max_length = 3072
  hparams.hidden_size = 256
  hparams.label_smoothing = 0.0
  # 8-way model-parallelism
  hparams.add_hparam("mesh_shape", "batch:8")
  hparams.add_hparam("layout", "batch:batch")
  hparams.add_hparam("filter_size", 1024)

  hparams.add_hparam("num_layers", 6)
  # Share weights between input and target embeddings
  hparams.shared_embedding = True

  hparams.shared_embedding_and_softmax_weights = True
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  hparams.add_hparam("d_kv", 32)

  # Image related hparams
  hparams.add_hparam("img_len", 32)
  hparams.add_hparam("num_channels", 3)
  hparams.add_hparam("row_blocks", 1)
  hparams.add_hparam("col_blocks", 1)
  hparams.add_hparam("rows_size", 32)
  hparams.add_hparam("cols_size", 32)

  # Model-specific parameters
  hparams.add_hparam("layer_sizes", [3, 4, 6, 3])
  hparams.add_hparam("filter_sizes", [64, 64, 128, 256, 512])
  hparams.add_hparam("is_cifar", False)

  # Variable init
  hparams.initializer = "normal_unit_scaling"
  hparams.initializer_gain = 2.

  # TODO(nikip): Change optimization scheme?
  hparams.learning_rate = 0.1
  return hparams


@registry.register_hparams
def mtf_resnet_tiny():
  """Catch bugs locally..."""
  hparams = mtf_resnet_base()
  hparams.num_layers = 2
  hparams.hidden_size = 64
  hparams.filter_size = 64
  hparams.batch_size = 16
  # data parallelism and model-parallelism
  hparams.col_blocks = 1
  hparams.mesh_shape = "batch:2"
  hparams.layout = "batch:batch"
  hparams.layer_sizes = [1, 2, 3]
  hparams.filter_sizes = [64, 64, 64]
  return hparams


@registry.register_hparams
def mtf_resnet_single():
  """Small single parameters."""
  hparams = mtf_resnet_tiny()
  hparams.mesh_shape = ""
  hparams.layout = ""
  hparams.hidden_size = 32
  hparams.filter_size = 32
  hparams.batch_size = 1
  hparams.num_encoder_layers = 1
  hparams.num_layers = 1
  hparams.block_length = 16
  return hparams


@registry.register_hparams
def mtf_resnet_base_single():
  """Small single parameters."""
  hparams = mtf_resnet_base()
  hparams.num_layers = 6
  hparams.filter_size = 256
  hparams.block_length = 128
  hparams.mesh_shape = ""
  hparams.layout = ""
  return hparams


@registry.register_hparams
def mtf_resnet_base_cifar():
  """Data parallel CIFAR parameters."""
  hparams = mtf_resnet_base()
  hparams.mesh_shape = "batch:32"
  hparams.layoyt = "batch:batch"
  hparams.batch_size = 8
  hparams.num_layers = 12
  hparams.block_length = 256
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.learning_rate = 0.5
  hparams.learning_rate_warmup_steps = 4000
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.unconditional = True
  return hparams
