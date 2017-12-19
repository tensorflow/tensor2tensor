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

"""Xception."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def residual_block(x, hparams):
  """A stack of convolution blocks with residual connection."""
  k = (hparams.kernel_height, hparams.kernel_width)
  dilations_and_kernels = [((1, 1), k) for _ in xrange(3)]
  y = common_layers.subseparable_conv_block(
      x,
      hparams.hidden_size,
      dilations_and_kernels,
      padding="SAME",
      separability=0,
      name="residual_block")
  x = common_layers.layer_norm(x + y, hparams.hidden_size, name="lnorm")
  return tf.nn.dropout(x, 1.0 - hparams.dropout)


def xception_internal(inputs, hparams):
  """Xception body."""
  with tf.variable_scope("xception"):
    cur = inputs

    if cur.get_shape().as_list()[1] > 200:
      # Large image, Xception entry flow
      cur = xception_entry(cur, hparams.hidden_size)
    else:
      # Small image, conv
      cur = common_layers.conv_block(
          cur,
          hparams.hidden_size, [((1, 1), (3, 3))],
          first_relu=False,
          padding="SAME",
          force2d=True,
          name="small_image_conv")

    for i in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % i):
        cur = residual_block(cur, hparams)

    return xception_exit(cur)


def xception_entry(inputs, hidden_dim):
  with tf.variable_scope("xception_entry"):

    def xnet_resblock(x, filters, res_relu, name):
      with tf.variable_scope(name):
        y = common_layers.separable_conv_block(
            x,
            filters, [((1, 1), (3, 3)), ((1, 1), (3, 3))],
            first_relu=True,
            padding="SAME",
            force2d=True,
            name="sep_conv_block")
        y = common_layers.pool(y, (3, 3), "MAX", "SAME", strides=(2, 2))
        return y + common_layers.conv_block(
            x,
            filters, [((1, 1), (1, 1))],
            padding="SAME",
            strides=(2, 2),
            first_relu=res_relu,
            force2d=True,
            name="res_conv0")

    inputs = common_layers.standardize_images(inputs)
    # TODO(lukaszkaiser): summaries here don't work in multi-problem case yet.
    # tf.summary.image("inputs", inputs, max_outputs=2)
    x = common_layers.conv_block(
        inputs,
        32, [((1, 1), (3, 3))],
        first_relu=False,
        padding="SAME",
        strides=(2, 2),
        force2d=True,
        name="conv0")
    x = common_layers.conv_block(
        x, 64, [((1, 1), (3, 3))], padding="SAME", force2d=True, name="conv1")
    x = xnet_resblock(x, min(128, hidden_dim), True, "block0")
    x = xnet_resblock(x, min(256, hidden_dim), False, "block1")
    return xnet_resblock(x, hidden_dim, False, "block2")


def xception_exit(inputs):
  with tf.variable_scope("xception_exit"):
    x = inputs
    x_shape = x.get_shape().as_list()
    if x_shape[1] is None or x_shape[2] is None:
      length_float = tf.to_float(tf.shape(x)[1])
      length_float *= tf.to_float(tf.shape(x)[2])
      spatial_dim_float = tf.sqrt(length_float)
      spatial_dim = tf.to_int32(spatial_dim_float)
      x_depth = x_shape[3]
      x = tf.reshape(x, [-1, spatial_dim, spatial_dim, x_depth])
    elif x_shape[1] != x_shape[2]:
      spatial_dim = int(math.sqrt(float(x_shape[1] * x_shape[2])))
      if spatial_dim * spatial_dim != x_shape[1] * x_shape[2]:
        raise ValueError("Assumed inputs were square-able but they were "
                         "not. Shape: %s" % x_shape)
      x = tf.reshape(x, [-1, spatial_dim, spatial_dim, x_depth])

    x = common_layers.conv_block_downsample(x, (3, 3), (2, 2), "SAME")
    return tf.nn.relu(x)


@registry.register_model
class Xception(t2t_model.T2TModel):

  def body(self, features):
    return xception_internal(features["inputs"], self._hparams)


@registry.register_hparams
def xception_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 128
  hparams.hidden_size = 768
  hparams.dropout = 0.2
  hparams.symbol_dropout = 0.2
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 8
  hparams.kernel_height = 3
  hparams.kernel_width = 3
  hparams.learning_rate_decay_scheme = "exp50k"
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 3.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  return hparams


@registry.register_hparams
def xception_tiny():
  hparams = xception_base()
  hparams.batch_size = 2
  hparams.hidden_size = 64
  hparams.num_hidden_layers = 2
  hparams.learning_rate_decay_scheme = "none"
  return hparams


@registry.register_hparams
def xception_tiny_tpu():
  hparams = xception_base()
  hparams.tpu_batch_size_per_shard = 2
  # The base exp50k scheme uses a cond which fails to compile on TPU
  hparams.learning_rate_decay_scheme = "noam"
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.optimizer = "TrueAdam"
  return hparams
