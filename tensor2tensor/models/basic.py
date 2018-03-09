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

"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class BasicFcRelu(t2t_model.T2TModel):

  def body(self, features):
    hparams = self._hparams
    x = features["inputs"]
    shape = common_layers.shape_list(x)
    x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
    for i in xrange(hparams.num_hidden_layers):
      x = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)
      x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
      x = tf.nn.relu(x)
    return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # 4D For T2T.


@registry.register_model
class BasicAutoencoder(t2t_model.T2TModel):
  """A basic autoencoder, try with image_mnist_rev or image_cifar10_rev."""

  def bottleneck(self, x, res_size):
    hparams = self._hparams
    x = tf.layers.dense(x, hparams.bottleneck_size, name="bottleneck")
    x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
    x = tf.layers.dense(x, res_size, name="unbottleneck")
    return x

  def body(self, features):
    hparams = self._hparams
    is_training = hparams.mode == tf.estimator.ModeKeys.TRAIN
    x = features["targets"]
    shape = common_layers.shape_list(x)
    kernel = (hparams.kernel_height, hparams.kernel_width)
    is1d = shape[2] == 1
    kernel = (hparams.kernel_height, 1) if is1d else kernel
    strides = (2, 1) if is1d else (2, 2)
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2**hparams.num_hidden_layers, axis=1)
    if not is1d:
      x, _ = common_layers.pad_to_same_length(
          x, x, final_length_divisible_by=2**hparams.num_hidden_layers, axis=2)
    # Down-convolutions.
    for i in xrange(hparams.num_hidden_layers):
      x = tf.layers.conv2d(
          x, hparams.hidden_size * 2**(i + 1), kernel, strides=strides,
          padding="SAME", activation=tf.nn.relu, name="conv_%d" % i)
      x = common_layers.layer_norm(x)
    # Bottleneck (mix during early training, not too important but very stable).
    b = self.bottleneck(x, hparams.hidden_size * 2**hparams.num_hidden_layers)
    x = common_layers.mix(b, x, hparams.bottleneck_warmup_steps, is_training)
    # Up-convolutions.
    for i in xrange(hparams.num_hidden_layers):
      j = hparams.num_hidden_layers - i - 1
      x = tf.layers.conv2d_transpose(
          x, hparams.hidden_size * 2**j, kernel, strides=strides,
          padding="SAME", activation=tf.nn.relu, name="deconv_%d" % j)
      x = common_layers.layer_norm(x)
    res = x[:, :shape[1], :shape[2], :]
    return common_layers.mix(res, features["targets"],
                             hparams.bottleneck_warmup_steps // 2, is_training)


@registry.register_hparams
def basic_fc_small():
  """Small fully connected model."""
  hparams = common_hparams.basic_params1()
  hparams.learning_rate = 0.1
  hparams.batch_size = 128
  hparams.hidden_size = 256
  hparams.num_hidden_layers = 2
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.dropout = 0.0
  return hparams


@registry.register_hparams
def basic_autoencoder():
  """Basic autoencoder model."""
  hparams = common_hparams.basic_params1()
  hparams.optimizer = "Adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.05
  hparams.batch_size = 128
  hparams.hidden_size = 64
  hparams.num_hidden_layers = 4
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.kernel_height = 4
  hparams.kernel_width = 4
  hparams.dropout = 0.1
  hparams.add_hparam("bottleneck_size", 128)
  hparams.add_hparam("bottleneck_warmup_steps", 3000)
  return hparams
