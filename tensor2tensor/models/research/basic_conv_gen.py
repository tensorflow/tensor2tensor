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
class BasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    hparams = self.hparams
    filters = hparams.hidden_size
    kernel1, kernel2 = (3, 3), (4, 4)

    # Concat frames and down-stride.
    cur_frame = tf.to_float(features["inputs"])
    prev_frame = tf.to_float(features["inputs_prev"])
    x = tf.concat([cur_frame, prev_frame], axis=-1)
    for _ in xrange(hparams.num_compress_steps):
      x = tf.layers.conv2d(x, filters, kernel2, activation=common_layers.belu,
                           strides=(2, 2), padding="SAME")
      x = common_layers.layer_norm(x)
      filters *= 2
    # Add embedded action.
    action = tf.reshape(features["action"], [-1, 1, 1, hparams.hidden_size])
    zeros = tf.zeros(common_layers.shape_list(x)[:-1] + [hparams.hidden_size])
    x = tf.concat([x, action + zeros], axis=-1)

    # Run a stack of convolutions.
    for i in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer%d" % i):
        y = tf.layers.conv2d(x, filters, kernel1, activation=common_layers.belu,
                             strides=(1, 1), padding="SAME")
        if i == 0:
          x = y
        else:
          x = common_layers.layer_norm(x + y)
    # Up-convolve.
    for _ in xrange(hparams.num_compress_steps):
      filters //= 2
      x = tf.layers.conv2d_transpose(
          x, filters, kernel2, activation=common_layers.belu,
          strides=(2, 2), padding="SAME")
      x = common_layers.layer_norm(x)

    # Reward prediction.
    reward_pred_h1 = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    # Rewards are {-1, 0, 1} so we add 1 to the raw gold ones, predict 3.
    reward_pred = tf.layers.dense(reward_pred_h1, 3, name="reward")
    reward_gold = tf.expand_dims(tf.to_int32(features["reward_raw"]) + 1, 1)
    reward_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=reward_gold, logits=reward_pred, name="reward_loss")
    reward_loss = tf.reduce_mean(reward_loss)
    return x, {"reward": reward_loss}


@registry.register_hparams
def basic_conv():
  """Basic 2-frame conv model."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 64
  hparams.batch_size = 8
  hparams.num_hidden_layers = 3
  hparams.optimizer = "Adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.05
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.add_hparam("num_compress_steps", 2)
  return hparams


@registry.register_hparams
def basic_conv_small():
  """Small conv model."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 32
  return hparams
