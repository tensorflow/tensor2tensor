
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
    print(features)
    filters = self.hparams.hidden_size
    cur_frame = tf.to_float(features["inputs"])
    prev_frame = tf.to_float(features["inputs_prev"])
    print(features["inputs"].shape, cur_frame.shape, prev_frame.shape)
    action = common_layers.embedding(tf.to_int64(features["action"]),
                                     10, filters)
    action = tf.reshape(action, [-1, 1, 1, filters])

    frames = tf.concat([cur_frame, prev_frame], axis=3)
    h1 = tf.layers.conv2d(frames, filters, kernel_size=(3, 3), padding="SAME")
    h2 = tf.layers.conv2d(tf.nn.relu(h1 + action), filters,
                          kernel_size=(5, 5), padding="SAME")
    res = tf.layers.conv2d(tf.nn.relu(h2 + action), 3 * 256,
                           kernel_size=(3, 3), padding="SAME")

    height = tf.shape(res)[1]
    width = tf.shape(res)[2]
    res = tf.reshape(res, [-1, height, width, 3, 256])
    return res


@registry.register_hparams
def basic_conv_small():
  # """Small conv model."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 32
  hparams.batch_size = 2
  return hparams
