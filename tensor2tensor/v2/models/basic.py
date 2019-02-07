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

"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin.tf


@gin.configurable(whitelist=["num_hidden_layers", "hidden_size", "dropout"])
class BasicFcRelu(tf.keras.Model):
  """Basic fully-connected + ReLU model."""

  def __init__(self, features_info=None, input_names=None, target_names=None,
               num_hidden_layers=2, hidden_size=64, dropout=0.1):
    super(BasicFcRelu, self).__init__()
    self._input_name = input_names[0]
    input_shape = features_info[self._input_name].shape
    num_output_classes = features_info[target_names[0]].num_classes
    self._num_hidden_layers = num_hidden_layers
    self._dense_layers = []
    self._dropout_layers = []

    # Now the model.
    self._flatten_layer = tf.keras.layers.Flatten(input_shape=input_shape)
    for i in range(num_hidden_layers):
      self._dense_layers.append(tf.keras.layers.Dense(
          hidden_size, activation="relu", name="layer_%d" % i))
      self._dropout_layers.append(tf.keras.layers.Dropout(
          rate=dropout))
    self._logits = tf.keras.layers.Dense(
        num_output_classes, activation="softmax")

  def call(self, inputs, training=False):
    x = tf.cast(inputs[self._input_name], tf.float32) / 255.0
    x = self._flatten_layer(x)
    for i in range(self._num_hidden_layers):
      x = self._dense_layers[i](x)
      x = self._dropout_layers[i](x, training=training)
    return self._logits(x)


def basic_fc_large():
  """Large set of parameters for this model."""
  gin.bind_parameter("BasicFcRelu.num_hidden_layers", 3)
  gin.bind_parameter("BasicFcRelu.hidden_size", 128)
  gin.bind_parameter("BasicFcRelu.dropout", 0.3)
  return BasicFcRelu


# TODO(lukaszkaiser): could we allow coding like this? it's much easier!
# This will run fine, but not train as new layers are made in each step!
@gin.configurable(whitelist=["num_hidden_layers", "hidden_size", "dropout"])
class BasicFcReluV2(tf.keras.Model):
  """Basic fully-connected + ReLU model, nicer code version."""

  def __init__(self, features_info=None, input_names=None, target_names=None,
               num_hidden_layers=2, hidden_size=64, dropout=0.1):
    super(BasicFcReluV2, self).__init__()
    self._input_name = input_names[0]
    self._input_shape = features_info[self._input_name].shape
    self._num_output_classes = features_info[target_names[0]].num_classes
    self._num_hidden_layers = num_hidden_layers
    self._dropout = dropout
    self._hidden_size = hidden_size

  def call(self, inputs, training=False):
    x = tf.cast(inputs[self._input_name], tf.float32) / 255.0
    x = tf.keras.layers.Flatten(
        input_shape=self._input_shape)(x)
    for i in range(self._num_hidden_layers):
      x = tf.keras.layers.Dense(
          self._hidden_size, activation="relu", name="layer_%d" % i)(x)
      x = tf.keras.layers.Dropout(rate=self._dropout)(x, training=training)
    return tf.keras.layers.Dense(
        self._num_output_classes, activation="softmax")(x)
