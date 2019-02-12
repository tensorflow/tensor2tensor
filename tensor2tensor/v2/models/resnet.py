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

from tensor2tensor.models import resnet
from tensor2tensor.v2 import keras_utils
import tensorflow as tf
import gin.tf


@gin.configurable(whitelist=["layer_sizes", "filter_sizes"])
class Resnet(tf.keras.Model):
  """Resnet."""

  def __init__(self, features_info=None, input_names=None, target_names=None,
               layer_sizes=None, filter_sizes=None):
    super(Resnet, self).__init__()
    # Base config for resnet-50.
    if layer_sizes is None:
      layer_sizes = [3, 4, 6, 3]
    if filter_sizes is None:
      filter_sizes = [64, 64, 128, 256, 512]
    self._input_name = input_names[0]
    num_output_classes = features_info[target_names[0]].num_classes

    # Now the model.
    def resnet_model(inputs, training):
      return resnet.resnet_v2(
          inputs,
          resnet.bottleneck_block,
          layer_sizes,
          filter_sizes,
          is_training=training,
          is_cifar=True)

    self._resnet = keras_utils.FunctionLayer(resnet_model)
    self._logits = tf.keras.layers.Dense(
        num_output_classes, activation=None)

  def call(self, inputs, training=False):
    x = tf.cast(inputs[self._input_name], tf.float32) / 255.0
    x = self._resnet(x, training)
    x = tf.reduce_mean(x, axis=[1, 2])
    return self._logits(x)
