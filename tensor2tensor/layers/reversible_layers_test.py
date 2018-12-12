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

"""Tests for reversible layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.layers import reversible_layers as reversible

import tensorflow as tf


class ReversibleLayersTest(tf.test.TestCase):

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testActNorm(self):
    np.random.seed(83243)
    batch_size = 25
    length = 15
    channels = 4
    inputs = 3. + 0.8 * np.random.randn(batch_size, length, channels)
    inputs = tf.cast(inputs, tf.float32)
    layer = reversible.ActNorm()
    outputs = layer(inputs)
    mean, variance = tf.nn.moments(outputs, axes=[0, 1])
    self.evaluate(tf.global_variables_initializer())
    mean_val, variance_val = self.evaluate([mean, variance])
    self.assertAllClose(mean_val, np.zeros(channels), atol=1e-3)
    self.assertAllClose(variance_val, np.ones(channels), atol=1e-3)

    inputs = 3. + 0.8 * np.random.randn(batch_size, length, channels)
    inputs = tf.cast(inputs, tf.float32)
    outputs = layer(inputs)
    mean, variance = tf.nn.moments(outputs, axes=[0, 1])
    self.evaluate(tf.global_variables_initializer())
    mean_val, variance_val = self.evaluate([mean, variance])
    self.assertAllClose(mean_val, np.zeros(channels), atol=0.25)
    self.assertAllClose(variance_val, np.ones(channels), atol=0.25)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testMADELeftToRight(self):
    np.random.seed(83243)
    batch_size = 2
    length = 3
    network = reversible.MADE([4], activation=tf.nn.relu)
    inputs = tf.zeros([batch_size, length])
    outputs = network(inputs)

    num_weights = sum([np.prod(weight.shape) for weight in network.weights])
    self.assertLen(network.weights, 4)
    self.assertEqual(num_weights, (3*4 + 4) + (4*3*2 + 3*2))

    self.evaluate(tf.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertAllEqual(outputs_val[:, 0], tf.zeros(batch_size))
    self.assertEqual(outputs_val.shape, (batch_size, 2 * length))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testMADERightToLeft(self):
    np.random.seed(1328)
    batch_size = 2
    length = 3
    network = reversible.MADE([4, 3],
                              input_order='right-to-left',
                              activation=tf.nn.relu,
                              use_bias=False)
    inputs = tf.zeros([batch_size, length])
    outputs = network(inputs)

    num_weights = sum([np.prod(weight.shape) for weight in network.weights])
    self.assertLen(network.weights, 3)
    self.assertEqual(num_weights, 3*4 + 4*3 + 3*3*2)

    self.evaluate(tf.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertAllEqual(outputs_val[:, -1], tf.zeros(batch_size))
    self.assertEqual(outputs_val.shape, (batch_size, 2 * length))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testMADENoHidden(self):
    np.random.seed(532)
    batch_size = 2
    length = 3
    network = reversible.MADE([], input_order='left-to-right')
    inputs = tf.zeros([batch_size, length])
    outputs = network(inputs)

    num_weights = sum([np.prod(weight.shape) for weight in network.weights])
    self.assertLen(network.weights, 2)
    self.assertEqual(num_weights, 3*3*2 + 3*2)

    self.evaluate(tf.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertAllEqual(outputs_val[:, 0], tf.zeros(batch_size))
    self.assertEqual(outputs_val.shape, (batch_size, 2 * length))


if __name__ == '__main__':
  tf.test.main()
