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

"""Tests for reversible layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensor2tensor.layers import reversible_layers as reversible
from tensor2tensor.utils import test_utils

import tensorflow as tf
from tensorflow_probability import edward2 as ed
tf.compat.v1.enable_eager_execution()



class ReversibleLayersTest(parameterized.TestCase, tf.test.TestCase):

  @test_utils.run_in_graph_and_eager_modes()
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

  @test_utils.run_in_graph_and_eager_modes()
  def testMADELeftToRight(self):
    np.random.seed(83243)
    batch_size = 2
    length = 3
    channels = 1
    units = 5
    network = reversible.MADE(units, [4], activation=tf.nn.relu)
    inputs = tf.zeros([batch_size, length, channels])
    outputs = network(inputs)

    num_weights = sum([np.prod(weight.shape) for weight in network.weights])
    # Disable lint error for open-source. pylint: disable=g-generic-assert
    self.assertEqual(len(network.weights), 4)
    # pylint: enable=g-generic-assert
    self.assertEqual(num_weights, (3*1*4 + 4) + (4*3*5 + 3*5))

    self.evaluate(tf.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertAllEqual(outputs_val[:, 0, :], np.zeros((batch_size, units)))
    self.assertEqual(outputs_val.shape, (batch_size, length, units))

  @test_utils.run_in_graph_and_eager_modes()
  def testMADERightToLeft(self):
    np.random.seed(1328)
    batch_size = 2
    length = 3
    channels = 5
    units = 1
    network = reversible.MADE(units, [4, 3],
                              input_order='right-to-left',
                              activation=tf.nn.relu,
                              use_bias=False)
    inputs = tf.zeros([batch_size, length, channels])
    outputs = network(inputs)

    num_weights = sum([np.prod(weight.shape) for weight in network.weights])
    # Disable lint error for open-source. pylint: disable=g-generic-assert
    self.assertEqual(len(network.weights), 3)
    # pylint: enable=g-generic-assert
    self.assertEqual(num_weights, 3*5*4 + 4*3 + 3*3*1)

    self.evaluate(tf.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertAllEqual(outputs_val[:, -1, :], np.zeros((batch_size, units)))
    self.assertEqual(outputs_val.shape, (batch_size, length, units))

  @test_utils.run_in_graph_and_eager_modes()
  def testMADENoHidden(self):
    np.random.seed(532)
    batch_size = 2
    length = 3
    channels = 5
    units = 4
    network = reversible.MADE(units, [], input_order='left-to-right')
    inputs = tf.zeros([batch_size, length, channels])
    outputs = network(inputs)

    num_weights = sum([np.prod(weight.shape) for weight in network.weights])
    # Disable lint error for open-source. pylint: disable=g-generic-assert
    self.assertEqual(len(network.weights), 2)
    # pylint: enable=g-generic-assert
    self.assertEqual(num_weights, 3*5*3*4 + 3*4)

    self.evaluate(tf.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertAllEqual(outputs_val[:, 0, :], np.zeros((batch_size, units)))
    self.assertEqual(outputs_val.shape, (batch_size, length, units))

  @test_utils.run_in_graph_and_eager_modes()
  def testTransformedRandomVariable(self):
    class Exp(tf.keras.layers.Layer):
      """Exponential activation function for reversible networks."""

      def __call__(self, inputs, *args, **kwargs):
        if not isinstance(inputs, ed.RandomVariable):
          return super(Exp, self).__call__(inputs, *args, **kwargs)
        return reversible.TransformedRandomVariable(inputs, self)

      def call(self, inputs):
        return tf.exp(inputs)

      def reverse(self, inputs):
        return tf.log(inputs)

      def log_det_jacobian(self, inputs):
        return -tf.log(inputs)

    x = ed.Normal(0., 1.)
    y = Exp()(x)
    y_sample = self.evaluate(y.distribution.sample())
    y_log_prob = self.evaluate(y.distribution.log_prob(y_sample))
    self.assertGreater(y_sample, 0.)
    self.assertTrue(np.isfinite(y_log_prob))


if __name__ == '__main__':
  tf.test.main()
