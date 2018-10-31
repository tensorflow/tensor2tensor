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

"""Tests for common Bayes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensor2tensor.layers import bayes

import tensorflow as tf


class BayesTest(parameterized.TestCase, tf.test.TestCase):

  # TODO(trandustin): Remove the hack in the code, or re-enable once T2T drops
  # support for TF 1.10
  # @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDenseReparameterizationKernel(self):
    inputs = tf.to_float(np.random.rand(5, 3, 12))
    layer = bayes.DenseReparameterization(4, activation=tf.nn.relu)
    outputs1 = layer(inputs)
    outputs2 = layer(inputs)
    self.evaluate(tf.global_variables_initializer())
    # res1, res2 = self.evaluate([outputs1, outputs2])
    res1, _ = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 4))
    self.assertAllGreaterEqual(res1, 0.)
    # self.assertNotAllClose(res1, res2)

  # TODO(trandustin): Remove the hack in the code, or re-enable once T2T drops
  # support for TF 1.10
  # @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDenseReparameterizationBias(self):
    inputs = tf.to_float(np.random.rand(5, 3, 12))
    layer = bayes.DenseReparameterization(4, kernel_initializer="zero",
                                          bias_initializer=None,
                                          activation=tf.nn.relu)
    outputs1 = layer(inputs)
    outputs2 = layer(inputs)
    self.evaluate(tf.global_variables_initializer())
    # res1, res2 = self.evaluate([outputs1, outputs2])
    res1, _ = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 4))
    self.assertAllGreaterEqual(res1, 0.)
    # self.assertNotAllClose(res1, res2)

  # TODO(trandustin): Remove the hack in the code, or re-enable once T2T drops
  # support for TF 1.10
  # @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDenseReparameterizationDeterministic(self):
    inputs = tf.to_float(np.random.rand(5, 3, 12))
    layer = bayes.DenseReparameterization(4, kernel_initializer="zero",
                                          bias_initializer="zero",
                                          activation=tf.nn.relu)
    outputs1 = layer(inputs)
    outputs2 = layer(inputs)
    self.evaluate(tf.global_variables_initializer())
    # res1, res2 = self.evaluate([outputs1, outputs2])
    res1, _ = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 4))
    self.assertAllGreaterEqual(res1, 0.)
    # self.assertAllClose(res1, res2)

  # TODO(trandustin): Remove the hack in the code, or re-enable once T2T drops
  # support for TF 1.10
  # @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDenseReparameterizationModel(self):
    inputs = tf.to_float(np.random.rand(3, 4, 4, 1))
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3,
                               kernel_size=2,
                               padding="SAME",
                               activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        bayes.DenseReparameterization(2, activation=None),
    ])
    outputs = model(inputs)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(outputs)
    self.assertEqual(res.shape, (3, 2))
    self.assertLen(model.losses, 1)


if __name__ == "__main__":
  tf.test.main()

