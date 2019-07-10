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

"""Tests for Gaussian process layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.layers import gaussian_process
from tensor2tensor.utils import test_utils

import tensorflow as tf
tf.compat.v1.enable_eager_execution()


class GaussianProcessTest(tf.test.TestCase):

  @test_utils.run_in_graph_and_eager_modes()
  def testGaussianProcessPosterior(self):
    train_batch_size = 3
    test_batch_size = 2
    input_dim = 4
    output_dim = 5
    features = tf.to_float(np.random.rand(train_batch_size, input_dim))
    labels = tf.to_float(np.random.rand(train_batch_size, output_dim))
    layer = gaussian_process.GaussianProcess(output_dim,
                                             conditional_inputs=features,
                                             conditional_outputs=labels)
    test_features = tf.to_float(np.random.rand(test_batch_size, input_dim))
    test_labels = tf.to_float(np.random.rand(test_batch_size, output_dim))
    test_outputs = layer(test_features)
    test_nats = -test_outputs.distribution.log_prob(test_labels)
    self.evaluate(tf.global_variables_initializer())
    test_nats_val, outputs_val = self.evaluate([test_nats, test_outputs])
    self.assertEqual(test_nats_val.shape, ())
    self.assertGreaterEqual(test_nats_val, 0.)
    self.assertEqual(outputs_val.shape, (test_batch_size, output_dim))

  @test_utils.run_in_graph_and_eager_modes()
  def testGaussianProcessPrior(self):
    batch_size = 3
    input_dim = 4
    output_dim = 5
    features = tf.to_float(np.random.rand(batch_size, input_dim))
    labels = tf.to_float(np.random.rand(batch_size, output_dim))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation=None),
        gaussian_process.GaussianProcess(output_dim),
    ])
    outputs = model(features)
    log_prob = outputs.distribution.log_prob(labels)
    self.evaluate(tf.global_variables_initializer())
    log_prob_val, outputs_val = self.evaluate([log_prob, outputs])
    self.assertEqual(log_prob_val.shape, ())
    self.assertLessEqual(log_prob_val, 0.)
    self.assertEqual(outputs_val.shape, (batch_size, output_dim))

  @test_utils.run_in_graph_and_eager_modes()
  def testSparseGaussianProcess(self):
    dataset_size = 10
    batch_size = 3
    input_dim = 4
    output_dim = 5
    features = tf.to_float(np.random.rand(batch_size, input_dim))
    labels = tf.to_float(np.random.rand(batch_size, output_dim))
    model = gaussian_process.SparseGaussianProcess(output_dim, num_inducing=2)
    with tf.GradientTape() as tape:
      predictions = model(features)
      nll = -tf.reduce_mean(predictions.distribution.log_prob(labels))
      kl = sum(model.losses) / dataset_size
      loss = nll + kl

    self.evaluate(tf.global_variables_initializer())
    grads = tape.gradient(nll, model.variables)
    for grad in grads:
      self.assertIsNotNone(grad)

    loss_val, predictions_val = self.evaluate([loss, predictions])
    self.assertEqual(loss_val.shape, ())
    self.assertGreaterEqual(loss_val, 0.)
    self.assertEqual(predictions_val.shape, (batch_size, output_dim))

  @test_utils.run_in_graph_and_eager_modes()
  def testBayesianLinearModel(self):
    """Tests that model makes reasonable predictions."""
    np.random.seed(42)
    train_batch_size = 5
    test_batch_size = 2
    num_features = 3
    noise_variance = 0.01
    coeffs = tf.range(num_features, dtype=tf.float32)
    features = tf.to_float(np.random.randn(train_batch_size, num_features))
    labels = (tf.tensordot(features, coeffs, [[-1], [0]])
              + noise_variance * tf.to_float(np.random.randn(train_batch_size)))

    model = gaussian_process.BayesianLinearModel(noise_variance=noise_variance)
    model.fit(features, labels)

    test_features = tf.to_float(np.random.randn(test_batch_size, num_features))
    test_labels = tf.tensordot(test_features, coeffs, [[-1], [0]])
    outputs = model(test_features)
    test_predictions = outputs.distribution.mean()
    test_predictions_variance = outputs.distribution.variance()

    [
        test_labels_val, test_predictions_val, test_predictions_variance_val,
    ] = self.evaluate(
        [test_labels, test_predictions, test_predictions_variance])
    self.assertEqual(test_predictions_val.shape, (test_batch_size,))
    self.assertEqual(test_predictions_variance_val.shape, (test_batch_size,))
    self.assertAllClose(test_predictions_val, test_labels_val, atol=0.1)
    self.assertAllLessEqual(test_predictions_variance_val, noise_variance)


if __name__ == "__main__":
  tf.test.main()
