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

  @tf.contrib.eager.run_test_in_graph_and_eager_modes
  def testTrainableNormalStddevConstraint(self):
    layer = bayes.DenseReparameterization(
        100, kernel_initializer=bayes.TrainableNormal())
    inputs = tf.random_normal([1, 1])
    out = layer(inputs)
    stddev = layer.kernel.distribution.scale
    self.evaluate(tf.global_variables_initializer())
    res, _ = self.evaluate([stddev, out])
    self.assertAllGreater(res, 0.)

  @parameterized.named_parameters(
      {"testcase_name": "_no_uncertainty", "kernel_initializer": "zeros",
       "bias_initializer": "zeros", "all_close": True},
      {"testcase_name": "_kernel_uncertainty", "kernel_initializer": None,
       "bias_initializer": "zeros", "all_close": False},
      {"testcase_name": "_bias_uncertainty", "kernel_initializer": "zeros",
       "bias_initializer": None, "all_close": False},
  )
  @tf.contrib.eager.run_test_in_graph_and_eager_modes
  def testDenseReparameterizationKernel(
      self, kernel_initializer, bias_initializer, all_close):
    inputs = tf.to_float(np.random.rand(5, 3, 12))
    layer = bayes.DenseReparameterization(
        4, kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer, activation=tf.nn.relu)
    outputs1 = layer(inputs)
    outputs2 = layer(inputs)
    self.evaluate(tf.global_variables_initializer())
    res1, res2 = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 4))
    self.assertAllGreaterEqual(res1, 0.)
    if all_close:
      self.assertAllClose(res1, res2)
    else:
      self.assertNotAllClose(res1, res2)
    layer.get_config()

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
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

  @parameterized.named_parameters(
      {"testcase_name": "_no_uncertainty", "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal", "bias_initializer": "zeros",
       "all_close": True},
      {"testcase_name": "_kernel_uncertainty", "kernel_initializer": None,
       "recurrent_initializer": "orthogonal", "bias_initializer": "zeros",
       "all_close": False},
      {"testcase_name": "_recurrent_uncertainty", "kernel_initializer": "zeros",
       "recurrent_initializer": None, "bias_initializer": "zeros",
       "all_close": False},
      {"testcase_name": "_bias_uncertainty", "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal", "bias_initializer": None,
       "all_close": False},
  )
  @tf.contrib.eager.run_test_in_graph_and_eager_modes
  def testLSTMCellReparameterization(
      self, kernel_initializer, recurrent_initializer, bias_initializer,
      all_close):
    batch_size, timesteps, dim = 5, 3, 12
    hidden_size = 10
    inputs = tf.to_float(np.random.rand(batch_size, timesteps, dim))
    cell = bayes.LSTMCellReparameterization(
        hidden_size, kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer)
    noise = tf.to_float(np.random.rand(1, hidden_size))
    h0, c0 = cell.get_initial_state(inputs)
    state = (h0 + noise, c0)
    outputs1, _ = cell(inputs[:, 0, :], state)
    outputs2, _ = cell(inputs[:, 0, :], state)
    cell.sample_weights()
    outputs3, _ = cell(inputs[:, 0, :], state)
    self.evaluate(tf.global_variables_initializer())
    res1, res2, res3 = self.evaluate([outputs1, outputs2, outputs3])
    self.assertEqual(res1.shape, (batch_size, hidden_size))
    self.assertAllClose(res1, res2)
    if all_close:
      self.assertAllClose(res1, res3)
    else:
      self.assertNotAllClose(res1, res3)
    cell.get_config()

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testLSTMCellReparameterizationModel(self):
    batch_size, timesteps, dim = 5, 3, 12
    hidden_size = 10
    inputs = tf.to_float(np.random.rand(batch_size, timesteps, dim))
    cell = bayes.LSTMCellReparameterization(hidden_size)
    model = tf.keras.Sequential([
        tf.keras.layers.RNN(cell, return_sequences=True)
    ])
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    state = (tf.zeros([1, hidden_size]), tf.zeros([1, hidden_size]))
    outputs3 = []
    for t in range(timesteps):
      out, state = cell(inputs[:, t, :], state)
      outputs3.append(out)
    outputs3 = tf.stack(outputs3, axis=1)
    self.evaluate(tf.global_variables_initializer())
    res1, res2, res3 = self.evaluate([outputs1, outputs2, outputs3])
    self.assertEqual(res1.shape, (batch_size, timesteps, hidden_size))
    self.assertEqual(res3.shape, (batch_size, timesteps, hidden_size))
    # NOTE: `cell.sample_weights` should have been called at the beginning of
    # each call, so these should be different.
    self.assertNotAllClose(res1, res2)
    # NOTE: We didn't call `cell.sample_weights` again before computing
    # `outputs3`, so the cell should have had the same weights as it did during
    # computation of `outputs2`, and thus yielded the same output tensor.
    self.assertAllClose(res2, res3)
    self.assertLen(model.losses, 2)


if __name__ == "__main__":
  tf.test.main()

