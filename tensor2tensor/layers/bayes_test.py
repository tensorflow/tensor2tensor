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

"""Tests for Bayesian neural network layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensor2tensor.layers import bayes
from tensor2tensor.utils import test_utils

import tensorflow as tf
import tensorflow_probability as tfp
ed = tfp.edward2
tf.compat.v1.enable_eager_execution()


class BayesTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {"layer": bayes.Conv2DFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": bayes.Conv2DFlipout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": bayes.Conv2DFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": bayes.Conv2DHierarchical,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": bayes.Conv2DHierarchical,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": bayes.Conv2DHierarchical,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": bayes.Conv2DReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": bayes.Conv2DReparameterization,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": bayes.Conv2DReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": bayes.Conv2DVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": bayes.Conv2DVariationalDropout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": bayes.Conv2DVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  @test_utils.run_in_graph_and_eager_modes
  def testConv2DKernel(self,
                       layer,
                       kernel_initializer,
                       bias_initializer,
                       all_close):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = tf.to_float(np.random.rand(5, 4, 4, 12))
    model = layer(4,
                  kernel_size=2,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  activation=tf.nn.relu)
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.evaluate(tf.global_variables_initializer())
    res1, res2 = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 3, 4))
    self.assertAllGreaterEqual(res1, 0.)
    if all_close:
      self.assertAllClose(res1, res2)
    else:
      self.assertNotAllClose(res1, res2)
    model.get_config()

  @parameterized.parameters(
      {"layer": bayes.Conv2DFlipout},
      {"layer": bayes.Conv2DHierarchical},
      {"layer": bayes.Conv2DReparameterization},
      {"layer": bayes.Conv2DVariationalDropout},
  )
  @test_utils.run_in_graph_and_eager_modes()
  def testConv2DModel(self, layer):
    inputs = tf.to_float(np.random.rand(3, 4, 4, 1))
    model = tf.keras.Sequential([
        layer(3, kernel_size=2, padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(outputs)
    self.assertEqual(res.shape, (3, 2))
    if layer == bayes.Conv2DHierarchical:
      self.assertLen(model.losses, 3)
    else:
      self.assertLen(model.losses, 1)

  @test_utils.run_in_graph_and_eager_modes
  def testTrainableNormalStddevConstraint(self):
    layer = bayes.DenseReparameterization(
        100, kernel_initializer="trainable_normal")
    inputs = tf.random_normal([1, 1])
    out = layer(inputs)
    stddev = layer.kernel.distribution.stddev()
    self.evaluate(tf.global_variables_initializer())
    res, _ = self.evaluate([stddev, out])
    self.assertAllGreater(res, 0.)

  @parameterized.parameters(
      {"layer": bayes.DenseDVI,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": bayes.DenseDVI,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": bayes.DenseDVI,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": bayes.DenseFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": bayes.DenseFlipout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": bayes.DenseFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": bayes.DenseReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": bayes.DenseReparameterization,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": bayes.DenseReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": bayes.DenseVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": bayes.DenseVariationalDropout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": bayes.DenseVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  @test_utils.run_in_graph_and_eager_modes
  def testDenseKernel(self,
                      layer,
                      kernel_initializer,
                      bias_initializer,
                      all_close):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = tf.to_float(np.random.rand(5, 3, 12))
    model = layer(4,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  activation=tf.nn.relu)
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.evaluate(tf.global_variables_initializer())
    res1, res2 = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 4))
    if layer != bayes.DenseDVI:
      self.assertAllGreaterEqual(res1, 0.)
    if all_close:
      self.assertAllClose(res1, res2)
    else:
      self.assertNotAllClose(res1, res2)
    model.get_config()

  @parameterized.parameters(
      {"layer": bayes.DenseDVI},
      {"layer": bayes.DenseFlipout},
      {"layer": bayes.DenseReparameterization},
      {"layer": bayes.DenseVariationalDropout},
  )
  @test_utils.run_in_graph_and_eager_modes
  def testDenseMean(self, layer):
    """Tests that forward pass can use other values, e.g., posterior mean."""
    tf.keras.backend.set_learning_phase(0)  # test time
    def take_mean(f, *args, **kwargs):
      """Sets random variable value to its mean."""
      rv = f(*args, **kwargs)
      rv._value = rv.distribution.mean()
      return rv
    inputs = tf.to_float(np.random.rand(5, 3, 7))
    model = layer(4, activation=tf.nn.relu, use_bias=False)
    outputs1 = model(inputs)
    with tfp.edward2.interception(take_mean):
      outputs2 = model(inputs)
    self.evaluate(tf.global_variables_initializer())
    res1, res2 = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 4))
    self.assertNotAllClose(res1, res2)
    if layer != bayes.DenseDVI:
      self.assertAllClose(res2, np.zeros((5, 3, 4)), atol=1e-4)

  @parameterized.parameters(
      {"layer": bayes.DenseDVI},
      {"layer": bayes.DenseFlipout},
      {"layer": bayes.DenseReparameterization},
      {"layer": bayes.DenseVariationalDropout},
      {"layer": bayes.DenseHierarchical},
  )
  @test_utils.run_in_graph_and_eager_modes()
  def testDenseLoss(self, layer):
    tf.keras.backend.set_learning_phase(1)  # training time
    features = tf.to_float(np.random.rand(5, 12))
    labels = tf.to_float(np.random.rand(5, 10))
    model = layer(10)

    # Imagine this is the 1st epoch.
    with tf.GradientTape(persistent=True) as tape:
      predictions = model(features)  # first call forces build
      model(features)  # ensure robustness after multiple calls
      nll = tf.losses.mean_squared_error(labels, predictions)
      kl = sum(model.losses)

    variables = [model.kernel_initializer.mean, model.kernel_initializer.stddev]
    for v in variables:
      self.assertIn(v, model.variables)

    # This will be fine, since the layer was built inside this tape, and thus
    # the distribution init ops were inside this tape.
    grads = tape.gradient(nll, variables)
    for grad in grads:
      self.assertIsNotNone(grad)
    grads = tape.gradient(kl, variables)
    for grad in grads:
      self.assertIsNotNone(grad)

    # Imagine this is the 2nd epoch.
    with tf.GradientTape(persistent=True) as tape:
      predictions = model(features)  # build is not called
      nll = tf.losses.mean_squared_error(labels, predictions)
      kl = sum(model.losses)

    variables = [model.kernel_initializer.mean, model.kernel_initializer.stddev]
    for v in variables:
      self.assertIn(v, model.variables)

    # This would fail, since the layer was built inside the tape from the 1st
    # epoch, and thus the distribution init ops were inside that tape instead of
    # this tape. By using a callable for the variable, this will no longer fail.
    grads = tape.gradient(nll, variables)
    for grad in grads:
      self.assertIsNotNone(grad)
    grads = tape.gradient(kl, variables)
    for grad in grads:
      self.assertIsNotNone(grad)

  @parameterized.parameters(
      {"layer": bayes.DenseDVI},
      {"layer": bayes.DenseFlipout},
      {"layer": bayes.DenseReparameterization},
      {"layer": bayes.DenseVariationalDropout},
      {"layer": bayes.DenseHierarchical},
  )
  @test_utils.run_in_graph_and_eager_modes()
  def testDenseModel(self, layer):
    inputs = tf.to_float(np.random.rand(3, 4, 4, 1))
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3,
                               kernel_size=2,
                               padding="SAME",
                               activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        layer(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(outputs)
    self.assertEqual(res.shape, (3, 2))
    if layer == bayes.DenseHierarchical:
      self.assertLen(model.losses, 3)
    else:
      self.assertLen(model.losses, 1)

  @parameterized.parameters(
      {"layer": bayes.DenseDVI},
      {"layer": bayes.DenseFlipout},
      {"layer": bayes.DenseReparameterization},
      {"layer": bayes.DenseVariationalDropout},
      {"layer": bayes.DenseHierarchical},
  )
  @test_utils.run_in_graph_and_eager_modes()
  def testDenseSubclass(self, layer):
    class DenseSubclass(layer):
      pass

    inputs = tf.to_float(np.random.rand(3, 4, 4, 1))
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3,
                               kernel_size=2,
                               padding="SAME",
                               activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        DenseSubclass(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(outputs)
    self.assertEqual(res.shape, (3, 2))
    if layer == bayes.DenseHierarchical:
      self.assertLen(model.losses, 3)
    else:
      self.assertLen(model.losses, 1)

  @test_utils.run_in_graph_and_eager_modes()
  def testDenseDVIIsDeterministic(self):
    """Tests that DenseDVI network has a deterministic loss function."""
    features = tf.to_float(np.random.rand(3, 2))
    labels = tf.to_float(np.random.rand(3, 1))
    model = tf.keras.Sequential([
        bayes.DenseDVI(5, activation=tf.nn.relu),
        bayes.DenseDVI(1, activation=None),
    ])
    outputs = model(features, training=True)
    nll = -tf.reduce_sum(outputs.distribution.log_prob(labels))
    kl = sum(model.losses)
    loss = nll + kl
    self.evaluate(tf.global_variables_initializer())
    res1 = self.evaluate(loss)
    res2 = self.evaluate(loss)
    self.assertEqual(res1, res2)

  @test_utils.run_in_graph_and_eager_modes()
  def testDenseDVIMoments(self):
    """Verifies DenseDVI's moments empirically with samples."""
    tf.set_random_seed(377269)
    batch_size = 3
    num_features = 5
    units = 128
    num_samples = 50000
    inputs = tf.to_float(np.random.rand(batch_size, num_features))
    layer = bayes.DenseDVI(units, activation=tf.nn.relu)

    outputs1 = layer(inputs)
    mean1 = outputs1.distribution.mean()
    covariance1 = outputs1.distribution.covariance()

    kernel_samples = layer.kernel.distribution.sample(num_samples)
    outputs2 = layer.activation(
        tf.einsum("bd,sdu->sbu", inputs, kernel_samples) +
        tf.reshape(layer.bias, [1, 1, units]))
    mean2 = tf.reduce_mean(outputs2, axis=0)
    centered_outputs2 = tf.transpose(outputs2 - mean2, [1, 2, 0])
    covariance2 = tf.matmul(centered_outputs2,
                            centered_outputs2,
                            transpose_b=True) / float(num_samples)

    self.evaluate(tf.global_variables_initializer())
    mean1_val, covariance1_val, mean2_val, covariance2_val = self.evaluate(
        [mean1, covariance1, mean2, covariance2])
    # Check % of mismatches is not too high according to heuristic thresholds.
    num_mismatches = np.sum(np.abs(mean1_val - mean2_val) > 5e-3)
    percent_mismatches = num_mismatches / float(batch_size * units)
    self.assertLessEqual(percent_mismatches, 0.05)
    num_mismatches = np.sum(np.abs(covariance1_val - covariance2_val) > 5e-3)
    percent_mismatches = num_mismatches / float(batch_size * units * units)
    self.assertLessEqual(percent_mismatches, 0.05)

  @parameterized.parameters(
      {"lstm_cell": bayes.LSTMCellFlipout,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "all_close": True},
      {"lstm_cell": bayes.LSTMCellFlipout,
       "kernel_initializer": "trainable_normal",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"lstm_cell": bayes.LSTMCellFlipout,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"lstm_cell": bayes.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "all_close": True},
      {"lstm_cell": bayes.LSTMCellReparameterization,
       "kernel_initializer": "trainable_normal",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"lstm_cell": bayes.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"lstm_cell": bayes.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  @test_utils.run_in_graph_and_eager_modes
  def testLSTMCell(self,
                   lstm_cell,
                   kernel_initializer,
                   recurrent_initializer,
                   bias_initializer,
                   all_close):
    batch_size, timesteps, dim = 5, 3, 12
    hidden_size = 10
    inputs = tf.to_float(np.random.rand(batch_size, timesteps, dim))
    cell = lstm_cell(hidden_size,
                     kernel_initializer=kernel_initializer,
                     recurrent_initializer=recurrent_initializer,
                     bias_initializer=bias_initializer)
    noise = tf.to_float(np.random.rand(1, hidden_size))
    h0, c0 = cell.get_initial_state(inputs)
    state = (h0 + noise, c0)
    outputs1, _ = cell(inputs[:, 0, :], state)
    outputs2, _ = cell(inputs[:, 0, :], state)
    cell.call_weights()
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

  @parameterized.parameters(
      {"lstm_cell": bayes.LSTMCellFlipout},
      {"lstm_cell": bayes.LSTMCellReparameterization},
  )
  @test_utils.run_in_graph_and_eager_modes()
  def testLSTMCellLoss(self, lstm_cell):
    features = tf.to_float(np.random.rand(5, 1, 12))
    labels = tf.to_float(np.random.rand(5, 10))
    cell = lstm_cell(10)
    state = (tf.zeros([1, 10]), tf.zeros([1, 10]))

    # Imagine this is the 1st epoch.
    with tf.GradientTape(persistent=True) as tape:
      predictions, _ = cell(features[:, 0, :], state)  # first call forces build
      cell(features[:, 0, :], state)  # ensure robustness after multiple calls
      cell.get_initial_state(features[:, 0, :])
      cell(features[:, 0, :], state)  # ensure robustness after multiple calls
      nll = tf.losses.mean_squared_error(labels, predictions)
      kl = sum(cell.losses)

    variables = [
        cell.kernel_initializer.mean, cell.kernel_initializer.stddev,
        cell.recurrent_initializer.mean, cell.recurrent_initializer.stddev,
    ]
    for v in variables:
      self.assertIn(v, cell.variables)

    # This will be fine, since the layer was built inside this tape, and thus
    # the distribution init ops were inside this tape.
    grads = tape.gradient(nll, variables)
    for grad in grads:
      self.assertIsNotNone(grad)
    grads = tape.gradient(kl, variables)
    for grad in grads:
      self.assertIsNotNone(grad)

    # Imagine this is the 2nd epoch.
    with tf.GradientTape(persistent=True) as tape:
      cell.get_initial_state(features[:, 0, :])
      predictions, _ = cell(features[:, 0, :], state)  # build is not called
      nll = tf.losses.mean_squared_error(labels, predictions)
      kl = sum(cell.losses)

    variables = [
        cell.kernel_initializer.mean, cell.kernel_initializer.stddev,
        cell.recurrent_initializer.mean, cell.recurrent_initializer.stddev,
    ]
    for v in variables:
      self.assertIn(v, cell.variables)

    # This would fail, since the layer was built inside the tape from the 1st
    # epoch, and thus the distribution init ops were inside that tape instead of
    # this tape. By using a callable for the variable, this will no longer fail.
    grads = tape.gradient(nll, variables)
    for grad in grads:
      self.assertIsNotNone(grad)
    grads = tape.gradient(kl, variables)
    for grad in grads:
      self.assertIsNotNone(grad)

  @parameterized.parameters(
      {"lstm_cell": bayes.LSTMCellFlipout},
      {"lstm_cell": bayes.LSTMCellReparameterization},
  )
  @test_utils.run_in_graph_and_eager_modes()
  def testLSTMCellModel(self, lstm_cell):
    batch_size, timesteps, dim = 5, 3, 12
    hidden_size = 10
    inputs = tf.to_float(np.random.rand(batch_size, timesteps, dim))
    cell = lstm_cell(hidden_size)
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
    # NOTE: `cell.call_weights` should have been called at the beginning of
    # each call, so these should be different.
    self.assertNotAllClose(res1, res2)
    # NOTE: We didn't call `cell.call_weights` again before computing
    # `outputs3`, so the cell should have had the same weights as it did
    # during computation of `outputs2`, and thus yielded the same output
    # tensor.
    self.assertAllClose(res2, res3)
    self.assertLen(model.losses, 2)

  @test_utils.run_in_graph_and_eager_modes()
  def testNCPNormalPerturb(self):
    batch_size = 3
    inputs = tf.to_float(np.random.rand(batch_size, 4))
    model = bayes.NCPNormalPerturb()
    outputs = model(inputs)
    inputs_val, outputs_val = self.evaluate([inputs, outputs])
    self.assertEqual(outputs_val.shape, (2 * batch_size, 4))
    self.assertAllEqual(inputs_val, outputs_val[:batch_size])

  @test_utils.run_in_graph_and_eager_modes()
  def testNCPCategoricalPerturb(self):
    input_dim = 5
    batch_size = 3
    inputs = tf.to_float(np.random.choice(input_dim, size=(batch_size, 4)))
    model = bayes.NCPCategoricalPerturb(input_dim)
    outputs = model(inputs)
    inputs_val, outputs_val = self.evaluate([inputs, outputs])
    self.assertEqual(outputs_val.shape, (2 * batch_size, 4))
    self.assertAllEqual(inputs_val, outputs_val[:batch_size])

  @test_utils.run_in_graph_and_eager_modes()
  def testNCPNormalOutput(self):
    batch_size = 3
    features = ed.Normal(loc=tf.random.normal([2 * batch_size, 1]), scale=1.)
    labels = tf.to_float(np.random.rand(batch_size))
    model = bayes.NCPNormalOutput(mean=labels)
    predictions = model(features)
    features_val, predictions_val = self.evaluate([features, predictions])
    self.assertLen(model.losses, 1)
    self.assertAllEqual(features_val[:batch_size], predictions_val)

  @test_utils.run_in_graph_and_eager_modes()
  def testMixtureLogistic(self):
    batch_size = 3
    features = tf.to_float(np.random.rand(batch_size, 4))
    labels = tf.to_float(np.random.rand(batch_size))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation=None),
        bayes.MixtureLogistic(5),
    ])
    outputs = model(features)
    log_likelihood = tf.reduce_sum(outputs.distribution.log_prob(labels))
    self.evaluate(tf.global_variables_initializer())
    log_likelihood_val, outputs_val = self.evaluate([log_likelihood, outputs])
    self.assertEqual(log_likelihood_val.shape, ())
    self.assertLessEqual(log_likelihood_val, 0.)
    self.assertEqual(outputs_val.shape, (batch_size,))


if __name__ == "__main__":
  tf.test.main()
