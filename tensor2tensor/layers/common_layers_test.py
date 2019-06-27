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

"""Tests for common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import kfac
import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import test_utils

import tensorflow as tf

tf.compat.v1.enable_eager_execution()


class CommonLayersTest(parameterized.TestCase, tf.test.TestCase):

  @test_utils.run_in_graph_and_eager_modes()
  def testIndexLastDimWithIndices(self):
    x = np.array([[2., 3., 4., 5.],
                  [6., 7., 8., 9.]])
    indices = np.array([2, 0])
    x_idx = common_layers.index_last_dim_with_indices(x, indices)

    expected = np.array([4., 6.])
    self.assertAllEqual(expected, self.evaluate(x_idx))

  @test_utils.run_in_graph_and_eager_modes()
  def testSaturatingSigmoid(self):
    x = np.array([-120.0, -100.0, 0.0, 100.0, 120.0], dtype=np.float32)
    y = common_layers.saturating_sigmoid(tf.constant(x))
    res = self.evaluate(y)
    self.assertAllClose(res, [0.0, 0.0, 0.5, 1.0, 1.0])

  @test_utils.run_in_graph_and_eager_modes()
  def testFlatten4D3D(self):
    x = np.random.randint(1, high=9, size=(3, 5, 2))
    y = common_layers.flatten4d3d(common_layers.embedding(x, 10, 7))
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (3, 5 * 2, 7))

  @test_utils.run_in_graph_and_eager_modes()
  def testEmbedding(self):
    x = np.random.randint(1, high=9, size=(3, 5))
    y = common_layers.embedding(x, 10, 16)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (3, 5, 16))

  @test_utils.run_in_graph_mode_only()
  def testShakeShake(self):
    x = np.random.rand(5, 7)
    with self.test_session() as session:
      x = tf.constant(x, dtype=tf.float32)
      y = common_layers.shakeshake([x, x, x, x, x])
      inp, res = session.run([x, y])
    self.assertAllClose(res, inp)

  @test_utils.run_in_graph_and_eager_modes()
  def testConv(self):
    x = np.random.rand(5, 7, 1, 11)
    y = common_layers.conv(tf.constant(x, dtype=tf.float32), 13, (3, 1))
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 5, 1, 13))

  @test_utils.run_in_graph_and_eager_modes()
  def testConv1d(self):
    x = np.random.rand(5, 7, 11)
    y = common_layers.conv1d(tf.constant(x, dtype=tf.float32), 13, 1)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 7, 13))

  @test_utils.run_in_graph_and_eager_modes()
  def testSeparableConv(self):
    x = np.random.rand(5, 7, 1, 11)
    y = common_layers.separable_conv(
        tf.constant(x, dtype=tf.float32), 13, (3, 1))
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 5, 1, 13))

  @test_utils.run_in_graph_and_eager_modes()
  def testSubSeparableConv(self):
    for sep in [0, 1, 2, 4]:
      x = np.random.rand(5, 7, 1, 12)
      with tf.variable_scope("sep_%d" % sep):
        y = common_layers.subseparable_conv(
            tf.constant(x, dtype=tf.float32), 16, (3, 1), separability=sep)
      self.evaluate(tf.global_variables_initializer())
      res = self.evaluate(y)
      self.assertEqual(res.shape, (5, 5, 1, 16))

  @test_utils.run_in_graph_and_eager_modes()
  def testConvBlock(self):
    x = np.random.rand(5, 7, 1, 11)
    y = common_layers.conv_block(
        tf.constant(x, dtype=tf.float32),
        13, [(1, (3, 3)), (1, (3, 3))],
        padding="SAME",
        normalizer_fn=common_layers.noam_norm)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 7, 1, 13))

  @test_utils.run_in_graph_and_eager_modes()
  def testSeparableConvBlock(self):
    x = np.random.rand(5, 7, 1, 11)
    y = common_layers.separable_conv_block(
        tf.constant(x, dtype=tf.float32),
        13, [(1, (3, 3)), (1, (3, 3))],
        padding="SAME")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 7, 1, 13))

  @test_utils.run_in_graph_and_eager_modes()
  def testSubSeparableConvBlock(self):
    for sep in [0, 1, 2, 4]:
      x = np.random.rand(5, 7, 1, 12)
      with tf.variable_scope("sep_%d" % sep):
        y = common_layers.subseparable_conv_block(
            tf.constant(x, dtype=tf.float32),
            16, [(1, (3, 3)), (1, (3, 3))],
            padding="SAME",
            separability=sep)
      self.evaluate(tf.global_variables_initializer())
      res = self.evaluate(y)
      self.assertEqual(res.shape, (5, 7, 1, 16))

  @test_utils.run_in_graph_and_eager_modes()
  def testPool(self):
    x = np.random.rand(5, 8, 1, 11)
    y = common_layers.pool(
        tf.constant(x, dtype=tf.float32), (2, 2), "AVG", "SAME")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 8, 1, 11))

  @test_utils.run_in_graph_and_eager_modes()
  def testConvBlockDownsample(self):
    x = np.random.rand(5, 7, 1, 11)
    y = common_layers.conv_block_downsample(
        tf.constant(x, dtype=tf.float32), (3, 1), (2, 1), "SAME")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 4, 1, 27))

  @test_utils.run_in_graph_and_eager_modes()
  def testGetTimingSignal(self):
    length = 7
    num_timescales = 10
    a = common_layers.get_timing_signal(length, num_timescales=num_timescales)
    res = self.evaluate(a)
    self.assertEqual(res.shape, (length, 2 * num_timescales))

  @test_utils.run_in_graph_and_eager_modes()
  def testAddTimingSignal(self):
    batch = 5
    length = 7
    height = 3
    depth = 35
    x = np.random.rand(batch, length, height, depth)
    a = common_layers.add_timing_signal(tf.constant(x, dtype=tf.float32))
    res = self.evaluate(a)
    self.assertEqual(res.shape, (batch, length, height, depth))

  @test_utils.run_in_graph_and_eager_modes()
  def testConvGRU(self):
    x = np.random.rand(5, 7, 3, 11)
    y = common_layers.conv_gru(tf.constant(x, dtype=tf.float32), (1, 3), 11)
    z = common_layers.conv_gru(
        tf.constant(x, dtype=tf.float32), (1, 3), 11, padding="LEFT")
    self.evaluate(tf.global_variables_initializer())
    res1 = self.evaluate(y)
    res2 = self.evaluate(z)
    self.assertEqual(res1.shape, (5, 7, 3, 11))
    self.assertEqual(res2.shape, (5, 7, 3, 11))

  @test_utils.run_in_graph_mode_only
  def testSRU(self):
    x = np.random.rand(5, 7, 3, 11)
    with self.test_session() as session:
      y = common_layers.sru(tf.constant(x, dtype=tf.float32))
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 3, 11))

  @test_utils.run_in_graph_and_eager_modes()
  def testLayerNorm(self):
    x = np.random.rand(5, 7, 11)
    y = common_layers.layer_norm(tf.constant(x, dtype=tf.float32), 11)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 7, 11))


  @test_utils.run_in_graph_and_eager_modes()
  def testGroupNorm(self):
    x = np.random.rand(5, 7, 3, 16)
    y = common_layers.group_norm(tf.constant(x, dtype=tf.float32))
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 7, 3, 16))

  @test_utils.run_in_graph_and_eager_modes()
  def testConvLSTM(self):
    x = np.random.rand(5, 7, 11, 13)
    y = common_layers.conv_lstm(tf.constant(x, dtype=tf.float32), (1, 3), 13)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 7, 11, 13))

  @test_utils.run_in_graph_and_eager_modes()
  def testPadToSameLength(self):
    x1 = np.random.rand(5, 7, 11)
    x2 = np.random.rand(5, 9, 11)
    a, b = common_layers.pad_to_same_length(
        tf.constant(x1, dtype=tf.float32), tf.constant(x2, dtype=tf.float32))
    c, d = common_layers.pad_to_same_length(
        tf.constant(x1, dtype=tf.float32),
        tf.constant(x2, dtype=tf.float32),
        final_length_divisible_by=4)
    res1, res2 = self.evaluate([a, b])
    res1a, res2a = self.evaluate([c, d])
    self.assertEqual(res1.shape, (5, 9, 11))
    self.assertEqual(res2.shape, (5, 9, 11))
    self.assertEqual(res1a.shape, (5, 12, 11))
    self.assertEqual(res2a.shape, (5, 12, 11))

  @test_utils.run_in_graph_and_eager_modes()
  def testShiftLeft(self):
    x1 = np.zeros((5, 7, 1, 11))
    x1[:, 0, :] = np.ones_like(x1[:, 0, :])
    expected = np.zeros((5, 7, 1, 11))
    expected[:, 1, :] = np.ones_like(expected[:, 1, :])
    a = common_layers.shift_right(tf.constant(x1, dtype=tf.float32))
    actual = self.evaluate(a)
    self.assertAllEqual(actual, expected)

  @test_utils.run_in_graph_and_eager_modes()
  def testConvStride2MultiStep(self):
    x1 = np.random.rand(5, 32, 16, 11)
    a = common_layers.conv_stride2_multistep(
        tf.constant(x1, dtype=tf.float32), 4, 16)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(a[0])
    self.assertEqual(actual.shape, (5, 2, 1, 16))

  @test_utils.run_in_graph_and_eager_modes()
  def testDeconvStride2MultiStep(self):
    x1 = np.random.rand(5, 2, 1, 11)
    a = common_layers.deconv_stride2_multistep(
        tf.constant(x1, dtype=tf.float32), 4, 16)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(a)
    self.assertEqual(actual.shape, (5, 32, 1, 16))

  @test_utils.run_in_graph_and_eager_modes()
  def testApplyNormLayer(self):
    x1 = np.random.rand(5, 2, 1, 11)
    x2 = common_layers.apply_norm(
        tf.constant(x1, dtype=tf.float32), "layer", depth=11, epsilon=1e-6)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  @test_utils.run_in_graph_and_eager_modes()
  def testApplyNormNoam(self):
    x1 = np.random.rand(5, 2, 1, 11)
    x2 = common_layers.apply_norm(
        tf.constant(x1, dtype=tf.float32), "noam", depth=11, epsilon=1e-6)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  @test_utils.run_in_graph_and_eager_modes()
  def testApplyNormBatch(self):
    x1 = np.random.rand(5, 2, 1, 11)
    x2 = common_layers.apply_norm(
        tf.constant(x1, dtype=tf.float32), "batch", depth=11, epsilon=1e-6)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  @test_utils.run_in_graph_and_eager_modes()
  def testApplyNormNone(self):
    x1 = np.random.rand(5, 2, 1, 11)
    x2 = common_layers.apply_norm(
        tf.constant(x1, dtype=tf.float32), "none", depth=11, epsilon=1e-6)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))
    self.assertAllClose(actual, x1, atol=1e-03)


  @test_utils.run_in_graph_mode_only()
  def testDenseWithLayerCollection(self):
    with tf.variable_scope("test_layer_collection"):
      x1 = tf.zeros([3, 4], tf.float32)
      layer_collection = kfac.LayerCollection()
      common_layers.dense(
          x1, units=10, layer_collection=layer_collection, name="y1")
      self.assertLen(layer_collection.get_blocks(), 1)

      # 3D inputs.
      x2 = tf.zeros([3, 4, 5], tf.float32)
      common_layers.dense(
          x2, units=10, layer_collection=layer_collection, name="y2")
      self.assertLen(layer_collection.get_blocks(), 2)

  def testGlobalPool1d(self):
    x1 = np.random.rand(5, 4, 11)
    no_mask = np.ones((5, 4))
    full_mask = np.zeros((5, 4))

    x1_ = tf.Variable(x1, dtype=tf.float32)
    no_mask_ = tf.Variable(no_mask, dtype=tf.float32)
    full_mask_ = tf.Variable(full_mask, dtype=tf.float32)

    none_mask_max = common_layers.global_pool_1d(x1_)
    no_mask_max = common_layers.global_pool_1d(x1_, mask=no_mask_)
    result1 = tf.reduce_sum(none_mask_max - no_mask_max)

    full_mask_max = common_layers.global_pool_1d(x1_, mask=full_mask_)
    result2 = tf.reduce_sum(full_mask_max)

    none_mask_avr = common_layers.global_pool_1d(x1_, "AVR")
    no_mask_avr = common_layers.global_pool_1d(x1_, "AVR", no_mask_)
    result3 = tf.reduce_sum(none_mask_avr - no_mask_avr)

    full_mask_avr = common_layers.global_pool_1d(x1_, "AVR", full_mask_)
    result4 = tf.reduce_sum(full_mask_avr)

    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate([result1, result2, result3, result4])
    self.assertAllEqual(actual[:3], [0.0, 0.0, 0.0])

  def testLinearSetLayer(self):
    x1 = np.random.rand(5, 4, 11)
    cont = np.random.rand(5, 13)
    x1_ = tf.Variable(x1, dtype=tf.float32)
    cont_ = tf.Variable(cont, dtype=tf.float32)

    simple_ff = common_layers.linear_set_layer(32, x1_)
    cont_ff = common_layers.linear_set_layer(32, x1_, context=cont_)

    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate([simple_ff, cont_ff])
    self.assertEqual(actual[0].shape, (5, 4, 32))
    self.assertEqual(actual[1].shape, (5, 4, 32))

  def testRavanbakhshSetLayer(self):
    x1 = np.random.rand(5, 4, 11)
    x1_ = tf.Variable(x1, dtype=tf.float32)
    layer = common_layers.ravanbakhsh_set_layer(32, x1_)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(layer)
    self.assertEqual(actual.shape, (5, 4, 32))

  @test_utils.run_in_graph_and_eager_modes()
  def testTopKthIterativeShape(self):
    x = np.random.rand(5, 2, 1, 12)
    y = common_layers.top_kth_iterative(tf.constant(x, dtype=tf.float32), 3)
    actual = self.evaluate(y)
    self.assertEqual(actual.shape, (5, 2, 1, 1))

  @test_utils.run_in_graph_and_eager_modes()
  def testTopKthIterativeValue(self):
    x = [1.0, 2.0, 3.0, 4.0]
    y = common_layers.top_kth_iterative(tf.constant(x, dtype=tf.float32), 3)
    actual = self.evaluate(y)
    self.assertEqual(int(actual[0]), 2.0)

  @test_utils.run_in_graph_and_eager_modes()
  def testBReLU(self):
    x = np.random.rand(5, 2, 1, 12)
    y = common_layers.brelu(tf.constant(x, dtype=tf.float32))
    actual = self.evaluate(y)
    self.assertEqual(actual.shape, (5, 2, 1, 12))

  @test_utils.run_in_graph_and_eager_modes()
  def testBELU(self):
    x = np.random.rand(5, 2, 1, 12)
    y = common_layers.belu(tf.constant(x, dtype=tf.float32))
    actual = self.evaluate(y)
    self.assertEqual(actual.shape, (5, 2, 1, 12))

  @test_utils.run_in_graph_and_eager_modes()
  def testNAC(self):
    x = np.random.rand(5, 2, 1, 12)
    y = common_layers.nac(tf.constant(x, dtype=tf.float32), 14)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(y)
    self.assertEqual(actual.shape, (5, 2, 1, 14))

  @test_utils.run_in_graph_and_eager_modes()
  def testNALU(self):
    x = np.random.rand(5, 2, 1, 12)
    y = common_layers.nalu(tf.constant(x, dtype=tf.float32), 14)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(y)
    self.assertEqual(actual.shape, (5, 2, 1, 14))

  @test_utils.run_in_graph_and_eager_modes()
  def testNALUzeros(self):
    x = np.random.rand(5, 2, 1, 12)
    y = common_layers.nalu(tf.zeros_like(x, dtype=tf.float32), 14)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(y)
    self.assertTrue(np.all(np.isfinite(actual)))
    self.assertEqual(actual.shape, (5, 2, 1, 14))

  @test_utils.run_in_graph_mode_only
  def testPaddingCrossEntropyFactored(self):
    vocab_size = 19
    rows = 5
    cols = 4
    depth = 11
    label_smoothing = 0.1
    features = np.random.rand(rows, cols, depth)
    weights = np.random.rand(vocab_size, depth)
    labels = np.random.randint(0, vocab_size - 1, size=(rows, cols))
    with self.test_session() as session:
      features = tf.to_float(features)
      weights = tf.to_float(weights)
      labels = tf.to_int32(labels)
      logits = tf.matmul(
          tf.reshape(features, [rows * cols, depth]), weights, transpose_b=True)
      logits = tf.reshape(logits, [rows, cols, vocab_size])
      loss_num, loss_den = common_layers.padded_cross_entropy(
          logits, labels, label_smoothing=label_smoothing, reduce_sum=False)
      factored_logits = common_layers.FactoredTensor(features, weights)
      loss_num_f, loss_den_f = common_layers.padded_cross_entropy_factored(
          factored_logits,
          labels=labels,
          label_smoothing=label_smoothing,
          reduce_sum=False)
      num, den, num_f, den_f = session.run(
          [loss_num, loss_den, loss_num_f, loss_den_f])
    self.assertEqual(num.shape, (rows, cols))
    self.assertEqual(den.shape, (rows, cols))
    self.assertEqual(num_f.shape, (rows, cols))
    self.assertEqual(den_f.shape, (rows, cols))
    self.assertAllClose(num, num_f)
    self.assertAllClose(den, den_f)

  @test_utils.run_in_graph_mode_only
  def testPaddingCrossEntropyFactoredGrad(self):
    vocab_size = 19
    rows = 5
    cols = 4
    depth = 11
    label_smoothing = 0.1
    features = np.random.rand(rows, cols, depth)
    weights = np.random.rand(vocab_size, depth)
    labels = np.random.randint(0, vocab_size - 1, size=(rows, cols))
    with self.test_session() as session:
      features = tf.to_float(features)
      weights = tf.to_float(weights)
      labels = tf.to_int32(labels)
      logits = tf.matmul(
          tf.reshape(features, [rows * cols, depth]), weights, transpose_b=True)
      logits = tf.reshape(logits, [rows, cols, vocab_size])
      loss_num, loss_den = common_layers.padded_cross_entropy(
          logits, labels, label_smoothing=label_smoothing, reduce_sum=False)
      factored_logits = common_layers.FactoredTensor(features, weights)
      loss_num_factored, loss_den_factored = (
          common_layers.padded_cross_entropy_factored(
              factored_logits,
              labels=labels,
              label_smoothing=label_smoothing,
              reduce_sum=False))
      df, dw = tf.gradients(ys=[loss_num, loss_den], xs=[features, weights])
      df_factored, dw_factored = tf.gradients(
          ys=[loss_num_factored, loss_den_factored], xs=[features, weights])
      actual_df, actual_dw, actual_df_factored, actual_dw_factored = (
          session.run([df, dw, df_factored, dw_factored]))
    self.assertEqual(actual_df.shape, (rows, cols, depth))
    self.assertEqual(actual_dw.shape, (vocab_size, depth))
    self.assertEqual(actual_df_factored.shape, (rows, cols, depth))
    self.assertEqual(actual_dw_factored.shape, (vocab_size, depth))
    self.assertAllClose(actual_df, actual_df_factored)
    self.assertAllClose(actual_dw, actual_dw_factored)

  @parameterized.parameters(
      (2, 4, 4, 5, True),
      (2, 4, 4, 5, False),
      (1, 16, 16, 1, True),
      (1, 16, 16, 1, False),
  )
  def testDmlLoss(self, batch, height, width, num_mixtures, reduce_sum):
    channels = 3
    pred = tf.random_normal([batch, height, width, num_mixtures * 10])
    labels = tf.random_uniform([batch, height, width, channels],
                               minval=0, maxval=256, dtype=tf.int32)
    actual_loss_num, actual_loss_den = common_layers.dml_loss(
        pred=pred, labels=labels, reduce_sum=reduce_sum)
    actual_loss = actual_loss_num / actual_loss_den

    real_labels = common_layers.convert_rgb_to_symmetric_real(labels)
    expected_loss = common_layers.discretized_mix_logistic_loss(
        pred=pred, labels=real_labels) / channels
    if reduce_sum:
      expected_loss = tf.reduce_mean(expected_loss)

    actual_loss_val, expected_loss_val = self.evaluate(
        [actual_loss, expected_loss])
    self.assertAllClose(actual_loss_val, expected_loss_val)

  @test_utils.run_in_graph_and_eager_modes()
  def testWeightsMultiProblemAll(self):
    labels = tf.constant(np.array([[12, 15, 1, 20, 100],
                                   [67, 1, 34, 45, 124],
                                   [78, 2, 34, 18, 29],
                                   [78, 123, 55, 1, 33],
                                   [1, 18, 22, 36, 59]]), dtype=tf.int32)
    taskid = 1
    expected_mask = np.array([[1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1],
                              [0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1],
                              [1, 1, 1, 1, 1]])
    actual_mask = common_layers.weights_multi_problem_all(labels, taskid)
    actual_mask_eval = self.evaluate(actual_mask)
    self.assertAllClose(expected_mask, actual_mask_eval)

  @test_utils.run_in_graph_and_eager_modes()
  def testWeightsMultiProblem(self):
    labels = tf.constant(np.array([[12, 15, 1, 20, 100],
                                   [67, 1, 34, 45, 124],
                                   [78, 2, 34, 18, 29],
                                   [78, 123, 55, 1, 33],
                                   [1, 18, 22, 36, 59]]), dtype=tf.int32)
    taskid = 1
    expected_mask = np.array([[0, 0, 0, 1, 1],
                              [0, 0, 1, 1, 1],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1],
                              [0, 1, 1, 1, 1]])
    actual_mask = common_layers.weights_multi_problem(labels, taskid)
    actual_mask_eval = self.evaluate(actual_mask)
    self.assertAllClose(expected_mask, actual_mask_eval)

  @test_utils.run_in_graph_and_eager_modes()
  def testDiscretizedMixLogisticLoss(self):
    batch = 2
    height = 4
    width = 4
    channels = 3
    num_mixtures = 5
    logits = tf.concat(  # assign all probability mass to first component
        [tf.ones([batch, height, width, 1]) * 1e8,
         tf.zeros([batch, height, width, num_mixtures - 1])],
        axis=-1)
    locs = tf.random_uniform([batch, height, width, num_mixtures * 3],
                             minval=-.9, maxval=.9)
    log_scales = tf.random_uniform([batch, height, width, num_mixtures * 3],
                                   minval=-1., maxval=1.)
    coeffs = tf.atanh(tf.zeros([batch, height, width, num_mixtures * 3]))
    pred = tf.concat([logits, locs, log_scales, coeffs], axis=-1)

    # Test labels that don't satisfy edge cases where 8-bit value is 0 or 255.
    labels = tf.random_uniform([batch, height, width, channels],
                               minval=-.9, maxval=.9)
    locs_0 = locs[..., :3]
    log_scales_0 = log_scales[..., :3]
    centered_labels = labels - locs_0
    inv_stdv = tf.exp(-log_scales_0)
    plus_in = inv_stdv * (centered_labels + 1. / 255.)
    min_in = inv_stdv * (centered_labels - 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    cdf_min = tf.nn.sigmoid(min_in)
    expected_loss = -tf.reduce_sum(tf.log(cdf_plus - cdf_min), axis=-1)

    actual_loss = common_layers.discretized_mix_logistic_loss(
        pred=pred, labels=labels)
    actual_loss_val, expected_loss_val = self.evaluate(
        [actual_loss, expected_loss])
    self.assertAllClose(actual_loss_val, expected_loss_val, rtol=1e-5)

  @test_utils.run_in_graph_and_eager_modes()
  def testSampleFromDiscretizedMixLogistic(self):
    batch = 2
    height = 4
    width = 4
    num_mixtures = 5
    seed = 42
    logits = tf.concat(  # assign all probability mass to first component
        [tf.ones([batch, height, width, 1]) * 1e8,
         tf.zeros([batch, height, width, num_mixtures - 1])],
        axis=-1)
    locs = tf.random_uniform([batch, height, width, num_mixtures * 3],
                             minval=-.9, maxval=.9)
    log_scales = tf.ones([batch, height, width, num_mixtures * 3]) * -1e8
    coeffs = tf.atanh(tf.zeros([batch, height, width, num_mixtures * 3]))
    pred = tf.concat([logits, locs, log_scales, coeffs], axis=-1)

    locs_0 = locs[..., :3]
    expected_sample = tf.clip_by_value(locs_0, -1., 1.)

    actual_sample = common_layers.sample_from_discretized_mix_logistic(
        pred, seed=seed)
    actual_sample_val, expected_sample_val = self.evaluate(
        [actual_sample, expected_sample])
    # Use a low tolerance: samples numerically differ, as the actual
    # implementation clips log-scales so they always contribute to sampling.
    self.assertAllClose(actual_sample_val, expected_sample_val, atol=1e-2)

  @test_utils.run_in_graph_and_eager_modes()
  def testFactoredTensorImplicitConversion(self):
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(6, 5)
    c = np.random.rand(3, 4, 6)
    # a factored representation of a Tensor of shape (3, 4, 6)
    factored = common_layers.FactoredTensor(tf.to_float(a), tf.to_float(b))
    # implicitly converts factored to a Tensor (performing the matmul)
    d = factored + tf.to_float(c)
    out = self.evaluate(d)
    self.assertEqual(out.shape, (3, 4, 6))

  @test_utils.run_in_graph_mode_only()
  def testConvHiddenReluMemoryEfficient(self):
    batch = 3
    length = 23
    io_size = 16
    filter_size = 7
    x = np.random.rand(batch, length, io_size)
    dy = np.random.rand(batch, length, io_size)
    with self.test_session() as session:
      x = tf.to_float(x)
      dy = tf.to_float(dy)
      f1 = tf.get_variable("f1", [1, io_size, filter_size])
      f2 = tf.get_variable("f2", [1, filter_size, io_size])
      norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
      y = common_layers.conv_hidden_relu_memory_efficient(
          x, filter_size, forget=False,
          test_vars=(f1, f2, norm_scale, norm_bias))
      y_forget = common_layers.conv_hidden_relu_memory_efficient(
          x, filter_size, forget=True,
          test_vars=(f1, f2, norm_scale, norm_bias))
      dx, df1, df2, dnorm_scale, dnorm_bias = tf.gradients(
          ys=[y], xs=[x, f1, f2, norm_scale, norm_bias], grad_ys=[dy])
      dx_f, df1_f, df2_f, dnorm_scale_f, dnorm_bias_f = tf.gradients(
          ys=[y_forget], xs=[x, f1, f2, norm_scale, norm_bias], grad_ys=[dy])
      session.run(tf.global_variables_initializer())
      (y, y_forget,
       dx, df1, df2, dnorm_scale, dnorm_bias,
       dx_f, df1_f, df2_f, dnorm_scale_f, dnorm_bias_f) = session.run(
           [y, y_forget,
            dx, df1, df2, dnorm_scale, dnorm_bias,
            dx_f, df1_f, df2_f, dnorm_scale_f, dnorm_bias_f])
    self.assertAllClose(y, y_forget)
    self.assertAllClose(df2, df2_f, rtol=2e-6, atol=2e-6)
    self.assertAllClose(df1, df1_f, rtol=2e-6, atol=2e-6)
    self.assertAllClose(dnorm_scale, dnorm_scale_f)
    self.assertAllClose(dnorm_bias, dnorm_bias_f)
    self.assertAllClose(dx, dx_f)

  @test_utils.run_in_graph_and_eager_modes()
  def testCycleGANUpsampleNnUpsampleConv(self):
    batch = 8
    height = 32
    width = 32
    num_channels = 3
    output_filters = 10
    stride = [2, 3]  # we want height to be x2 and width to be x3
    random_input = np.random.rand(batch, height, width, num_channels).astype(
        np.float32)

    # nn_upsample_conv gives exactly the shapes we'd expect.
    upsampled_output = common_layers.cyclegan_upsample(
        random_input, output_filters, stride, "nn_upsample_conv")
    upsampled_output_shape = tf.shape(upsampled_output)
    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        [batch, height * stride[0], width * stride[1], output_filters],
        self.evaluate(upsampled_output_shape))

  @test_utils.run_in_graph_and_eager_modes()
  def testCycleGANUpsampleBilinearUpsampleConv(self):
    batch = 8
    height = 32
    width = 32
    num_channels = 3
    output_filters = 10
    stride = [2, 3]  # we want height to be x2 and width to be x3
    random_input = np.random.rand(batch, height, width, num_channels).astype(
        np.float32)

    # bilinear_upsample_conv gives exactly the shapes we'd expect.
    upsampled_output = common_layers.cyclegan_upsample(
        random_input, output_filters, stride, "bilinear_upsample_conv")
    upsampled_output_shape = tf.shape(upsampled_output)
    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        [batch, height * stride[0], width * stride[1], output_filters],
        self.evaluate(upsampled_output_shape))

  @test_utils.run_in_graph_and_eager_modes()
  def testCycleGANUpsampleConv2dTranspose(self):
    batch = 8
    height = 32
    width = 32
    num_channels = 3
    output_filters = 10
    stride = [2, 3]  # we want height to be x2 and width to be x3
    random_input = tf.convert_to_tensor(
        np.random.rand(batch, height, width, num_channels), dtype=tf.float32)

    # conv2d_transpose is a little tricky.
    # height_new = (height_old - 1) * stride + kernel - 2*padding - correction
    # here kernel = 3, padding = 0, correction = 1
    upsampled_height = (height - 1) * stride[0] + 3 - 2*0 - 1
    upsampled_width = (width - 1) * stride[1] + 3 - 2*0 - 1
    upsampled_output = common_layers.cyclegan_upsample(random_input,
                                                       output_filters, stride,
                                                       "conv2d_transpose")
    upsampled_output_shape = tf.shape(upsampled_output)
    self.evaluate(tf.global_variables_initializer())
    self.assertAllEqual(
        [batch, upsampled_height, upsampled_width, output_filters],
        self.evaluate(upsampled_output_shape))

  def testSpectralNorm(self):
    # Test that after 20 calls to apply_spectral_norm, the spectral
    # norm of the normalized matrix is close to 1.0
    with tf.Graph().as_default():
      weights = tf.get_variable("w", dtype=tf.float32, shape=[2, 3, 50, 100])
      weights = tf.multiply(weights, 10.0)
      normed_weight, assign_op = common_layers.apply_spectral_norm(weights)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(20):
          sess.run(assign_op)
          normed_weight, assign_op = common_layers.apply_spectral_norm(
              weights)
        normed_weight = sess.run(normed_weight).reshape(-1, 100)
        _, s, _ = np.linalg.svd(normed_weight)
        self.assertTrue(np.allclose(s[0], 1.0, rtol=0.1))


class FnWithCustomGradTest(tf.test.TestCase):

  @test_utils.run_in_graph_mode_only()
  def testCorrectness(self):

    w = tf.random_uniform([6, 10])

    def fn(a, b, c):
      return tf.layers.dense(
          a,
          10,
          use_bias=False,
          kernel_initializer=lambda shape, dtype, partition_info: w
      ) + tf.matmul(b, c)

    def grad_fn(inputs, variables, outputs, grad_outputs):
      outputs = outputs[0]
      grad_outputs = grad_outputs[0]
      grad_inputs = tf.gradients(outputs, inputs, grad_ys=grad_outputs)
      grad_vars = tf.gradients(outputs, variables, grad_ys=grad_outputs)
      return grad_inputs, grad_vars

    custom_fn = common_layers.fn_with_custom_grad(grad_fn)(fn)

    a = tf.random_uniform([11, 6])
    b = tf.random_uniform([11, 7])
    c = tf.random_uniform([7, 10])

    out = fn(a, b, c)
    custom_out = custom_fn(a, b, c)
    self.assertEqual(out.get_shape().as_list(),
                     custom_out.get_shape().as_list())

    loss = tf.reduce_mean(out)
    custom_loss = tf.reduce_mean(custom_out)

    grads = tf.gradients(loss, [a, b, c] + [tf.trainable_variables()[0]])
    custom_grads = tf.gradients(custom_loss,
                                [a, b, c] + [tf.trainable_variables()[1]])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out_val, custom_out_val, grads_val, custom_grads_val = sess.run(
          [out, custom_out, grads, custom_grads])
      self.assertAllClose(out_val, custom_out_val)
      for g1, g2 in zip(grads_val, custom_grads_val):
        self.assertAllClose(g1, g2)

  @test_utils.run_in_graph_mode_only()
  def testCustomGrad(self):

    def fn(a, b, c):
      return tf.layers.dense(a, 10, use_bias=False) + tf.matmul(b, c)

    def grad_fn(inputs, variables, unused_outputs, unused_grad_outputs):
      grad_inputs = [tf.ones_like(t) * (i + 1.) for i, t in enumerate(inputs)]
      grad_vars = [
          tf.ones_like(t) * (i + len(inputs) + 1.)
          for i, t in enumerate(variables)
      ]
      return grad_inputs, grad_vars

    a = tf.random_uniform([11, 6])
    b = tf.random_uniform([11, 7])
    c = tf.random_uniform([7, 10])
    w = tf.random_uniform([6, 10])
    out = common_layers.fn_with_custom_grad(grad_fn)(fn)(a, b, c)
    loss = tf.reduce_mean(out)
    grads = tf.gradients(loss, [a, b, c, tf.trainable_variables()[0]])
    expected_grads = [
        tf.ones_like(t) * (i + 1.) for i, t in enumerate([a, b, c, w])
    ]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      g_val, eg_val = sess.run([grads, expected_grads])
      for g1, g2 in zip(g_val, eg_val):
        self.assertAllClose(g1, g2)


class RecomputeTest(tf.test.TestCase):

  @test_utils.run_in_graph_mode_only()
  def testRecompute(self):

    def layer(x, name=None):
      with tf.variable_scope(name, default_name="layer"):
        x = tf.contrib.layers.layer_norm(x)
        x = tf.layers.conv1d(
            x,
            10,
            1,
            use_bias=False,
            kernel_initializer=tf.constant_initializer(42.42))
        x = tf.nn.relu(x)
        return x

    def fn(x):
      out = x
      for _ in range(3):
        out = layer(out)
      return out

    @common_layers.recompute_grad
    def fn_recompute(x):
      return fn(x)

    x = tf.random_uniform((3, 1, 3))
    recompute_vars = None
    with tf.variable_scope("recompute") as vs:
      out1 = tf.reduce_sum(fn_recompute(x))
      recompute_vars = vs.trainable_variables()
    reg_vars = None
    with tf.variable_scope("regular") as vs:
      out2 = tf.reduce_sum(fn(x))
      reg_vars = vs.trainable_variables()

    grad1 = tf.gradients(out1, recompute_vars)
    grad2 = tf.gradients(out2, reg_vars)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outs = sess.run([out1, out2, grad1, grad2])
      self.assertAllClose(outs[0], outs[1])
      for g1, g2 in zip(outs[2], outs[3]):
        self.assertAllClose(g1, g2)


if __name__ == "__main__":
  tf.test.main()
