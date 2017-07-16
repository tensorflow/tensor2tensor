# Copyright 2017 The Tensor2Tensor Authors.
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

# Dependency imports

import numpy as np
from tensor2tensor.models import common_layers

import tensorflow as tf


class CommonLayersTest(tf.test.TestCase):

  def testStandardizeImages(self):
    x = np.random.rand(5, 7, 7, 3)
    with self.test_session() as session:
      y = common_layers.standardize_images(tf.constant(x))
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 7, 3))

  def testImageAugmentation(self):
    x = np.random.rand(500, 500, 3)
    with self.test_session() as session:
      y = common_layers.image_augmentation(tf.constant(x))
      res = session.run(y)
    self.assertEqual(res.shape, (299, 299, 3))

  def testSaturatingSigmoid(self):
    x = np.array([-120.0, -100.0, 0.0, 100.0, 120.0], dtype=np.float32)
    with self.test_session() as session:
      y = common_layers.saturating_sigmoid(tf.constant(x))
      res = session.run(y)
    self.assertAllClose(res, [0.0, 0.0, 0.5, 1.0, 1.0])

  def testFlatten4D3D(self):
    x = np.random.random_integers(1, high=8, size=(3, 5, 2))
    with self.test_session() as session:
      y = common_layers.flatten4d3d(common_layers.embedding(x, 10, 7))
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (3, 5 * 2, 7))

  def testEmbedding(self):
    x = np.random.random_integers(1, high=8, size=(3, 5))
    with self.test_session() as session:
      y = common_layers.embedding(x, 10, 16)
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (3, 5, 16))

  def testShakeShake(self):
    x = np.random.rand(5, 7)
    with self.test_session() as session:
      x = tf.constant(x, dtype=tf.float32)
      y = common_layers.shakeshake([x, x, x, x, x])
      session.run(tf.global_variables_initializer())
      inp, res = session.run([x, y])
    self.assertAllClose(res, inp)

  def testConv(self):
    x = np.random.rand(5, 7, 1, 11)
    with self.test_session() as session:
      y = common_layers.conv(tf.constant(x, dtype=tf.float32), 13, (3, 1))
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 5, 1, 13))

  def testConv1d(self):
    x = np.random.rand(5, 7, 11)
    with self.test_session() as session:
      y = common_layers.conv1d(tf.constant(x, dtype=tf.float32), 13, 1)
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 13))

  def testSeparableConv(self):
    x = np.random.rand(5, 7, 1, 11)
    with self.test_session() as session:
      y = common_layers.separable_conv(
          tf.constant(x, dtype=tf.float32), 13, (3, 1))
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 5, 1, 13))

  def testSubSeparableConv(self):
    for sep in [0, 1, 2, 4]:
      x = np.random.rand(5, 7, 1, 12)
      with self.test_session() as session:
        with tf.variable_scope("sep_%d" % sep):
          y = common_layers.subseparable_conv(
              tf.constant(x, dtype=tf.float32), 16, (3, 1), separability=sep)
        session.run(tf.global_variables_initializer())
        res = session.run(y)
      self.assertEqual(res.shape, (5, 5, 1, 16))

  def testConvBlock(self):
    x = np.random.rand(5, 7, 1, 11)
    with self.test_session() as session:
      y = common_layers.conv_block(
          tf.constant(x, dtype=tf.float32),
          13, [(1, (3, 3)), (1, (3, 3))],
          padding="SAME",
          normalizer_fn=common_layers.noam_norm)
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 1, 13))

  def testSeparableConvBlock(self):
    x = np.random.rand(5, 7, 1, 11)
    with self.test_session() as session:
      y = common_layers.separable_conv_block(
          tf.constant(x, dtype=tf.float32),
          13, [(1, (3, 3)), (1, (3, 3))],
          padding="SAME")
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 1, 13))

  def testSubSeparableConvBlock(self):
    for sep in [0, 1, 2, 4]:
      x = np.random.rand(5, 7, 1, 12)
      with self.test_session() as session:
        with tf.variable_scope("sep_%d" % sep):
          y = common_layers.subseparable_conv_block(
              tf.constant(x, dtype=tf.float32),
              16, [(1, (3, 3)), (1, (3, 3))],
              padding="SAME",
              separability=sep)
        session.run(tf.global_variables_initializer())
        res = session.run(y)
      self.assertEqual(res.shape, (5, 7, 1, 16))

  def testPool(self):
    x = np.random.rand(5, 8, 1, 11)
    with self.test_session() as session:
      y = common_layers.pool(
          tf.constant(x, dtype=tf.float32), (2, 2), "AVG", "SAME")
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 8, 1, 11))

  def testConvBlockDownsample(self):
    x = np.random.rand(5, 7, 1, 11)
    with self.test_session() as session:
      y = common_layers.conv_block_downsample(
          tf.constant(x, dtype=tf.float32), (3, 1), (2, 1), "SAME")
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 4, 1, 27))

  def testSimpleAttention(self):
    x = np.random.rand(5, 7, 1, 11)
    y = np.random.rand(5, 9, 1, 11)
    with self.test_session() as session:
      a = common_layers.simple_attention(
          tf.constant(x, dtype=tf.float32), tf.constant(y, dtype=tf.float32))
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 7, 1, 11))

  def testGetTimingSignal(self):
    length = 7
    num_timescales = 10
    with self.test_session() as session:
      a = common_layers.get_timing_signal(length, num_timescales=num_timescales)
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (length, 2 * num_timescales))

  def testAddTimingSignal(self):
    batch = 5
    length = 7
    height = 3
    depth = 35
    x = np.random.rand(batch, length, height, depth)
    with self.test_session() as session:
      a = common_layers.add_timing_signal(tf.constant(x, dtype=tf.float32))
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (batch, length, height, depth))

  def testAttention1D(self):
    batch = 5
    target_length = 7
    source_length = 13
    source_depth = 9
    target_depth = 11
    attention_size = 21
    output_size = 15
    num_heads = 7
    source = np.random.rand(batch, source_length, source_depth)
    target = np.random.rand(batch, target_length, target_depth)
    mask = np.random.rand(batch, target_length, source_length)
    with self.test_session() as session:
      a = common_layers.attention_1d_v0(
          tf.constant(source, dtype=tf.float32),
          tf.constant(target, dtype=tf.float32), attention_size, output_size,
          num_heads, tf.constant(mask, dtype=tf.float32))
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (batch, target_length, output_size))

  def testMultiscaleConvSum(self):
    x = np.random.rand(5, 9, 1, 11)
    with self.test_session() as session:
      y = common_layers.multiscale_conv_sum(
          tf.constant(x, dtype=tf.float32),
          13, [((1, 1), (5, 5)), ((2, 2), (3, 3))],
          "AVG",
          padding="SAME")
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 9, 1, 13))

  def testConvGRU(self):
    x = np.random.rand(5, 7, 3, 11)
    with self.test_session() as session:
      y = common_layers.conv_gru(tf.constant(x, dtype=tf.float32), (1, 3), 11)
      z = common_layers.conv_gru(
          tf.constant(x, dtype=tf.float32), (1, 3), 11, padding="LEFT")
      session.run(tf.global_variables_initializer())
      res1 = session.run(y)
      res2 = session.run(z)
    self.assertEqual(res1.shape, (5, 7, 3, 11))
    self.assertEqual(res2.shape, (5, 7, 3, 11))

  def testLayerNorm(self):
    x = np.random.rand(5, 7, 11)
    with self.test_session() as session:
      y = common_layers.layer_norm(tf.constant(x, dtype=tf.float32), 11)
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 11))

  def testConvLSTM(self):
    x = np.random.rand(5, 7, 11, 13)
    with self.test_session() as session:
      y = common_layers.conv_lstm(tf.constant(x, dtype=tf.float32), (1, 3), 13)
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 11, 13))

  def testPadToSameLength(self):
    x1 = np.random.rand(5, 7, 11)
    x2 = np.random.rand(5, 9, 11)
    with self.test_session() as session:
      a, b = common_layers.pad_to_same_length(
          tf.constant(x1, dtype=tf.float32), tf.constant(x2, dtype=tf.float32))
      c, d = common_layers.pad_to_same_length(
          tf.constant(x1, dtype=tf.float32),
          tf.constant(x2, dtype=tf.float32),
          final_length_divisible_by=4)
      res1, res2 = session.run([a, b])
      res1a, res2a = session.run([c, d])
    self.assertEqual(res1.shape, (5, 9, 11))
    self.assertEqual(res2.shape, (5, 9, 11))
    self.assertEqual(res1a.shape, (5, 12, 11))
    self.assertEqual(res2a.shape, (5, 12, 11))

  def testShiftLeft(self):
    x1 = np.zeros((5, 7, 1, 11))
    x1[:, 0, :] = np.ones_like(x1[:, 0, :])
    expected = np.zeros((5, 7, 1, 11))
    expected[:, 1, :] = np.ones_like(expected[:, 1, :])
    with self.test_session() as session:
      a = common_layers.shift_left(tf.constant(x1, dtype=tf.float32))
      actual = session.run(a)
    self.assertAllEqual(actual, expected)

  def testConvStride2MultiStep(self):
    x1 = np.random.rand(5, 32, 16, 11)
    with self.test_session() as session:
      a = common_layers.conv_stride2_multistep(
          tf.constant(x1, dtype=tf.float32), 4, 16)
      session.run(tf.global_variables_initializer())
      actual = session.run(a[0])
    self.assertEqual(actual.shape, (5, 2, 1, 16))

  def testDeconvStride2MultiStep(self):
    x1 = np.random.rand(5, 2, 1, 11)
    with self.test_session() as session:
      a = common_layers.deconv_stride2_multistep(
          tf.constant(x1, dtype=tf.float32), 4, 16)
      session.run(tf.global_variables_initializer())
      actual = session.run(a)
    self.assertEqual(actual.shape, (5, 32, 1, 16))

  def testGetNormLayerFn(self):
    norm_type = "layer"
    with self.test_session() as session:
      a = common_layers.get_norm(norm_type)
      x1 = np.random.rand(5, 2, 1, 11)
      x2 = a(tf.constant(x1, dtype=tf.float32), name="layer", filters=11)
      session.run(tf.global_variables_initializer())
      actual = session.run(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  def testGetNormNoamFn(self):
    norm_type = "noam"
    with self.test_session() as session:
      a = common_layers.get_norm(norm_type)
      x1 = np.random.rand(5, 2, 1, 11)
      x2 = a(tf.constant(x1, dtype=tf.float32), name="noam")
      session.run(tf.global_variables_initializer())
      actual = session.run(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  def testGetNormBatchFn(self):
    norm_type = "batch"
    with self.test_session() as session:
      a = common_layers.get_norm(norm_type)
      x1 = np.random.rand(5, 2, 1, 11)
      x2 = a(tf.constant(x1, dtype=tf.float32), name="batch")
      session.run(tf.global_variables_initializer())
      actual = session.run(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  def testGetNormNoneFn(self):
    norm_type = "none"
    with self.test_session() as session:
      a = common_layers.get_norm(norm_type)
      x1 = np.random.rand(5, 2, 1, 11)
      x2 = a(tf.constant(x1, dtype=tf.float32), name="none")
      session.run(tf.global_variables_initializer())
      actual = session.run(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))
    self.assertAllClose(actual, x1, atol=1e-03)

  def testResidualFn(self):
    norm_type = "batch"
    with self.test_session() as session:
      x1 = np.random.rand(5, 2, 1, 11)
      x2 = np.random.rand(5, 2, 1, 11)
      x3 = common_layers.residual_fn(
          tf.constant(x1, dtype=tf.float32),
          tf.constant(x2, dtype=tf.float32),
          norm_type, 0.1)
      session.run(tf.global_variables_initializer())
      actual = session.run(x3)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  def testResidualFnWithLayerNorm(self):
    norm_type = "layer"
    with self.test_session() as session:
      x1 = np.random.rand(5, 2, 1, 11)
      x2 = np.random.rand(5, 2, 1, 11)
      x3 = common_layers.residual_fn(
          tf.constant(x1, dtype=tf.float32),
          tf.constant(x2, dtype=tf.float32),
          norm_type, 0.1, epsilon=0.1)
      session.run(tf.global_variables_initializer())
      actual = session.run(x3)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  def testGlobalPool1d(self):
    x1 = np.random.rand(5, 4, 11)
    no_mask = np.ones((5, 4))
    full_mask = np.zeros((5, 4))

    with self.test_session() as session:
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

      session.run(tf.global_variables_initializer())
      actual = session.run([result1, result2, result3, result4])
    self.assertAllEqual(actual[:3], [0.0, 0.0, 0.0])

  def testLinearSetLayer(self):
    x1 = np.random.rand(5, 4, 11)
    cont = np.random.rand(5, 13)
    with self.test_session() as session:
      x1_ = tf.Variable(x1, dtype=tf.float32)
      cont_ = tf.Variable(cont, dtype=tf.float32)

      simple_ff = common_layers.linear_set_layer(32, x1_)
      cont_ff = common_layers.linear_set_layer(32, x1_, context=cont_)

      session.run(tf.global_variables_initializer())
      actual = session.run([simple_ff, cont_ff])
    self.assertEqual(actual[0].shape, (5, 4, 32))
    self.assertEqual(actual[1].shape, (5, 4, 32))

  def testRavanbakhshSetLayer(self):
    x1 = np.random.rand(5, 4, 11)
    with self.test_session() as session:
      x1_ = tf.Variable(x1, dtype=tf.float32)
      layer = common_layers.ravanbakhsh_set_layer(32, x1_)
      session.run(tf.global_variables_initializer())
      actual = session.run(layer)
    self.assertEqual(actual.shape, (5, 4, 32))


if __name__ == "__main__":
  tf.test.main()
