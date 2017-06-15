# Copyright 2017 Google Inc.
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

  def testConv(self):
    x = np.random.rand(5, 7, 1, 11)
    with self.test_session() as session:
      y = common_layers.conv(tf.constant(x, dtype=tf.float32), 13, (3, 3))
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 5, 1, 13))

  def testSeparableConv(self):
    x = np.random.rand(5, 7, 1, 11)
    with self.test_session() as session:
      y = common_layers.separable_conv(
          tf.constant(x, dtype=tf.float32), 13, (3, 3))
      session.run(tf.global_variables_initializer())
      res = session.run(y)
    self.assertEqual(res.shape, (5, 5, 1, 13))

  def testSubSeparableConv(self):
    for sep in [0, 1, 2, 4]:
      x = np.random.rand(5, 7, 1, 12)
      with self.test_session() as session:
        with tf.variable_scope("sep_%d" % sep):
          y = common_layers.subseparable_conv(
              tf.constant(x, dtype=tf.float32), 16, (3, 3), separability=sep)
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
    x1 = np.random.rand(5, 32, 1, 11)
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


if __name__ == "__main__":
  tf.test.main()
