# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Tests for area attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from tensor2tensor.layers import area_attention
import tensorflow.compat.v1 as tf


class AreaAttentionTest(parameterized.TestCase, tf.test.TestCase):

  def testComputeAreaFeatures1D(self):
    features = tf.constant([[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                            [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1], [7.1, 8.1],
                             [9.1, 10.1]]],
                           dtype=tf.float32)
    area_mean, area_std, area_sum, area_height, area_widths = (
        area_attention.compute_area_features(features, max_area_width=3,
                                             epsilon=0.))
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res1, res2, res3, res4, res5 = session.run([area_mean, area_std, area_sum,
                                                  area_height, area_widths])
    self.assertAllClose(((((1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                           (2, 3), (4, 5), (6, 7), (8, 9),
                           (3, 4), (5, 6), (7, 8)),
                          ((1.1, 2.1), (3.1, 4.1), (5.1, 6.1), (7.1, 8.1),
                           (9.1, 10.1),
                           (2.1, 3.1), (4.1, 5.1), (6.1, 7.1), (8.1, 9.1),
                           (3.1, 4.1), (5.1, 6.1), (7.1, 8.1)))),
                        res1,
                        msg="mean_1d")
    expected_std = np.array([[[0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                              [1, 1], [1, 1], [1, 1], [1, 1],
                              [1.63299, 1.63299], [1.63299, 1.63299],
                              [1.63299, 1.63299]],
                             [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                              [1, 1], [1, 1], [1, 1], [1, 1],
                              [1.63299, 1.63299], [1.63299, 1.63299],
                              [1.63299, 1.63299]]])
    self.assertAllClose(expected_std, res2, atol=1e-2, msg="std_1d")
    self.assertAllClose([[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                          [4, 6], [8, 10], [12, 14], [16, 18],
                          [9, 12], [15, 18], [21, 24]],
                         [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1], [7.1, 8.1],
                          [9.1, 10.1],
                          [4.2, 6.2], [8.2, 10.2], [12.2, 14.2], [16.2, 18.2],
                          [9.3, 12.3], [15.3, 18.3], [21.3, 24.3]]],
                        res3,
                        msg="sum_1d")
    self.assertAllEqual([[[1], [1], [1], [1], [1],
                          [1], [1], [1], [1],
                          [1], [1], [1]],
                         [[1], [1], [1], [1], [1],
                          [1], [1], [1], [1],
                          [1], [1], [1]]],
                        res4,
                        msg="height_1d")
    self.assertAllEqual([[[1], [1], [1], [1], [1],
                          [2], [2], [2], [2],
                          [3], [3], [3]],
                         [[1], [1], [1], [1], [1],
                          [2], [2], [2], [2],
                          [3], [3], [3]]],
                        res5,
                        msg="width_1d")

  def testComputeAreaFeatures2D(self):
    features = tf.constant([[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
                            [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1], [7.1, 8.1],
                             [9.1, 10.1], [11.1, 12.1]]],
                           dtype=tf.float32)
    area_mean, area_std, area_sum, area_height, area_widths = (
        area_attention.compute_area_features(features, max_area_width=3,
                                             max_area_height=2,
                                             height=2, epsilon=0.))
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res1, _, res3, res4, res5 = session.run([area_mean, area_std, area_sum,
                                               area_height, area_widths])
    expected_means = [[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                       [2, 3], [4, 5], [8, 9], [10, 11],
                       [3, 4], [9, 10],
                       [4, 5], [6, 7], [8, 9],
                       [5, 6], [7, 8],
                       [6, 7]],
                      [[1.1, 2.1], [3.1, 4.1], [5.1, 6.1], [7.1, 8.1],
                       [9.1, 10.1], [11.1, 12.1],
                       [2.1, 3.1], [4.1, 5.1], [8.1, 9.1], [10.1, 11.1],
                       [3.1, 4.1], [9.1, 10.1],
                       [4.1, 5.1], [6.1, 7.1], [8.1, 9.1],
                       [5.1, 6.1], [7.1, 8.1],
                       [6.1, 7.1]]]
    self.assertAllClose(expected_means, res1, msg="mean_1d")
    expected_heights = [[[1], [1], [1], [1], [1], [1],
                         # 1x2
                         [1], [1], [1], [1],
                         # 1x3
                         [1], [1],
                         # 2x1
                         [2], [2], [2],
                         # 2x2
                         [2], [2],
                         # 2x3
                         [2]],
                        [[1], [1], [1], [1], [1], [1],
                         # 1x2
                         [1], [1], [1], [1],
                         # 1x3
                         [1], [1],
                         # 2x1
                         [2], [2], [2],
                         # 2x2
                         [2], [2],
                         # 2x3
                         [2]]]
    self.assertAllEqual(expected_heights, res4, msg="height_1d")
    expected_widths = [[[1], [1], [1], [1], [1], [1],
                        # 1x2
                        [2], [2], [2], [2],
                        # 1x3
                        [3], [3],
                        # 2x1
                        [1], [1], [1],
                        # 2x2
                        [2], [2],
                        # 2x3
                        [3]],
                       [[1], [1], [1], [1], [1], [1],
                        # 1x2
                        [2], [2], [2], [2],
                        # 1x3
                        [3], [3],
                        # 2x1
                        [1], [1], [1],
                        # 2x2
                        [2], [2],
                        # 2x3
                        [3]]]
    self.assertAllEqual(expected_widths, res5, msg="width_1d")
    sizes = np.multiply(np.array(expected_heights), np.array(expected_widths))
    expected_sums = np.multiply(np.array(expected_means), sizes)
    self.assertAllClose(expected_sums, res3, msg="sum_1d")

  def testAreaMean(self):
    batch_size = 256
    feature_len = 100
    memory_height = 10
    heads = 2
    key_len = 2
    depth = 128
    max_area_height = 3
    max_area_width = 3
    queries = tf.random_uniform([batch_size, heads, key_len, depth],
                                minval=-10.0, maxval=10.0)
    features = tf.random_uniform([batch_size, heads, feature_len, depth],
                                 minval=-10.0, maxval=10.0)
    target_values = tf.random_uniform([batch_size, heads, key_len, depth],
                                      minval=-0.2, maxval=0.2)
    keys = tf.layers.dense(features, units=depth)
    values = tf.layers.dense(features, units=depth)
    mean_attention = area_attention.dot_product_area_attention(
        queries, keys, values,
        bias=None,
        area_key_mode="mean",
        name="mean_key",
        max_area_width=max_area_width,
        max_area_height=max_area_height,
        memory_height=memory_height)
    mean_gradients = tf.gradients(
        tf.reduce_mean(
            tf.pow(target_values - mean_attention, 2)), features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      result = session.run([mean_gradients])
    self.assertFalse(np.any(np.logical_not(np.isfinite(result))))

  def test2DAreaMax(self):
    batch_size = 256
    feature_len = 100
    memory_height = 10
    heads = 2
    key_len = 6
    depth = 128
    max_area_height = 3
    max_area_width = 3
    queries = tf.random_uniform([batch_size, heads, key_len, depth],
                                minval=-10.0, maxval=10.0)
    features = tf.random_uniform([batch_size, heads, feature_len, depth],
                                 minval=-10.0, maxval=10.0)
    target_values = tf.random_uniform([batch_size, heads, key_len, depth],
                                      minval=-0.2, maxval=0.2)
    keys = tf.layers.dense(features, units=depth)
    values = tf.layers.dense(features, units=depth)
    max_attention = area_attention.dot_product_area_attention(
        queries, keys, values,
        bias=None,
        area_key_mode="max",
        area_value_mode="max",
        name="max_key",
        max_area_width=max_area_width,
        max_area_height=max_area_height,
        memory_height=memory_height)
    max_gradients = tf.gradients(tf.reduce_mean(
        tf.pow(target_values - max_attention, 2)), features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      result1, result2 = session.run([max_gradients, max_attention])
    self.assertFalse(np.any(np.logical_not(np.isfinite(result1))))
    self.assertFalse(np.any(np.logical_not(np.isfinite(result2))))

  def test1DAreaMax(self):
    batch_size = 256
    feature_len = 100
    heads = 2
    key_len = 15
    depth = 128
    max_area_width = 3
    queries = tf.random_uniform([batch_size, heads, key_len, depth],
                                minval=-10.0, maxval=10.0)
    features = tf.random_uniform([batch_size, heads, feature_len, depth],
                                 minval=-10.0, maxval=10.0)
    feature_length = tf.constant(
        np.concatenate(
            (np.random.randint(max_area_width, feature_len, [batch_size - 1]),
             np.array([feature_len])), axis=0), tf.int32)
    base_mask = tf.expand_dims(tf.sequence_mask(feature_length), 1)
    mask = tf.expand_dims(base_mask, 3)
    mask = tf.tile(mask, [1, heads, 1, depth])
    features = tf.where(mask, features, tf.zeros_like(features))
    # [batch, 1, 1, memory_length]
    bias_mask = tf.expand_dims(base_mask, 1)
    bias = tf.where(
        bias_mask,
        tf.zeros_like(bias_mask, tf.float32),
        tf.ones_like(bias_mask, tf.float32) * -1e9)
    target_values = tf.random_uniform([batch_size, heads, key_len, depth],
                                      minval=-0.2, maxval=0.2)
    keys = tf.layers.dense(features, units=depth)
    values = tf.layers.dense(features, units=depth)
    max_attention = area_attention.dot_product_area_attention(
        queries, keys, values,
        bias=bias,
        area_key_mode="max",
        area_value_mode="max",
        name="max_key",
        max_area_width=max_area_width)
    max_gradients = tf.gradients(
        tf.reduce_mean(
            tf.pow(target_values - max_attention, 2)), features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      result1, result2 = session.run([max_gradients, max_attention])
    self.assertFalse(np.any(np.logical_not(np.isfinite(result1))))
    self.assertFalse(np.any(np.logical_not(np.isfinite(result2))))

if __name__ == "__main__":
  tf.test.main()
