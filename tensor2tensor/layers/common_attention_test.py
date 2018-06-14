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
"""Tests for common attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

import tensorflow as tf


class CommonAttentionTest(tf.test.TestCase):

  def testDotProductAttention(self):
    x = np.random.rand(5, 7, 12, 32)
    y = np.random.rand(5, 7, 12, 32)
    with self.test_session() as session:
      a = common_attention.dot_product_attention(
          tf.constant(x, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32), None)
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 7, 12, 32))

  def testMaskedLocalAttention1D(self):
    q = np.array([[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]])
    k = np.array([[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]])
    v = np.ones((1, 1, 8, 1))
    with self.test_session() as session:
      q_ = tf.constant(q, dtype=tf.float32)
      k_ = tf.constant(k, dtype=tf.float32)
      v_ = tf.constant(v, dtype=tf.float32)
      y = common_attention.masked_local_attention_1d(
          q_, k_, v_, block_length=tf.constant(2))
      res = session.run(y)

    self.assertEqual(res.shape, (1, 1, 8, 1))

  def testLocalUnmaskedAttention1D(self):
    x = np.random.rand(5, 4, 25, 16)
    y = np.random.rand(5, 4, 25, 16)
    with self.test_session() as session:
      a = common_attention.local_attention_1d(
          tf.constant(x, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          block_length=4,
          filter_width=3)
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 4, 25, 16))

  def testLocalUnmaskedAttention1DMatchingBlockLength(self):
    x = np.random.rand(5, 4, 25, 16)
    y = np.random.rand(5, 4, 25, 16)
    with self.test_session() as session:
      a = common_attention.local_attention_1d(
          tf.constant(x, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          block_length=5,
          filter_width=3)
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 4, 25, 16))

  def testLocalUnmaskedAttention2D(self):
    x = np.random.rand(5, 4, 25, 25, 16)
    y = np.random.rand(5, 4, 25, 25, 16)
    with self.test_session() as session:
      a = common_attention.local_attention_2d(
          tf.constant(x, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          query_shape=(4, 4),
          memory_flange=(3, 3))
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 4, 25, 25, 16))

  def testLocalUnmaskedAttention2DMatchingBlockLength(self):
    x = np.random.rand(5, 4, 25, 25, 16)
    y = np.random.rand(5, 4, 25, 25, 16)
    with self.test_session() as session:
      a = common_attention.local_attention_2d(
          tf.constant(x, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          query_shape=(5, 5),
          memory_flange=(3, 3))
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 4, 25, 25, 16))

  def testMultiheadSelfAttentionMemoryEfficient(self):
    num_heads = 4
    io_size = 16
    batch = 2
    length = 7
    head_size = 5
    x = np.random.rand(batch, length, io_size)
    dy = np.random.rand(batch, length, io_size)
    with self.test_session() as session:
      x = tf.to_float(x)
      dy = tf.to_float(dy)
      bias = common_attention.attention_bias_lower_triangle(length)
      wqkv = tf.get_variable(
          "wqkv", [num_heads, 1, io_size, 3 * head_size],
          initializer=tf.random_normal_initializer(stddev=io_size**-0.5))
      wo = tf.get_variable(
          "wo", [num_heads, 1, head_size, io_size],
          initializer=tf.random_normal_initializer(
              stddev=(head_size * num_heads)**-0.5))
      norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
      y = common_attention.multihead_self_attention_memory_efficient(
          x, bias, num_heads, head_size=head_size, forget=False,
          test_vars=(wqkv, wo, norm_scale, norm_bias))
      y_forget = common_attention.multihead_self_attention_memory_efficient(
          x, bias, num_heads, head_size=head_size, forget=True,
          test_vars=(wqkv, wo, norm_scale, norm_bias))
      dx, dwqkv, dwo, dnorm_scale, dnorm_bias = tf.gradients(
          ys=[y], xs=[x, wqkv, wo, norm_scale, norm_bias], grad_ys=[dy])
      dx_f, dwqkv_f, dwo_f, dnorm_scale_f, dnorm_bias_f = tf.gradients(
          ys=[y_forget], xs=[x, wqkv, wo, norm_scale, norm_bias], grad_ys=[dy])
      session.run(tf.global_variables_initializer())
      (y, y_forget,
       dx, dwqkv, dwo, dnorm_scale, dnorm_bias,
       dx_f, dwqkv_f, dwo_f, dnorm_scale_f, dnorm_bias_f) = session.run(
           [y, y_forget,
            dx, dwqkv, dwo, dnorm_scale, dnorm_bias,
            dx_f, dwqkv_f, dwo_f, dnorm_scale_f, dnorm_bias_f])
    self.assertAllClose(y, y_forget)
    self.assertAllClose(dwo, dwo_f)
    self.assertAllClose(dwqkv, dwqkv_f)
    self.assertAllClose(dnorm_scale, dnorm_scale_f)
    self.assertAllClose(dnorm_bias, dnorm_bias_f)
    self.assertAllClose(dx, dx_f)

  def test2dGatherAndScatterInvertibility(self):
    """2d gather and scatter invertibility test."""
    batch_size = 2
    num_heads = 2
    height = 4
    width = 6
    depth = 8
    query_shape = (2, 3)
    x = np.random.rand(batch_size, num_heads, height, width, depth)
    with self.test_session() as session:
      x_indices = common_attention.gather_indices_2d(
          x, query_shape, query_shape)
      gathered_x = common_attention.gather_blocks_2d(x, x_indices)
      x_shape = tf.constant([batch_size, num_heads, height, width, depth])
      scattered_x = common_attention.scatter_blocks_2d(
          gathered_x, x_indices, x_shape)
      session.run(tf.global_variables_initializer())
      res = session.run(scattered_x)
    self.assertAllClose(x, res)

  def test2dBlockRasterScanMask(self):
    """Testing the 2d block raster scan mask."""
    query_shape = (2, 3)
    memory_flange = (2, 1)
    with self.test_session() as session:
      mask = common_attention.make_2d_block_raster_mask(
          query_shape, memory_flange)
      res = session.run(mask)
    correct_mask = np.array(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
          1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
          1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          1.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    self.assertAllClose(correct_mask, res)

  def test2dGather(self):
    """Testing 2d index gather and block gather functions."""
    batch_size = 2
    num_heads = 2
    height = 4
    width = 6
    depth = 8
    query_shape = (2, 3)
    x = np.random.rand(batch_size, num_heads, height, width, depth)
    y = np.reshape(x, (batch_size, num_heads, -1, depth))
    correct_indices = [[0, 1, 2, 6, 7, 8],
                       [3, 4, 5, 9, 10, 11],
                       [12, 13, 14, 18, 19, 20],
                       [15, 16, 17, 21, 22, 23]]
    correct_gathered_x = [[[y[0, 0, correct_indices[0]],
                            y[0, 0, correct_indices[1]],
                            y[0, 0, correct_indices[2]],
                            y[0, 0, correct_indices[3]]],
                           [y[0, 1, correct_indices[0]],
                            y[0, 1, correct_indices[1]],
                            y[0, 1, correct_indices[2]],
                            y[0, 1, correct_indices[3]]]],
                          [[y[1, 0, correct_indices[0]],
                            y[1, 0, correct_indices[1]],
                            y[1, 0, correct_indices[2]],
                            y[1, 0, correct_indices[3]]],
                           [y[1, 1, correct_indices[0]],
                            y[1, 1, correct_indices[1]],
                            y[1, 1, correct_indices[2]],
                            y[1, 1, correct_indices[3]]]]]

    with self.test_session() as session:
      x_indices = common_attention.gather_indices_2d(
          x, query_shape, query_shape)
      gathered_x = common_attention.gather_blocks_2d(x, x_indices)
      x_indices, gathered_x = session.run([x_indices, gathered_x])
    self.assertAllEqual(correct_indices, x_indices)
    self.assertAllClose(correct_gathered_x, gathered_x)

  def testGetMemoryRegion(self):
    """Testing the function that gathers the flanged memory region."""
    np.set_printoptions(threshold=np.inf)
    batch_size = 2
    num_heads = 2
    height = 4
    width = 6
    depth = 3
    query_shape = (2, 3)
    memory_flange = (1, 1)

    x = np.random.rand(batch_size, num_heads, height, width, depth)
    y = np.reshape(x, (batch_size, num_heads, -1, depth))
    zeros = np.zeros((depth), dtype=np.float32)
    five_zeros = np.array([zeros]*5)
    seven_zeros = np.array([zeros]*7)
    two_zeros = np.array([zeros]*2)
    zeros = np.array([zeros])

    correct_x_flange = [[[seven_zeros,
                          np.concatenate((five_zeros, y[0, 0, [2, 8]]),
                                         axis=0),
                          np.concatenate((zeros, y[0, 0, [6, 7, 8, 9]],
                                          two_zeros), axis=0),
                          np.concatenate((y[0, 0, [8, 9, 10, 11]], zeros,
                                          y[0, 0, [14, 20]]), axis=0)],
                         [seven_zeros,
                          np.concatenate((five_zeros, y[0, 1, [2, 8]]),
                                         axis=0),
                          np.concatenate((zeros, y[0, 1, [6, 7, 8, 9]],
                                          two_zeros), axis=0),
                          np.concatenate((y[0, 1, [8, 9, 10, 11]], zeros,
                                          y[0, 1, [14, 20]]), axis=0)]],
                        [[seven_zeros,
                          np.concatenate((five_zeros, y[1, 0, [2, 8]]),
                                         axis=0),
                          np.concatenate((zeros, y[1, 0, [6, 7, 8, 9]],
                                          two_zeros), axis=0),
                          np.concatenate((y[1, 0, [8, 9, 10, 11]], zeros,
                                          y[1, 0, [14, 20]]), axis=0)],
                         [seven_zeros,
                          np.concatenate((five_zeros, y[1, 1, [2, 8]]),
                                         axis=0),
                          np.concatenate((zeros, y[1, 1, [6, 7, 8, 9]],
                                          two_zeros), axis=0),
                          np.concatenate((y[1, 1, [8, 9, 10, 11]], zeros,
                                          y[1, 1, [14, 20]]), axis=0)]]]
    correct_x_flange = np.array(correct_x_flange)
    correct_x_center = [[[y[0, 0, [0, 1, 2, 6, 7, 8]],
                          y[0, 0, [3, 4, 5, 9, 10, 11]],
                          y[0, 0, [12, 13, 14, 18, 19, 20]],
                          y[0, 0, [15, 16, 17, 21, 22, 23]]],
                         [y[0, 1, [0, 1, 2, 6, 7, 8]],
                          y[0, 1, [3, 4, 5, 9, 10, 11]],
                          y[0, 1, [12, 13, 14, 18, 19, 20]],
                          y[0, 1, [15, 16, 17, 21, 22, 23]]]],
                        [[y[1, 0, [0, 1, 2, 6, 7, 8]],
                          y[1, 0, [3, 4, 5, 9, 10, 11]],
                          y[1, 0, [12, 13, 14, 18, 19, 20]],
                          y[1, 0, [15, 16, 17, 21, 22, 23]]],
                         [y[1, 1, [0, 1, 2, 6, 7, 8]],
                          y[1, 1, [3, 4, 5, 9, 10, 11]],
                          y[1, 1, [12, 13, 14, 18, 19, 20]],
                          y[1, 1, [15, 16, 17, 21, 22, 23]]]]]
    correct_x_center = np.array(correct_x_center)
    with self.test_session() as session:
      x_indices = common_attention.gather_indices_2d(
          x, query_shape, query_shape)
      x_flange, x_center = common_attention.get_memory_region(
          tf.constant(x, dtype=tf.float32),
          query_shape,
          memory_flange,
          x_indices)
      session.run(tf.global_variables_initializer())
      [x_flange, x_center] = session.run([x_flange, x_center])
    self.assertAllClose(correct_x_flange, x_flange)
    self.assertAllClose(correct_x_center, x_center)

  def testGetShiftedCenterBlocks(self):
    """Testing the function that gathers the flanged memory region."""
    np.set_printoptions(threshold=np.inf)
    batch_size = 2
    num_heads = 2
    height = 4
    width = 6
    depth = 3
    query_shape = (2, 3)

    x = np.random.rand(batch_size, num_heads, height, width, depth)
    y = np.reshape(x, (batch_size, num_heads, -1, depth))
    zeros = np.zeros((depth), dtype=np.float32)
    zeros = np.array([zeros])

    correct_gathered_x = [[[np.concatenate((zeros, y[0, 0, [0, 1, 2, 6, 7]]),
                                           axis=0),
                            np.concatenate((zeros, y[0, 0, [3, 4, 5, 9, 10]]),
                                           axis=0),
                            np.concatenate((zeros,
                                            y[0, 0, [12, 13, 14, 18, 19]]),
                                           axis=0),
                            np.concatenate((zeros,
                                            y[0, 0, [15, 16, 17, 21, 22]]),
                                           axis=0)],
                           [np.concatenate((zeros, y[0, 1, [0, 1, 2, 6, 7]]),
                                           axis=0),
                            np.concatenate((zeros, y[0, 1, [3, 4, 5, 9, 10]]),
                                           axis=0),
                            np.concatenate((zeros,
                                            y[0, 1, [12, 13, 14, 18, 19]]),
                                           axis=0),
                            np.concatenate((zeros,
                                            y[0, 1, [15, 16, 17, 21, 22]]),
                                           axis=0)]],
                          [[np.concatenate((zeros, y[1, 0, [0, 1, 2, 6, 7]]),
                                           axis=0),
                            np.concatenate((zeros, y[1, 0, [3, 4, 5, 9, 10]]),
                                           axis=0),
                            np.concatenate((zeros,
                                            y[1, 0, [12, 13, 14, 18, 19]]),
                                           axis=0),
                            np.concatenate((zeros,
                                            y[1, 0, [15, 16, 17, 21, 22]]),
                                           axis=0)],
                           [np.concatenate((zeros, y[1, 1, [0, 1, 2, 6, 7]]),
                                           axis=0),
                            np.concatenate((zeros, y[1, 1, [3, 4, 5, 9, 10]]),
                                           axis=0),
                            np.concatenate((zeros,
                                            y[1, 1, [12, 13, 14, 18, 19]]),
                                           axis=0),
                            np.concatenate((zeros,
                                            y[1, 1, [15, 16, 17, 21, 22]]),
                                           axis=0)]]]
    correct_gathered_x = np.array(correct_gathered_x)
    with self.test_session() as session:
      x_indices = common_attention.gather_indices_2d(
          x, query_shape, query_shape)
      gathered_x = common_attention.get_shifted_center_blocks(
          tf.constant(x, dtype=tf.float32),
          x_indices)
      session.run(tf.global_variables_initializer())
      x_indices, gathered_x = session.run([x_indices, gathered_x])
    self.assertAllClose(correct_gathered_x, gathered_x)

  def testDotProductAttentionRelative(self):
    x = np.random.rand(5, 7, 12, 32)
    y = np.random.rand(5, 7, 12, 32)
    with self.test_session() as session:
      a = common_attention.dot_product_attention_relative(
          tf.constant(x, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          None,
          max_relative_position=3)
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 7, 12, 32))

  def testBiasBatchCoordinates(self):
    """Testing the batch coordinates mask."""
    q = tf.constant([0, 0, 1, 1, 1, 1, 2, 2, 2], dtype=tf.int32)
    q = tf.expand_dims(q, axis=-1)

    k = tf.constant([0, 0, 0, 2, 2, 3, 3, 3], dtype=tf.int32)
    k = tf.expand_dims(k, axis=-1)

    ground_truth = np.array([
        [0, 0, 0, 1, 1, 1, 1, 1],  # 0
        [0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],  # 1 (just masked)
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],  # 2
        [1, 1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
    ], np.float32) * -1e9

    bias = common_attention.attention_bias_coordinates(q, k)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      self.assertAllClose(
          bias.eval(),
          ground_truth,
      )

  def testBiasFuture(self):
    """Testing the sequence order mask."""
    q = tf.constant([0, 1, 2, 3, 0, 1, 2, 0, 1], dtype=tf.int32)
    q = tf.expand_dims(q, axis=-1)

    k = tf.constant([0, 1, 2, 3, 4, 0, 1, 2], dtype=tf.int32)
    k = tf.expand_dims(k, axis=-1)

    ground_truth = np.array([
        [0, 1, 1, 1, 1, 0, 1, 1],  # 0
        [0, 0, 1, 1, 1, 0, 0, 1],  # 1
        [0, 0, 0, 1, 1, 0, 0, 0],  # 2
        [0, 0, 0, 0, 1, 0, 0, 0],  # 3
        [0, 1, 1, 1, 1, 0, 1, 1],  # 0
        [0, 0, 1, 1, 1, 0, 0, 1],  # 1
        [0, 0, 0, 1, 1, 0, 0, 0],  # 2
        [0, 1, 1, 1, 1, 0, 1, 1],  # 0
        [0, 0, 1, 1, 1, 0, 0, 1],  # 1
    ], np.float32) * -1e9

    bias = common_attention.attention_bias_future(q, k)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      self.assertAllClose(
          bias.eval(),
          ground_truth,
      )


if __name__ == "__main__":
  tf.test.main()
