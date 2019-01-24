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

from absl.testing import parameterized
import numpy as np

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

import tensorflow as tf


class CommonAttentionTest(parameterized.TestCase, tf.test.TestCase):

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testAddPositionalEmbedding(self):
    x = np.random.rand(5, 3, 12)
    y = common_attention.add_positional_embedding(
        tf.constant(x, dtype=tf.float32),
        max_length=4,
        name="pos_embedding")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 3, 12))

  @parameterized.parameters(
      {"input_shape": (5, 3, 12)},
      {"input_shape": (5, 5, 5, 12)},
      {"input_shape": (5, 3, 3, 3, 12)},
  )
  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testAddPositionalEmbeddingNd(self, input_shape):
    x = np.random.rand(*input_shape)
    y = common_attention.add_positional_embedding_nd(
        tf.constant(x, dtype=tf.float32),
        max_length=5,
        name="pos_embedding")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, input_shape)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDotProductAttention(self):
    x = np.random.rand(5, 7, 12, 32)
    y = np.random.rand(5, 7, 12, 32)
    a = common_attention.dot_product_attention(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32), None)
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 7, 12, 32))

  @parameterized.named_parameters(
      ("", 1, 1, 8, 4, 1, 2),
      ("dynamic_batch", None, 1, 8, 4, 1, 2),
      ("batches", 4, 3, 8, 4, 1, 2),
      ("depth_v", 1, 1, 8, 4, 3, 2),
      ("block_length", 1, 1, 8, 4, 1, 4),
  )
  def testMaskedWithinBlockLocalAttention1D(self, batch, heads, length,
                                            depth_k, depth_v, block_length):
    if batch is None:
      batch = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
    q = tf.random_normal([batch, heads, length, depth_k])
    k = tf.random_normal([batch, heads, length, depth_k])
    v = tf.random_normal([batch, heads, length, depth_v])
    output = common_attention.masked_within_block_local_attention_1d(
        q, k, v, block_length=block_length)
    if isinstance(batch, tf.Tensor):
      batch, res = self.evaluate([batch, output])
    else:
      res = self.evaluate(output)

    self.assertEqual(res.shape, (batch, heads, length, depth_v))

  @parameterized.named_parameters(
      ("", 1, 1, 8, 4, 1, 2),
      ("dynamic_batch", None, 1, 8, 4, 1, 2),
      ("batches", 4, 3, 8, 4, 1, 2),
      ("depth_v", 1, 1, 8, 4, 3, 2),
      ("block_length", 1, 1, 8, 4, 1, 4),
  )
  def testMaskedLocalAttention1D(self, batch, heads, length, depth_k, depth_v,
                                 block_length):
    if batch is None:
      batch = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
    q = tf.random_normal([batch, heads, length, depth_k])
    k = tf.random_normal([batch, heads, length, depth_k])
    v = tf.random_normal([batch, heads, length, depth_v])
    output = common_attention.masked_local_attention_1d(
        q, k, v, block_length=block_length)
    if isinstance(batch, tf.Tensor):
      batch, res = self.evaluate([batch, output])
    else:
      res = self.evaluate(output)

    self.assertEqual(res.shape, (batch, heads, length, depth_v))

  @parameterized.named_parameters(
      ("", 1, 1, 8, 4, 4, (2, 2)),
      ("dynamic_batch", None, 1, 8, 4, 4, (2, 2)),
      ("batches", 3, 2, 8, 4, 4, (2, 2)),
      # TODO(trandustin): Extend function to enable depth_k != depth_v.
      # ("depth_v", 1, 1, 8, 4, 1, (2, 2)),
      ("query_shape", 1, 1, 8, 4, 4, (4, 4)),
  )
  def testMaskedLocalAttention2D(self, batch, heads, length, depth_k, depth_v,
                                 query_shape):
    if batch is None:
      batch = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
    q = tf.random_normal([batch, heads, length, length, depth_k])
    k = tf.random_normal([batch, heads, length, length, depth_k])
    v = tf.random_normal([batch, heads, length, length, depth_v])
    output = common_attention.masked_local_attention_2d(
        q,
        k,
        v,
        query_shape=query_shape,
        memory_flange=(2, 2))
    if isinstance(batch, tf.Tensor):
      batch, res = self.evaluate([batch, output])
    else:
      res = self.evaluate(output)

    self.assertEqual(res.shape, (batch, heads, length, length, depth_v))

  @parameterized.named_parameters(
      ("matching_block_length", 3, 4, 25, 16, 16, 5),
      ("unmatching_block_length", 3, 4, 25, 16, 16, 4),
      ("dynamic_batch", None, 4, 25, 16, 16, 5),
      ("different_depth_v", 3, 4, 25, 16, 17, 5),
  )
  def testLocalUnmaskedAttention1D(self, batch, heads, length,
                                   depth_k, depth_v, block_length):
    if batch is None:
      batch = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
    q = tf.random_normal([batch, heads, length, depth_k])
    k = tf.random_normal([batch, heads, length, depth_k])
    v = tf.random_normal([batch, heads, length, depth_v])
    output = common_attention.local_attention_1d(
        q, k, v, block_length=block_length, filter_width=3)
    if isinstance(batch, tf.Tensor):
      batch, res = self.evaluate([batch, output])
    else:
      res = self.evaluate(output)

    self.assertEqual(res.shape, (batch, heads, length, depth_v))

  @parameterized.named_parameters(
      ("matching_block_length", 3, 4, 25, 16, 16, (4, 4)),
      ("unmatching_block_length", 3, 4, 25, 16, 16, (5, 5)),
      ("dynamic_batch", None, 4, 25, 16, 16, (4, 4)),
      # TODO(trandustin): Extend function to enable depth_k != depth_v.
      # ("different_depth_v", 3, 4, 25, 16, 17, (4, 4)),
  )
  def testLocalUnmaskedAttention2D(self, batch, heads, length,
                                   depth_k, depth_v, query_shape):
    if batch is None:
      batch = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
    q = tf.random_normal([batch, heads, length, length, depth_k])
    k = tf.random_normal([batch, heads, length, length, depth_k])
    v = tf.random_normal([batch, heads, length, length, depth_v])
    output = common_attention.local_attention_2d(
        q,
        k,
        v,
        query_shape=query_shape,
        memory_flange=(3, 3))
    if isinstance(batch, tf.Tensor):
      batch, res = self.evaluate([batch, output])
    else:
      res = self.evaluate(output)

    self.assertEqual(res.shape, (batch, heads, length, length, depth_v))

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

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def test2dGatherAndScatterInvertibility(self):
    """2d gather and scatter invertibility test."""
    batch_size = 2
    num_heads = 2
    height = 4
    width = 6
    depth = 8
    query_shape = (2, 3)
    x = np.random.rand(batch_size, num_heads, height, width, depth)
    x_indices = common_attention.gather_indices_2d(
        x, query_shape, query_shape)
    gathered_x = common_attention.gather_blocks_2d(x, x_indices)
    x_shape = tf.constant([batch_size, num_heads, height, width, depth])
    scattered_x = common_attention.scatter_blocks_2d(
        gathered_x, x_indices, x_shape)
    res = self.evaluate(scattered_x)
    self.assertAllClose(x, res)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def test2dBlockRasterScanMask(self):
    """Testing the 2d block raster scan mask."""
    query_shape = (2, 3)
    memory_flange = (2, 1)
    mask = common_attention.make_2d_block_raster_mask(
        query_shape, memory_flange)
    res = self.evaluate(mask)
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

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
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

    x_indices = common_attention.gather_indices_2d(
        x, query_shape, query_shape)
    gathered_x = common_attention.gather_blocks_2d(x, x_indices)
    x_indices, gathered_x = self.evaluate([x_indices, gathered_x])
    self.assertAllEqual(correct_indices, x_indices)
    self.assertAllClose(correct_gathered_x, gathered_x)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
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
    x_indices = common_attention.gather_indices_2d(
        x, query_shape, query_shape)
    x_flange, x_center = common_attention.get_memory_region(
        tf.constant(x, dtype=tf.float32),
        query_shape,
        memory_flange,
        x_indices)
    [x_flange, x_center] = self.evaluate([x_flange, x_center])
    self.assertAllClose(correct_x_flange, x_flange)
    self.assertAllClose(correct_x_center, x_center)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
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
    x_indices = common_attention.gather_indices_2d(
        x, query_shape, query_shape)
    gathered_x = common_attention.get_shifted_center_blocks(
        tf.constant(x, dtype=tf.float32),
        x_indices)
    x_indices, gathered_x = self.evaluate([x_indices, gathered_x])
    self.assertAllClose(correct_gathered_x, gathered_x)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDotProductAttentionRelative(self):
    x = np.random.rand(5, 7, 12, 32)
    y = np.random.rand(5, 7, 12, 32)
    a = common_attention.dot_product_attention_relative(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        None,
        max_relative_position=3)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 7, 12, 32))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testRelativeAttentionV2(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 4, 16, 7)
    y = np.random.rand(5, 4, 16, 7)
    max_relative_position = 3
    a = common_attention.dot_product_self_attention_relative_v2(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        None,
        max_relative_position=max_relative_position,
        heads_share_relative_embedding=False)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 4, 16, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testRelativeAttentionV2SharedRel(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 4, 16, 7)
    y = np.random.rand(5, 4, 16, 7)
    max_relative_position = 3
    a = common_attention.dot_product_self_attention_relative_v2(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        None,
        max_relative_position=max_relative_position,
        heads_share_relative_embedding=True)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 4, 16, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testRelativeAttentionV2MaxRelativeLargerThanLength(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 4, 3, 7)
    y = np.random.rand(5, 4, 3, 7)
    max_relative_position = 16
    a = common_attention.dot_product_self_attention_relative_v2(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        None,
        max_relative_position=max_relative_position,
        heads_share_relative_embedding=False)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 4, 3, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDotProductUnMaskedAttentionRelativeV2(self):
    x = np.random.rand(5, 7, 12, 32)
    y = np.random.rand(5, 7, 12, 32)
    a = common_attention.dot_product_unmasked_self_attention_relative_v2(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        None,
        35)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 7, 12, 32))

  def python_relative_att(self, q, k, v, batch, num_heads, height, width,
                          depth, height_key_relative_embeddings,
                          width_key_relative_embeddings,
                          heads_share_relative_embedding):
    """Relative attention computation in numpy.

    For query index (i,j) and key index (l, m) the logit is
    q_i k_j^T + q_i rh_{l-i}^T + q_i rw_{m-j}^T, where rh and ry are the set of
    relative embeddings in height and width spatial dimensions, respectively.

    Args:
      q: [batch, heads, height, width, depth] tensor
      k: [batch, heads, height, width, depth] tensor
      v: [batch, heads, height, width, depth] tensor
      batch: int scalar
      num_heads: int scalar
      height: int scalar
      width: int scalar
      depth: int scalar
      height_key_relative_embeddings: a tensor of relative embeddings
      width_key_relative_embeddings: a tensor of relative embeddings
      heads_share_relative_embedding: a boolean

    Returns:
      att_output: A tensor
    """

    logits = np.zeros((batch, num_heads, height*width, height*width))
    for b in range(batch):
      for h in range(num_heads):
        for i in range(height*width):
          q_col = i%width
          q_row = int((i-q_col)/width)
          for j in range(height*width):
            k_col = j%width
            k_row = int((j-k_col)/width)
            logit = np.dot(q[b][h][q_row][q_col], k[b][h][k_row][k_col])
            width_rel_dist = k_col - q_col
            width_rel_index = width-1 + width_rel_dist
            if heads_share_relative_embedding:
              width_rel_logit = (
                  np.dot(q[b][h][q_row][q_col],
                         width_key_relative_embeddings[width_rel_index]))
            else:
              width_rel_logit = (
                  np.dot(q[b][h][q_row][q_col],
                         width_key_relative_embeddings[h][width_rel_index]))
            height_rel_dist = k_row - q_row
            height_rel_index = height-1 + height_rel_dist
            if heads_share_relative_embedding:
              height_rel_logit = (
                  np.dot(q[b][h][q_row][q_col],
                         height_key_relative_embeddings[height_rel_index]))
            else:
              height_rel_logit = (
                  np.dot(q[b][h][q_row][q_col],
                         height_key_relative_embeddings[h][height_rel_index]))
            logits[b, h, i, j] = logit + width_rel_logit + height_rel_logit
    # now to do a softmax across the logits
    att = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    # comparing the outputs
    att_output = np.matmul(att,
                           np.reshape(v, (
                               batch, num_heads, height*width, depth)))
    att_output = np.reshape(att_output,
                            (batch, num_heads, height, width, depth))
    return att_output

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDotProductUnMaskedAttentionRelative2d(self):
    batch = 1
    height = 3
    width = 3
    num_heads = 2
    max_relative_position = 6
    depth = 5
    heads_share_relative_embedding = False
    q = np.random.rand(batch, num_heads, height, width, depth)
    k = np.random.rand(batch, num_heads, height, width, depth)
    v = np.random.rand(batch, num_heads, height, width, depth)
    a = common_attention.dot_product_unmasked_self_attention_relative_2d(
        tf.constant(q, dtype=tf.float32),
        tf.constant(k, dtype=tf.float32),
        tf.constant(v, dtype=tf.float32),
        None,
        max_relative_position=max_relative_position,
        heads_share_relative_embedding=heads_share_relative_embedding)

    self.evaluate(tf.global_variables_initializer())
    res, height_key_relative_embeddings, width_key_relative_embeddings = (
        self.evaluate(a))
    att_output = self.python_relative_att(
        q, k, v, batch, num_heads, height, width, depth,
        height_key_relative_embeddings, width_key_relative_embeddings,
        heads_share_relative_embedding)
    self.assertEqual(res.shape, (batch, num_heads, height, width, depth))
    self.assertAllClose(res, att_output)

  @parameterized.parameters(
      (1, 10, 12, 2, 6, 3),
      (1, 1, 12, 2, 6, 3),
      (2, 10, 1, 2, 6, 3),
      (1, 10, 12, 2, 1, 1),
      (1, 10, 12, 2, 2, 8),
      (4, 10, 12, 2, 12, 10),
  )
  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testDotProductUnMaskedAttentionRelative2dSharedOneRow(
      self, batch, height, width, num_heads, max_relative_position, depth):
    heads_share_relative_embedding = True
    q = np.random.rand(batch, num_heads, height, width, depth)
    k = np.random.rand(batch, num_heads, height, width, depth)
    v = np.random.rand(batch, num_heads, height, width, depth)

    a = common_attention.dot_product_unmasked_self_attention_relative_2d(
        tf.constant(q, dtype=tf.float32),
        tf.constant(k, dtype=tf.float32),
        tf.constant(v, dtype=tf.float32),
        None,
        max_relative_position=max_relative_position,
        heads_share_relative_embedding=heads_share_relative_embedding)

    self.evaluate(tf.global_variables_initializer())
    (res, height_key_relative_embeddings,
     width_key_relative_embeddings) = self.evaluate(a)
    att_output = self.python_relative_att(
        q, k, v, batch, num_heads, height, width, depth,
        height_key_relative_embeddings, width_key_relative_embeddings,
        heads_share_relative_embedding)
    self.assertEqual(res.shape,
                     (batch, num_heads, height, width, depth))
    self.assertAllClose(res, att_output)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testRelativeAttentionV2Unmasked(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 4, 16, 7)
    y = np.random.rand(5, 4, 16, 7)
    max_relative_position = 3
    a = common_attention.dot_product_unmasked_self_attention_relative_v2(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        None,
        max_relative_position=max_relative_position,
        heads_share_relative_embedding=False)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 4, 16, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testRelativeAttentionV2UnmaskedSharedRel(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 4, 16, 7)
    y = np.random.rand(5, 4, 16, 7)
    max_relative_position = 3
    a = common_attention.dot_product_unmasked_self_attention_relative_v2(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        None,
        max_relative_position=max_relative_position,
        heads_share_relative_embedding=True)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 4, 16, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testRelativeAttentionV2UnmaskedRelativeLargerThanLength(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 4, 3, 7)
    y = np.random.rand(5, 4, 3, 7)
    max_relative_position = 16
    a = common_attention.dot_product_unmasked_self_attention_relative_v2(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        None,
        max_relative_position=max_relative_position,
        heads_share_relative_embedding=False)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 4, 3, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testMaskedRelativeLocalAttentionV2(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 4, 16, 7)
    y = np.random.rand(5, 4, 16, 7)
    block_length = 3
    a = common_attention.masked_relative_local_attention_1d(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        block_length=block_length,
        heads_share_relative_embedding=True,
        add_relative_to_values=False,
        name="masked_relative_local_attention_1d")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 4, 16, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testMaskedRelativeLocalAttentionV2AddRelativeValues(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 4, 16, 7)
    y = np.random.rand(5, 4, 16, 7)
    block_length = 3
    a = common_attention.masked_relative_local_attention_1d(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        block_length=block_length,
        heads_share_relative_embedding=True,
        add_relative_to_values=False,
        name="masked_relative_local_attention_1d")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 4, 16, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testMaskedRelativeLocalAttentionV2SeqShorterThanBlockLength(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 7, 2, 7)
    y = np.random.rand(5, 7, 2, 7)
    block_length = 3
    a = common_attention.masked_relative_local_attention_1d(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        block_length=block_length,
        heads_share_relative_embedding=True,
        name="masked_relative_local_attention_1d")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 7, 2, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testMaskedRelativeLocalAttentionV2SeqShorterThanTwiceBlockLength(self):
    # (batch, heads, length, depth)
    x = np.random.rand(5, 7, 5, 7)
    y = np.random.rand(5, 7, 5, 7)
    block_length = 3
    a = common_attention.masked_relative_local_attention_1d(
        tf.constant(x, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        tf.constant(y, dtype=tf.float32),
        block_length=block_length,
        heads_share_relative_embedding=True,
        name="masked_relative_local_attention_1d")
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(a)
    self.assertEqual(res.shape, (5, 7, 5, 7))

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
    self.assertAllClose(self.evaluate(bias), ground_truth)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
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
    self.assertAllClose(self.evaluate(bias), ground_truth)


if __name__ == "__main__":
  tf.test.main()

