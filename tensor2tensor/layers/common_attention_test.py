# coding=utf-8
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

"""Tests for common attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

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

  def test2dGatherAndScatter(self):
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

if __name__ == "__main__":
  tf.test.main()
