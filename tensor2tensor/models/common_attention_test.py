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
from tensor2tensor.models import common_attention

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

  def testMaskedLocalAttention(self):
    q = np.array([[[[1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0]]]])
    k = np.array([[[[1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0]]]])
    v = np.ones((1, 1, 8, 1))
    with self.test_session() as session:
      q_ = tf.constant(q, dtype=tf.float32)
      k_ = tf.constant(k, dtype=tf.float32)
      v_ = tf.constant(v, dtype=tf.float32)
      y = common_attention.masked_local_attention_1d(
          q_, k_, v_, block_length=tf.constant(2))
      res = session.run(y)

    self.assertEqual(res.shape, (1, 1, 8, 1))

  def testLocalUnmaskedAttention(self):
    x = np.random.rand(5, 4, 25, 16)
    y = np.random.rand(5, 4, 25, 16)
    with self.test_session() as session:
      a = common_attention.unmasked_local_attention_1d(
          tf.constant(x, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          block_length=4, filter_width=3)
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 4, 25, 16))

  def testLocalUnmaskedAttentionMatchingBlockLength(self):
    x = np.random.rand(5, 4, 25, 16)
    y = np.random.rand(5, 4, 25, 16)
    with self.test_session() as session:
      a = common_attention.unmasked_local_attention_1d(
          tf.constant(x, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          tf.constant(y, dtype=tf.float32),
          block_length=5, filter_width=3)
      session.run(tf.global_variables_initializer())
      res = session.run(a)
    self.assertEqual(res.shape, (5, 4, 25, 16))


if __name__ == "__main__":
  tf.test.main()
