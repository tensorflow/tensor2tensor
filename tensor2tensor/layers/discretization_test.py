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

"""Tests for tensor2tensor.layers.discretization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Dependency imports
import numpy as np
from tensor2tensor.layers import discretization
import tensorflow as tf


class DiscretizationTest(tf.test.TestCase):

  def setUp(self):
    tf.set_random_seed(1234)
    np.random.seed(123)

  def testBitToIntZeros(self):
    x_bit = tf.zeros(shape=[1, 10], dtype=tf.float32)
    x_int = tf.zeros(shape=[1], dtype=tf.int32)
    diff = discretization.bit_to_int(x_bit, num_bits=10) - x_int
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      d = sess.run(diff)
      self.assertEqual(d, 0)

  def testBitToIntOnes(self):
    x_bit = tf.ones(shape=[1, 3], dtype=tf.float32)
    x_int = 7 * tf.ones(shape=[1], dtype=tf.int32)
    diff = discretization.bit_to_int(x_bit, num_bits=3) - x_int
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      d = sess.run(diff)
      self.assertEqual(d, 0)

  def testIntToBitZeros(self):
    x_bit = tf.zeros(shape=[1, 10], dtype=tf.float32)
    x_int = tf.zeros(shape=[1], dtype=tf.int32)
    diff = discretization.int_to_bit(x_int, num_bits=10) - x_bit
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      d = sess.run(diff)
      self.assertTrue(np.all(d == 0))

  def testIntToBitOnes(self):
    x_bit = tf.ones(shape=[1, 3], dtype=tf.float32)
    x_int = 7 * tf.ones(shape=[1], dtype=tf.int32)
    diff = discretization.int_to_bit(x_int, num_bits=3) - x_bit
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      d = sess.run(diff)
      self.assertTrue(np.all(d == 0))

  def testProjectHidden(self):
    hidden_size = 60
    block_dim = 20
    num_blocks = 3
    x = tf.zeros(shape=[1, hidden_size], dtype=tf.float32)
    projection_tensors = tf.random_normal(
        shape=[num_blocks, hidden_size, block_dim], dtype=tf.float32)
    x_projected = discretization.project_hidden(x, projection_tensors,
                                                hidden_size, num_blocks)
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      x_projected_eval = sess.run(x_projected)
      self.assertEqual(np.shape(x_projected_eval), (1, num_blocks, block_dim))
      self.assertTrue(np.all(x_projected_eval == 0))

  def testSliceHiddenZeros(self):
    hidden_size = 60
    block_dim = 20
    num_blocks = 3
    x = tf.zeros(shape=[1, hidden_size], dtype=tf.float32)
    x_sliced = discretization.slice_hidden(x, hidden_size, num_blocks)
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      x_sliced_eval = sess.run(x_sliced)
      self.assertEqual(np.shape(x_sliced_eval), (1, num_blocks, block_dim))
      self.assertTrue(np.all(x_sliced_eval == 0))

  def testSliceHiddenOnes(self):
    hidden_size = 60
    block_dim = 20
    num_blocks = 3
    x = tf.ones(shape=[1, hidden_size], dtype=tf.float32)
    x_sliced = discretization.slice_hidden(x, hidden_size, num_blocks)
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      x_sliced_eval = sess.run(x_sliced)
      self.assertEqual(np.shape(x_sliced_eval), (1, num_blocks, block_dim))
      self.assertTrue(np.all(x_sliced_eval == 1))

  def testNearestNeighbors(self):
    x = tf.constant([[0, 0.9, 0], [0.8, 0., 0.]], dtype=tf.float32)
    x = tf.expand_dims(x, axis=0)
    means = tf.constant(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]], dtype=tf.float32)
    means = tf.stack([means, means], axis=0)
    x_means_hot = discretization.nearest_neighbor(x, means, block_v_size=4)
    x_means_hot_test = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    x_means_hot_test = np.expand_dims(x_means_hot_test, axis=0)
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      x_means_hot_eval = sess.run(x_means_hot)
      self.assertEqual(np.shape(x_means_hot_eval), (1, 2, 4))
      self.assertTrue(np.all(x_means_hot_eval == x_means_hot_test))


if __name__ == '__main__':
  tf.test.main()
