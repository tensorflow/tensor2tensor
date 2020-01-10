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

"""Tests for discretization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.layers import discretization
from tensor2tensor.utils import test_utils

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


class DiscretizationTest(tf.test.TestCase):
  """Tests for discretization layers."""

  def setUp(self):
    tf.set_random_seed(1234)
    np.random.seed(123)

  @test_utils.run_in_graph_and_eager_modes()
  def testBitToIntZeros(self):
    x_bit = tf.zeros(shape=[1, 10], dtype=tf.float32)
    x_int = tf.zeros(shape=[1], dtype=tf.int32)
    diff = discretization.bit_to_int(x_bit, num_bits=10) - x_int
    d = self.evaluate(diff)
    self.assertEqual(d, 0)

  @test_utils.run_in_graph_and_eager_modes()
  def testBitToIntOnes(self):
    x_bit = tf.ones(shape=[1, 3], dtype=tf.float32)
    x_int = 7 * tf.ones(shape=[1], dtype=tf.int32)
    diff = discretization.bit_to_int(x_bit, num_bits=3) - x_int
    d = self.evaluate(diff)
    self.assertEqual(d, 0)

  @test_utils.run_in_graph_and_eager_modes()
  def testIntToBitZeros(self):
    x_bit = tf.zeros(shape=[1, 10], dtype=tf.float32)
    x_int = tf.zeros(shape=[1], dtype=tf.int32)
    diff = discretization.int_to_bit(x_int, num_bits=10) - x_bit
    d = self.evaluate(diff)
    self.assertTrue(np.all(d == 0))

  @test_utils.run_in_graph_and_eager_modes()
  def testIntToBitOnes(self):
    x_bit = tf.ones(shape=[1, 3], dtype=tf.float32)
    x_int = 7 * tf.ones(shape=[1], dtype=tf.int32)
    diff = discretization.int_to_bit(x_int, num_bits=3) - x_bit
    d = self.evaluate(diff)
    self.assertTrue(np.all(d == 0))

  @test_utils.run_in_graph_and_eager_modes()
  def testProjectHidden(self):
    hidden_size = 60
    block_dim = 20
    num_blocks = 3
    x = tf.zeros(shape=[1, 1, hidden_size], dtype=tf.float32)
    projection_tensors = tf.random_normal(
        shape=[num_blocks, hidden_size, block_dim], dtype=tf.float32)
    x_projected = discretization.project_hidden(x, projection_tensors,
                                                hidden_size, num_blocks)
    x_projected_eval = self.evaluate(x_projected)
    self.assertEqual(np.shape(x_projected_eval), (1, 1, num_blocks, block_dim))
    self.assertTrue(np.all(x_projected_eval == 0))

  @test_utils.run_in_graph_and_eager_modes()
  def testSliceHiddenZeros(self):
    hidden_size = 60
    block_dim = 20
    num_blocks = 3
    x = tf.zeros(shape=[1, 1, hidden_size], dtype=tf.float32)
    x_sliced = discretization.slice_hidden(x, hidden_size, num_blocks)
    x_sliced_eval = self.evaluate(x_sliced)
    self.assertEqual(np.shape(x_sliced_eval), (1, 1, num_blocks, block_dim))
    self.assertTrue(np.all(x_sliced_eval == 0))

  @test_utils.run_in_graph_and_eager_modes()
  def testSliceHiddenOnes(self):
    hidden_size = 60
    block_dim = 20
    num_blocks = 3
    x = tf.ones(shape=[1, 1, hidden_size], dtype=tf.float32)
    x_sliced = discretization.slice_hidden(x, hidden_size, num_blocks)
    x_sliced_eval = self.evaluate(x_sliced)
    self.assertEqual(np.shape(x_sliced_eval), (1, 1, num_blocks, block_dim))
    self.assertTrue(np.all(x_sliced_eval == 1))

  @test_utils.run_in_graph_and_eager_modes()
  def testNearestNeighbors(self):
    x = tf.constant([[0, 0.9, 0], [0.8, 0., 0.]], dtype=tf.float32)
    x = tf.reshape(x, [1, 1, 2, 3])
    means = tf.constant(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]], dtype=tf.float32)
    means = tf.stack([means, means], axis=0)
    x_means_hot, _ = discretization.nearest_neighbor(
        x, means, block_v_size=4)
    x_means_hot_test = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    x_means_hot_test = np.expand_dims(x_means_hot_test, axis=0)
    x_means_hot_eval = self.evaluate(x_means_hot)
    self.assertEqual(np.shape(x_means_hot_eval), (1, 2, 4))
    self.assertTrue(np.all(x_means_hot_eval == x_means_hot_test))

  @test_utils.run_in_graph_mode_only()
  def testGetVQBottleneck(self):
    bottleneck_bits = 2
    bottleneck_size = 2**bottleneck_bits
    hidden_size = 3
    means, _, ema_count = discretization.get_vq_codebook(
        bottleneck_size, hidden_size)
    assign_op = means.assign(tf.zeros(shape=[bottleneck_size, hidden_size]))
    means_new, _, _ = discretization.get_vq_codebook(bottleneck_size,
                                                     hidden_size)
    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      sess.run(assign_op)
      self.assertTrue(np.all(sess.run(means_new) == 0))
      self.assertTrue(np.all(sess.run(ema_count) == 0))

  @test_utils.run_in_graph_and_eager_modes()
  def testVQNearestNeighbors(self):
    x = tf.constant([[0, 0.9, 0], [0.8, 0., 0.]], dtype=tf.float32)
    means = tf.constant(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]], dtype=tf.float32)
    x_means_hot, _, _ = discretization.vq_nearest_neighbor(x, means)
    x_means_hot_test = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
    x_means_hot_eval = self.evaluate(x_means_hot)
    self.assertEqual(np.shape(x_means_hot_eval), (2, 4))
    self.assertTrue(np.all(x_means_hot_eval == x_means_hot_test))

  def testVQDiscreteBottleneck(self):
    x = tf.constant([[0, 0.9, 0], [0.8, 0., 0.]], dtype=tf.float32)
    x_means_hot, _ = discretization.vq_discrete_bottleneck(x, bottleneck_bits=2)
    self.evaluate(tf.global_variables_initializer())
    x_means_hot_eval = self.evaluate(x_means_hot)
    self.assertEqual(np.shape(x_means_hot_eval), (2, 4))

  def testVQDiscreteUnbottlenck(self):
    x = tf.constant([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=tf.int32)
    x_means = discretization.vq_discrete_unbottleneck(x, hidden_size=3)
    self.evaluate(tf.global_variables_initializer())
    x_means_eval = self.evaluate(x_means)
    self.assertEqual(np.shape(x_means_eval), (2, 3))

  def testGumbelSoftmaxDiscreteBottleneck(self):
    x = tf.constant([[0, 0.9, 0], [0.8, 0., 0.]], dtype=tf.float32)
    tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, tf.constant(1))
    x_means_hot, _ = discretization.gumbel_softmax_discrete_bottleneck(
        x, bottleneck_bits=2)
    self.evaluate(tf.global_variables_initializer())
    x_means_hot_eval = self.evaluate(x_means_hot)
    self.assertEqual(np.shape(x_means_hot_eval), (2, 4))

  @test_utils.run_in_graph_mode_only()
  def testDiscreteBottleneckVQ(self):
    hidden_size = 60
    z_size = 4
    x = tf.zeros(shape=[100, 1, hidden_size], dtype=tf.float32)
    with tf.variable_scope("test", reuse=tf.AUTO_REUSE):
      means = tf.get_variable("means",
                              shape=[1, 1, 2**z_size, hidden_size],
                              initializer=tf.constant_initializer(0.),
                              dtype=tf.float32)
      ema_count = []
      ema_count_i = tf.get_variable(
          "ema_count",
          [1, 2**z_size],
          initializer=tf.constant_initializer(0),
          trainable=False)
      ema_count.append(ema_count_i)
      ema_means = []
      with tf.colocate_with(means):
        ema_means_i = tf.get_variable("ema_means",
                                      initializer=means.initialized_value()[0],
                                      trainable=False)
        ema_means.append(ema_means_i)
      x_means_dense, x_means_hot, _, _, _ = discretization.discrete_bottleneck(
          x, hidden_size, z_size, 32, means=means, num_blocks=1,
          ema_means=ema_means, ema_count=ema_count, name="test")
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        x_means_dense_eval, x_means_hot_eval = sess.run(
            [x_means_dense, x_means_hot])
        means_eval = sess.run(means)
      self.assertEqual(x_means_dense_eval.shape, (100, 1, hidden_size))
      self.assertEqual(x_means_hot_eval.shape, (100, 1))
      self.assertTrue(np.all(means_eval == np.zeros(
          (1, 1, 2**z_size, hidden_size))))

  @test_utils.run_in_graph_mode_only()
  def testDiscreteBottleneckVQCond(self):
    hidden_size = 60
    z_size = 4
    x = tf.zeros(shape=[100, 1, hidden_size], dtype=tf.float32)
    with tf.variable_scope("test2", reuse=tf.AUTO_REUSE):
      means = tf.get_variable("means",
                              shape=[1, 1, 2**z_size, hidden_size],
                              initializer=tf.constant_initializer(0.),
                              dtype=tf.float32)
      ema_count = []
      ema_count_i = tf.get_variable(
          "ema_count",
          [1, 2**z_size],
          initializer=tf.constant_initializer(0),
          trainable=False)
      ema_count.append(ema_count_i)
      ema_means = []
      with tf.colocate_with(means):
        ema_means_i = tf.get_variable("ema_means",
                                      initializer=means.initialized_value()[0],
                                      trainable=False)
        ema_means.append(ema_means_i)
      cond = tf.cast(0.0, tf.bool)
      x_means_dense, x_means_hot, _, _, _ = discretization.discrete_bottleneck(
          x, hidden_size, z_size, 32, means=means, num_blocks=1, cond=cond,
          ema_means=ema_means, ema_count=ema_count, name="test2")
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        x_means_dense_eval, x_means_hot_eval = sess.run(
            [x_means_dense, x_means_hot])
        means_eval = sess.run(means)
      self.assertEqual(x_means_dense_eval.shape, (100, 1, hidden_size))
      self.assertEqual(x_means_hot_eval.shape, (100, 1))
      self.assertAllClose(means_eval, np.zeros((1, 1, 2**z_size,
                                                hidden_size)))


if __name__ == "__main__":
  tf.test.main()
