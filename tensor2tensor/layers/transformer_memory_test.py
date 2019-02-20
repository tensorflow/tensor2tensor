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

"""Tests for tensor2tensor.layers.transformer_memory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2tensor.layers import transformer_memory
import tensorflow as tf


class TransformerMemoryTest(parameterized.TestCase, tf.test.TestCase):

  def testInitialize(self):
    batch_size = 2
    feature_dim = 3
    memory_size = 4
    memory = transformer_memory.TransformerMemory(
        batch_size, feature_dim, memory_size)
    segment = tf.constant([[[1., 2., 3.], [1., 1., 1.],
                            [3., 2., 1.], [2., 2., 2.]],
                           [[3., 3., 3.], [1., 2., 3.],
                            [3., 2., 1.], [2., 2., 2.]]])
    update_op = memory.update(segment)
    mem_vals, mem_times, mem_len_so_far = memory.get()
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(update_op)
      vals, times, length_so_far = session.run([
          mem_vals, mem_times, mem_len_so_far])
    self.assertAllEqual([[[1., 2., 3.], [1., 1., 1.],
                          [3., 2., 1.], [2., 2., 2.]],
                         [[3., 3., 3.], [1., 2., 3.],
                          [3., 2., 1.], [2., 2., 2.]]], vals)
    self.assertAllEqual([[1., 1., 1., 1.], [1., 1., 1., 1.]], times)
    self.assertAllEqual([4, 4], length_so_far)

  def testUpdate(self):
    batch_size = 2
    feature_dim = 3
    memory_size = 4
    memory = transformer_memory.TransformerMemory(
        batch_size, feature_dim, memory_size)
    segment = tf.constant([[[1., 2., 3.], [2., 2., 2.],
                            [3., 2., 1.], [2., 2., 2.]],
                           [[2., 2., 2.], [1., 2., 3.],
                            [3., 2., 1.], [2., 2., 2.]]])
    init_op = memory.set(segment, [[1., 2., 3., 4.], [2., 1., 5., 1.]],
                         [10, 9])
    new_segment = tf.constant(
        [[[1., 2., 3.], [3., 2., 1.],
          [2., 2., 2.], [2., 2., 2.]],
         [[2., 2., 2.], [1., 2., 3.],
          [3., 2., 1.], [2., 2., 2.]]])
    update_op = memory.update(new_segment)
    mem_vals, mem_times, mem_len_so_far = memory.get()
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(init_op)
      session.run(update_op)
      vals, times, length_so_far = session.run([
          mem_vals, mem_times, mem_len_so_far])
      print(vals, times, length_so_far)
    self.assertAllEqual([2, 4, 3], vals.shape)
    self.assertAllEqual([2, 4], times.shape)
    self.assertAllEqual([14, 13], length_so_far)

  def testReset(self):
    batch_size = 2
    feature_dim = 3
    memory_size = 4
    memory = transformer_memory.TransformerMemory(
        batch_size, feature_dim, memory_size)
    segment = tf.constant([[[1., 2., 3.], [1., 1., 1.],
                            [3., 2., 1.], [2., 2., 2.]],
                           [[3., 3., 3.], [1., 2., 3.],
                            [3., 2., 1.], [2., 2., 2.]]])
    update_op = memory.set(segment, [[1., 2., 3., 4.], [2., 1., 5., 1.]],
                           [10, 9])
    reset_op = memory.reset([1])
    mem_vals, mem_times, mem_len_so_far = memory.get()
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(update_op)
      session.run(reset_op)
      vals, times, length_so_far = session.run([
          mem_vals, mem_times, mem_len_so_far])
    self.assertAllEqual([[[1., 2., 3.], [1., 1., 1.],
                          [3., 2., 1.], [2., 2., 2.]],
                         [[0., 0., 0.], [0., 0., 0.],
                          [0., 0., 0.], [0., 0., 0.]]], vals)
    self.assertAllEqual([[1., 2., 3., 4.], [0., 0., 0., 0.]], times)
    self.assertAllEqual([10, 0], length_so_far)

if __name__ == "__main__":
  tf.test.main()
