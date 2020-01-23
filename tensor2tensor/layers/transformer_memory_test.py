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

"""Tests for tensor2tensor.layers.transformer_memory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2tensor.layers import transformer_memory
import tensorflow.compat.v1 as tf


class TransformerMemoryTest(parameterized.TestCase, tf.test.TestCase):

  def testRead(self):
    batch_size = 2
    key_depth = 3
    val_depth = 5
    memory_size = 4
    window_size = 6
    x_depth = 10
    memory = transformer_memory.TransformerMemory(
        batch_size, key_depth, val_depth, memory_size)
    x = tf.random_uniform([batch_size, window_size, x_depth], minval=1.0)
    vals = tf.random_uniform([batch_size, memory_size, val_depth], minval=1.0)
    logits = tf.random_uniform([batch_size, memory_size], minval=1.0)
    update_op = memory.set(vals, logits)
    with tf.control_dependencies([update_op]):
      logits, retrieved_values = memory.read(x)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      logits_values, values = session.run([logits, retrieved_values])
    self.assertAllEqual([batch_size, window_size, memory_size],
                        logits_values.shape)
    self.assertAllEqual([batch_size, window_size, val_depth], values.shape)

  def testWrite(self):
    batch_size = 2
    key_depth = 3
    val_depth = 5
    memory_size = 4
    window_size = 6
    x_depth = 10
    memory = transformer_memory.TransformerMemory(
        batch_size, key_depth, val_depth, memory_size)
    x = tf.random_uniform([batch_size, window_size, x_depth], minval=1.0)
    vals = tf.random_uniform([batch_size, memory_size, val_depth], minval=1.0)
    logits = tf.random_uniform([batch_size, memory_size], minval=1.0)
    update_op = memory.set(vals, logits)
    with tf.control_dependencies([update_op]):
      logits, _ = memory.read(x)
      write_op = memory.write(x, logits)
    mem_vals, mem_logits = memory.get()
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(write_op)
      updated_vals, updated_logits = session.run([mem_vals, mem_logits])
    self.assertAllEqual([batch_size, memory_size, val_depth],
                        updated_vals.shape)
    self.assertAllEqual([batch_size, memory_size], updated_logits.shape)

  def testReset(self):
    batch_size = 2
    key_depth = 3
    val_depth = 5
    memory_size = 4
    memory = transformer_memory.TransformerMemory(
        batch_size, key_depth, val_depth, memory_size)
    vals = tf.random_uniform([batch_size, memory_size, val_depth], minval=1.0)
    logits = tf.random_uniform([batch_size, memory_size], minval=1.0)
    update_op = memory.set(vals, logits)
    reset_op = memory.reset([1])
    mem_vals, mem_logits = memory.get()
    assert_op1 = tf.assert_equal(mem_vals[0], vals[0])
    assert_op2 = tf.assert_equal(mem_logits[0], logits[0])
    with tf.control_dependencies([assert_op1, assert_op2]):
      all_zero1 = tf.reduce_sum(tf.abs(mem_vals[1]))
      all_zero2 = tf.reduce_sum(tf.abs(mem_logits[1]))
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(update_op)
      session.run(reset_op)
      zero1, zero2 = session.run([all_zero1, all_zero2])
    self.assertAllEqual(0, zero1)
    self.assertAllEqual(0, zero2)

  def testLoss(self):
    batch_size = 2
    key_depth = 5
    val_depth = 5
    memory_size = 4
    window_size = 3
    x_depth = 5
    memory = transformer_memory.TransformerMemory(
        batch_size, key_depth, val_depth, memory_size)
    x = tf.random_uniform([batch_size, window_size, x_depth], minval=.0)
    memory_results, _, _, _ = (
        memory.pre_attention(
            tf.random_uniform([batch_size], minval=0, maxval=1, dtype=tf.int32),
            x, None, None))
    x = memory.post_attention(memory_results, x)
    with tf.control_dependencies([tf.print("x", x)]):
      is_nan = tf.reduce_any(tf.math.is_nan(x))
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      for _ in range(100):
        is_nan_value, _ = session.run([is_nan, x])
    self.assertEqual(is_nan_value, False)

if __name__ == "__main__":
  tf.test.main()
