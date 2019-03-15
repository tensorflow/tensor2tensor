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

"""The memory unit for Transformer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class TransformerMemory(object):
  """Implements the Memory module.

  Based on Neural Turing Machines: arXiv:1410.5401 [cs.NE]
  """

  def __init__(self, batch_size, key_depth, val_depth, memory_size,
               sharpen_factor=1.):
    """Initialize the memory object.

    Args:
      batch_size: the batch size.
      key_depth: the depth of the memory keys.
      val_depth: the depth of the memory values.
      memory_size: the number of items in the memory.
      sharpen_factor: the sharpen_factor for addressing the memory.
    """
    self.batch_size = batch_size
    self.key_depth = key_depth
    self.val_depth = val_depth
    self.memory_size = memory_size
    self.sharpen_factor = sharpen_factor
    self.mem_vals = tf.get_variable(
        "memvals", [self.batch_size, self.memory_size, self.val_depth],
        dtype=tf.float32, trainable=False,
        initializer=tf.constant_initializer(.0))
    self.mean_logits = tf.get_variable(
        "meanlogits", [self.batch_size, self.memory_size],
        dtype=tf.float32, trainable=False,
        initializer=tf.constant_initializer(.0))

  def _address_content(self, x):
    """Address the memory based on content similarity.

    Args:
      x: a tensor in the shape of [batch_size, length, depth].
    Returns:
      the logits for each memory entry [batch_size, length, memory_size].
    """
    mem_keys = tf.layers.dense(self.mem_vals, self.key_depth, name="mem_key")
    mem_query = tf.layers.dense(x, self.key_depth, name="mem_query")
    norm = tf.matmul(
        tf.norm(mem_query, axis=-1, keepdims=True),
        tf.norm(mem_keys, axis=-1, keepdims=True), transpose_b=True)
    cos_dist = tf.div(
        tf.matmul(mem_query, mem_keys, transpose_b=True), norm,
        name="cos_dist")
    access_logits = self.sharpen_factor * cos_dist
    return access_logits

  def read(self, x):
    """Read from the memory.

    An external component can use the results via a simple MLP,
    e.g., fn(x W_x + retrieved_mem W_m).

    Args:
      x: a tensor in the shape of [batch_size, length, depth].
    Returns:
      access_logits: the logits for accessing the memory in shape of
          [batch_size, length, memory_size].
      retrieved_mem: the retrieved results in the shape of
          [batch_size, length, val_depth].
    """
    access_logits = self._address_content(x)
    weights = tf.nn.softmax(access_logits)
    retrieved_mem = tf.reduce_sum(
        tf.multiply(tf.expand_dims(weights, 3),
                    tf.expand_dims(self.mem_vals, axis=1)), axis=2)
    return access_logits, retrieved_mem

  def write(self, x, access_logits):
    """Write to the memory based on a combination of similarity and least used.

    Based on arXiv:1607.00036v2 [cs.LG].

    Args:
      x: a tensor in the shape of [batch_size, length, depth].
      access_logits: the logits for accessing the memory.
    Returns:
      the update op.
    """
    gamma = tf.layers.dense(x, 1, activation=tf.sigmoid, name="gamma")
    write_logits = access_logits - gamma * tf.expand_dims(self.mean_logits, 1)
    candidate_value = tf.layers.dense(x, self.val_depth,
                                      activation=tf.nn.relu,
                                      name="candidate_value")
    erase_gates = tf.layers.dense(x, self.memory_size,
                                  activation=tf.nn.sigmoid,
                                  name="erase")
    write_weights = tf.nn.softmax(write_logits)
    erase = tf.multiply(tf.expand_dims(1 - erase_gates * write_weights, 3),
                        tf.expand_dims(self.mem_vals, 1))
    addition = tf.multiply(
        tf.expand_dims(write_weights, 3), tf.expand_dims(candidate_value, 2))
    update_value_op = self.mem_vals.assign(
        tf.reduce_sum(erase + addition, axis=1))
    with tf.control_dependencies([update_value_op]):
      write_op = self.mean_logits.assign(
          self.mean_logits * 0.1 + tf.reduce_sum(write_logits * 0.9, axis=1))
      return write_op

  def set(self, mem_vals, mean_logits):
    set_op = tf.group([
        self.mem_vals.assign(mem_vals),
        self.mean_logits.assign(mean_logits)])
    return set_op

  def get(self):
    return self.mem_vals, self.mean_logits

  def reset(self, entries_to_reset):
    """Reset the entries in the memory.

    Args:
      entries_to_reset: a 1D tensor.
    Returns:
      the reset op.
    """
    num_updates = tf.size(entries_to_reset)
    update_vals = tf.scatter_update(
        self.mem_vals, entries_to_reset,
        tf.tile(tf.expand_dims(
            tf.fill([self.memory_size, self.val_depth], .0), 0),
                [num_updates, 1, 1]))
    update_logits = tf.scatter_update(
        self.mean_logits, entries_to_reset,
        tf.tile(tf.expand_dims(
            tf.fill([self.memory_size], .0), 0),
                [num_updates, 1]))
    reset_op = tf.group([update_vals, update_logits])
    return reset_op
