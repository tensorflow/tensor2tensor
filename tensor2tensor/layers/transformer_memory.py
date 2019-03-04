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

"""The memory unit for remembering a sequence as a collection of clusters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
import tensorflow as tf


class TransformerMemory(object):
  """Implements the Memory module.

  It compresses a sequence by storing items into appropriate clusters.
  A single item can be allocated into multiple clusters like a mixture model.
  Each vector in the memory represents the centroid of the cluster that is
  updated in an online fashion. The memory also keeps the total amount of
  probability mass that is used for updating each item that indicates the amount
  of change that has been made to each cluster.
  """

  def __init__(self, batch_size, feature_dim, memory_size):
    """Initialize the memory object.

    Args:
      batch_size: the batch size.
      feature_dim: the depth of the feature.
      memory_size: the number of clusters to maintain in the memory, which does
          not have to be the same as the segment length.
    """
    self.feature_dim = feature_dim
    self.batch_size = batch_size
    self.memory_size = memory_size
    self.mem_vals = tf.get_variable(
        "memvals", [self.batch_size, self.memory_size, self.feature_dim],
        dtype=tf.float32, trainable=False,
        initializer=tf.constant_initializer(.0))
    self.mem_times = tf.get_variable(
        "memtimes", [self.batch_size, self.memory_size], dtype=tf.float32,
        trainable=False, initializer=tf.constant_initializer(.0))
    self.seq_length_so_far = tf.get_variable(
        "seqlensofar", [self.batch_size], dtype=tf.int32,
        trainable=False, initializer=tf.constant_initializer(0))

  def set(self, mem_vals, mem_times, seq_length_so_far):
    set_op = tf.group([
        self.mem_vals.assign(mem_vals),
        self.mem_times.assign(mem_times),
        self.seq_length_so_far.assign(seq_length_so_far)])
    return set_op

  def get(self):
    return self.mem_vals, self.mem_times, self.seq_length_so_far

  def incremental_update(self, event):
    """Add a new event to the memory and also advance the time.

    Args:
      event: a tensor in the shape of [batch_size, depth].
    Returns:
      the update op.
    """
    event = tf.expand_dims(event, 1)
    similarity_logits = tf.matmul(event, tf.transpose(
        self.mem_vals, [0, 2, 1]))
    similarity_logits = tf.squeeze(similarity_logits, [1])
    max_logits = tf.reduce_max(similarity_logits, -1, keep_dims=True)
    similarity_logits = tf.where(
        tf.less(self.mem_times, 0.5),
        tf.tile(max_logits, [1, self.memory_size]) + 1.0,
        similarity_logits)
    _, indices = tf.nn.top_k(similarity_logits)
    update_mask = tf.cast(tf.one_hot(indices, self.memory_size), tf.float32)
    update_times = self.mem_times.assign_add(update_mask)
    with tf.control_dependencies([update_times]):
      add_to_vals = tf.where(
          tf.cast(update_mask, tf.bool),
          tf.zeros_like(self.mem_vals),
          tf.div(event - self.mem_vals, tf.expand_dims(self.mem_times, 2)))
      return self.mem_vals.assign_add(add_to_vals)

  def update(self, segment):
    """Update the memory given the segment of events.

    It might be useful to consider adding a decay to each cluster to favor
    recent events.

    Args:
      segment: a tensor of shape [batch_size, segment_length, depth].
    Returns:
      the update op.
    """
    attention_logits = tf.matmul(segment, tf.transpose(
        self.mem_vals, [0, 2, 1]))
    alloc_probs = tf.nn.softmax(attention_logits)
    aggregated_alloc_probs = tf.reduce_sum(alloc_probs, axis=1)
    time_increment = tf.where(
        tf.equal(self.seq_length_so_far, 0),
        tf.ones_like(self.mem_times),
        aggregated_alloc_probs)
    update_times = self.mem_times.assign_add(time_increment)
    with tf.control_dependencies([update_times]):
      allocations = tf.multiply(
          tf.expand_dims(alloc_probs, 3), tf.expand_dims(segment, 2))
      allocations = tf.reduce_sum(allocations, axis=1)
      add_to_vals = tf.where(
          tf.equal(self.seq_length_so_far, 0),
          segment,
          tf.div(allocations - self.mem_vals,
                 tf.expand_dims(self.mem_times, 2)))
      update_vals = self.mem_vals.assign_add(add_to_vals)
      with tf.control_dependencies([update_vals]):
        segment_length = common_layers.shape_list(segment)[1]
        update_seq_length = self.seq_length_so_far.assign_add(
            tf.tile(tf.expand_dims(segment_length, 0), [self.batch_size]))
    return update_seq_length

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
            tf.fill([self.memory_size, self.feature_dim], .0), 0),
                [num_updates, 1, 1]))
    update_times = tf.scatter_update(
        self.mem_times, entries_to_reset,
        tf.tile(tf.expand_dims(
            tf.fill([self.memory_size], .0), 0), [num_updates, 1]))
    update_segs = tf.scatter_update(
        self.seq_length_so_far, entries_to_reset, tf.fill([num_updates], 0))
    reset_op = tf.group([update_vals, update_times, update_segs])
    return reset_op
