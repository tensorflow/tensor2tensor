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
"""Utilities for attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

import tensorflow as tf


def compute_mpnn_qkv(node_states,
                     total_key_depth,
                     total_value_depth,
                     num_edge_types,
                     ignore_zero=True):
  """Computes query, key and value for edge matrices.

  Args:
    node_states: a Tensor with shape [batch, num_nodes, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    num_edge_types: a integer specifying number of edge types
    ignore_zero: If true, then edge type 0 will not be considered. Equivalent
      to have a linear transformation of all 0's for edge type 0
  Returns:
    q: [batch, num_nodes, channels]
    k: [batch, num_nodes * num_edge_types, channels]
    v: [batch, num_nodes * num_edge_types, channels]
  """
  memory_antecedent = node_states
  def _compute(inp, depth, filter_width, padding, name):
    if filter_width == 1:
      return common_layers.dense(inp, depth, use_bias=False, name=name)
    else:
      return common_layers.conv1d(inp, depth, filter_width, padding, name=name)
  # For edge type 0, if ignore_zero, don't multiply with linear transformation,
  # but just concat a bunch of 0's not only for efficiency but to make
  # sure that it doesn't contribute anything to the terms
  # TODO(avaswani): Better way to do this.
  q = _compute(node_states, total_key_depth, 1, "VALID", "q_mpnn")
  q_shape = common_layers.shape_list(q)
  # k and v edge transforms have shape
  # [batch, length, depth*nonignored_edge_types]
  nonignored_edge_types = num_edge_types-int(ignore_zero)
  k = _compute(memory_antecedent, total_key_depth*nonignored_edge_types, 1,
               "VALID", "k_mpnn")
  v = _compute(memory_antecedent, total_value_depth*nonignored_edge_types,
               1, "VALID", "v_mpnn")
  batch = q_shape[0]
  length = q_shape[1]
  k = tf.reshape(k,
                 [batch, length, nonignored_edge_types, total_key_depth])
  v = tf.reshape(v,
                 [q_shape[0], q_shape[1], nonignored_edge_types,
                  total_value_depth])
  if ignore_zero:
    k = tf.pad(k, [[0, 0], [0, 0], [1, 0], [0, 0]])
    v = tf.pad(v, [[0, 0], [0, 0], [1, 0], [0, 0]])

  k = tf.reshape(k,
                 [q_shape[0], q_shape[1]*num_edge_types, total_key_depth])
  v = tf.reshape(v,
                 [q_shape[0], q_shape[1]*num_edge_types, total_value_depth])
  return q, k, v


def multihead_mpnn_attention(node_states,
                             total_key_depth,
                             total_value_depth,
                             output_depth,
                             num_heads,
                             adjacency_matrix=None,
                             num_edge_types=5,
                             ignore_zero=True,
                             name="mpnn_attention"):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    node_states: A tensor of shape [batch, length, depth]
    total_key_depth: An integer for key dimension
    total_value_depth: An integer for value dimensions
    output_depth: An intger for output dimemsions
    num_heads: An integer
    adjacency_matrix: An tensor of ints of shape [batch, length, length]
    num_edge_types: An integer indicating number of edge bins
    ignore_zero: A flag that says that edge type 0 should be ignored
    name: A string

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, output_depth]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionaly returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_mpnn_attention",
      values=[node_states]):
    q, k, v = compute_mpnn_qkv(node_states,
                               total_key_depth,
                               total_value_depth,
                               num_edge_types,
                               ignore_zero=ignore_zero)
    # reshaping k and v for head splitting
    q_shape = tf.shape(q)
    q = common_attention.split_heads(q, num_heads)
    k = common_attention.split_heads(k, num_heads)
    v = common_attention.split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    # make the heads dimension leading. We will loop over heads.
    q = tf.transpose(q, [1, 0, 2, 3])
    k = tf.transpose(k, [1, 0, 2, 3])
    v = tf.transpose(v, [1, 0, 2, 3])
    # putting edge as the dimension after batch for k and v
    # k and v will be [heads, batch, num_edge_types, length, depth]
    k = tf.reshape(k, [num_heads, q_shape[0], q_shape[1], num_edge_types,
                       total_key_depth//num_heads])
    k = tf.transpose(k, [0, 1, 3, 2, 4])

    v = tf.reshape(v, [num_heads, q_shape[0], q_shape[1], num_edge_types,
                       total_value_depth//num_heads])
    v = tf.transpose(v, [0, 1, 3, 2, 4])

    # doing attention separately for each head
    head_outputs = []
    for head_id in range(num_heads):
      output = dot_product_mpnn_attention(q[head_id],
                                          k[head_id],
                                          v[head_id],
                                          adjacency_matrix,
                                          num_edge_types)
      head_outputs.append(tf.expand_dims(output, axis=0))
    # making x = [heads, batch, length, total_value_depth//num_heads]
    x = tf.concat(head_outputs, axis=0)
    x = tf.transpose(x, [1, 0, 2, 3])
    # making x [batch, length, depth]
    x = common_attention.combine_heads(x)
    x = common_layers.dense(
        x, output_depth, use_bias=False, name="output_transform")
    return x


def dot_product_mpnn_attention(q, k, v, adjacency_matrix, num_edge_types,
                               ignore_zero=True, name=None):
  """Dot product attention with edge vectors.

  Args:
    q: [batch, length, key_depth] tensor
    k: [batch, num_edge_types, length, key_depth]
    v: [batch, num_edge_types, length, depth]
    adjacency_matrix: [batch, length, length] tensor of int edge types
    num_edge_types: an int, specifying number of edge types
    ignore_zero: A flag that says that edge type 0 should be ignored
    name: optional string

  Returns:
    A tensor of shape [batch, length, depth(q)]
  """
  with tf.variable_scope(
      name, default_name="dot_product_mpnn_attention",
      values=[q, k, v, adjacency_matrix, num_edge_types]):
    # Computing attention mask
    # all edge logits will have shape [batch, edge_types, len, len]
    all_edge_logits = tf.matmul(
        tf.tile(tf.expand_dims(q, axis=1), [1, num_edge_types, 1, 1]),
        k, transpose_b=True)
    # adjacency_matrix_one_hot has shape [batch, len, len, num_edge_types]
    adjacency_matrix_one_hot = tf.one_hot(adjacency_matrix, num_edge_types)
    # making adjacency_matrix_one_hot [batch, edge_types, len, len]
    adjacency_matrix_one_hot = tf.transpose(adjacency_matrix_one_hot,
                                            [0, 3, 1, 2])
    # getting dot products for q_i, k_j, and e_{ij}. This assumes that for
    # edge type 0, the dot products are 0
    all_edge_logits *= adjacency_matrix_one_hot
    # logits will be [batch, length, length] after educing along
    # axis 1 which has dimension num_edge_types.
    logits = tf.reduce_sum(all_edge_logits, axis=1)
    # ignoring edges if needed
    bias = 0
    if ignore_zero:
      bias = tf.to_float(tf.equal(adjacency_matrix, 0)) * -1e9
    logits += bias
    # getting compatibilities
    compatibility = tf.nn.softmax(logits)
    common_attention.attention_image_summary(
        tf.expand_dims(compatibility, axis=1), None)
    # getting edge compatibilities ready to compute values.
    # after tiling, edge_compatibility will be
    # [batch, num_edge_types, length, length]
    edge_compatibility = tf.tile(
        tf.expand_dims(compatibility, axis=1), [1, num_edge_types, 1, 1])
    # computing values
    edge_compatibility *= adjacency_matrix_one_hot
    # all edge values will be [batch, num_edge_types, length, depth]
    # We also assumed that the linear transformations for edge_type 0 will
    # all be zeros. That is [batch, 0] is a length*depth tensor of 0's
    all_edge_values = tf.matmul(edge_compatibility, v)
    # reducing along the num_edge_types dimension
    output = tf.reduce_sum(all_edge_values, axis=1)
    return output

