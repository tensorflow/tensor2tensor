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
from tensor2tensor.utils import expert_utils

import tensorflow as tf


def multihead_graph_attention(query_antecedent,
                              memory_antecedent,
                              bias,
                              total_key_depth,
                              total_value_depth,
                              output_depth,
                              num_heads,
                              dropout_rate,
                              image_shapes=None,
                              attention_type="edge_vector",
                              name="multihead_graph_attention",
                              save_weights_to=None,
                              make_image_summary=True,
                              dropout_broadcast_dims=None,
                              adjacency_matrix=None,
                              num_edge_types=5,
                              vars_3d=False,
                              **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    adjacency_matrix: an optional tensor of shape [batch, len_q, len_q]
      containing edge vectors for attention
    num_edge_types: number of edge types, an int
    vars_3d: use 3-dimensional variables for input/output transformations
    **kwargs (dict): Parameters for the attention function

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hiddem_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
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
  vars_3d_num_heads = num_heads if vars_3d else None
  with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

    q, k, v = common_attention.compute_qkv(
        query_antecedent, memory_antecedent, total_key_depth,
        total_value_depth, vars_3d_num_heads=vars_3d_num_heads)
    q = common_attention.split_heads(q, num_heads)
    k = common_attention.split_heads(k, num_heads)
    v = common_attention.split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack

    elif attention_type == "edge_vector":
      x = graph_attention(q, k, v, bias, dropout_rate, image_shapes,
                          save_weights_to=save_weights_to,
                          make_image_summary=make_image_summary,
                          dropout_broadcast_dims=dropout_broadcast_dims,
                          adjacency_matrix=adjacency_matrix,
                          num_edge_types=num_edge_types)

    x = common_attention.combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = common_layers.dense(
          x, output_depth, use_bias=False, name="output_transform")
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x


@expert_utils.add_name_scope()
def make_edge_vectors(adjacency_matrix, num_edge_types, depth, name=None):
  """Gets edge vectors for the edge types in the adjacency matrix.

  Args:
    adjacency_matrix: A [batch, num_nodes, num_nodes] tensor of ints.
    num_edge_types: Number of different edge types
    depth: Number of channels
    name: a string
  Returns:
    A [batch, num_nodes, num_nodes, depth] vector of tensors
  """
  with tf.variable_scope(name, default_name="edge_vectors"):
    att_adj_vectors_shape = [num_edge_types, depth]
    adjacency_matrix_shape = common_layers.shape_list(adjacency_matrix)
    adj_vectors = (
        tf.get_variable(
            "adj_vectors",
            att_adj_vectors_shape,
            initializer=tf.random_normal_initializer(0, depth**-0.5)) *
        (depth**0.5))
    # Avoiding gathers so that it works on TPUs
    # adjacency_matrix_one_hot has shape
    # [batch, num_nodes, num_nodes, num_edge_types]

    adjacency_matrix_one_hot = tf.one_hot(adjacency_matrix, num_edge_types)

    att_adj_vectors = tf.matmul(
        tf.reshape(tf.to_float(adjacency_matrix_one_hot), [-1, num_edge_types]),
        adj_vectors)
    return tf.reshape(att_adj_vectors,
                      [adjacency_matrix_shape[0], adjacency_matrix_shape[1],
                       adjacency_matrix_shape[2], depth])


def graph_attention(q,
                    k,
                    v,
                    bias,
                    dropout_rate=0.0,
                    image_shapes=None,
                    name=None,
                    make_image_summary=True,
                    save_weights_to=None,
                    dropout_broadcast_dims=None,
                    adjacency_matrix=None,
                    num_edge_types=5):
  """graph attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    adjacency_matrix: optional matrix of [batch, length, length] ids indicating
      edge type
    num_edge_types: an int indicating number of edge types
  Returns:
    A Tensor of shape [batch, length, depth(q)]
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)
    if adjacency_matrix is not None:
      key_head_depth = common_layers.shape_list(q)[-1]
      adjacency_vectors = make_edge_vectors(
          adjacency_matrix, num_edge_types, key_head_depth, name)
      # zeroing out the vectors that have 0 entries in the adjacency
      adjacency_vectors *= tf.to_float(
          tf.expand_dims(adjacency_matrix, axis=-1))
      # transposing q to be [batch, length_q, heads, depth_k]
      # to allow for matmul with [batch, length_q, length_q, depth_k]
      q_t = tf.transpose(q, [0, 2, 1, 3])
      adj_logits = tf.matmul(q_t, adjacency_vectors, transpose_b=True)
      logits += tf.transpose(adj_logits, [0, 2, 1, 3])
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
    # dropping out the attention links for each of the heads
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      common_attention.attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)


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
    # logits will be [batch, length, length] after reducing along
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

