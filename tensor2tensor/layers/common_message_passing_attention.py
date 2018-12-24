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

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, output_depth]

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
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):

    q, k, v = common_attention.compute_qkv(
        query_antecedent,
        memory_antecedent,
        total_key_depth,
        total_value_depth,
        vars_3d_num_heads=vars_3d_num_heads)
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
      x = graph_attention(
          q,
          k,
          v,
          bias,
          dropout_rate,
          image_shapes,
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
def make_edge_vectors(adjacency_matrix,
                      num_edge_types,
                      depth,
                      name=None):
  """Gets edge vectors for the edge types in the adjacency matrix.

  Args:
    adjacency_matrix: A [batch, num_nodes, num_nodes, num_edge_types] tensor.
    num_edge_types: Number of different edge types
    depth: Number of channels
    name: A optional string name for scoping
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

    att_adj_vectors = tf.matmul(
        tf.reshape(tf.to_float(adjacency_matrix), [-1, num_edge_types]),
        adj_vectors)
    # Reshape to be [batch, num_nodes, num_nodes, depth].
    att_adj_vectors = tf.reshape(att_adj_vectors, [
        adjacency_matrix_shape[0], adjacency_matrix_shape[1],
        adjacency_matrix_shape[2], depth
    ])
    return att_adj_vectors


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
          adjacency_matrix,
          num_edge_types,
          key_head_depth,
          name=name)
      # transposing q to be [batch, length_q, heads, depth_k]
      # to allow for matmul with [batch, length_q, length_q, depth_k]
      q_t = tf.transpose(q, [0, 2, 1, 3])
      adj_logits = tf.matmul(q_t, adjacency_vectors, transpose_b=True)
      logits += tf.transpose(adj_logits, [0, 2, 1, 3])
      # [batch, depth, num_nodes, num_nodes]
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


def _compute_edge_transforms(node_states,
                             depth,
                             num_transforms,
                             name="transform"):
  """Helper function that computes transformation for keys and values.

  Let B be the number of batches.
  Let N be the number of nodes in the graph.
  Let D be the size of the node hidden states.
  Let K be the size of the attention keys/queries (total_key_depth).
  Let V be the size of the attention values (total_value_depth).
  Let T be the total number of transforms (num_transforms).

  Computes the transforms for keys or values for attention.
  * For each node N_j and edge type t, a key K_jt of size K is computed. When an
    edge of type t goes from node N_j to any other node, K_jt is the key that is
    in the attention process.
  * For each node N_j and edge type t, a value V_jt of size V is computed. When
    an edge of type t goes from node N_j to node N_i, Attention(Q_i, K_jt)
    produces a weight w_ijt. The message sent along this edge is w_ijt * V_jt.

  Args:
    node_states: A tensor of shape [B, L, D]
    depth: An integer (K or V)
    num_transforms: An integer (T),
    name: A name for the function

  Returns:
    x: A The attention keys or values for each node and edge type
      (shape [B, N*T, K or V])
  """
  node_shapes = common_layers.shape_list(node_states)
  x = common_layers.dense(
      node_states,
      depth * num_transforms,
      use_bias=False,
      name=name)

  batch = node_shapes[0]  # B.
  length = node_shapes[1]  # N.

  # Making the fourth dimension explicit by separating the vectors of size
  # K*T (in k) and V*T (in v) into two-dimensional matrices with shape [K, T]
  # (in k) and [V, T] in v.
  #
  x = tf.reshape(x, [batch, length, num_transforms, depth])

  # Flatten out the fourth dimension.
  x = tf.reshape(x, [batch, length * num_transforms, depth])

  return x


def compute_mpnn_qkv(node_states,
                     total_key_depth,
                     total_value_depth,
                     num_transforms):
  """Computes query, key and value for edge matrices.

  Let B be the number of batches.
  Let N be the number of nodes in the graph.
  Let D be the size of the node hidden states.
  Let K be the size of the attention keys/queries (total_key_depth).
  Let V be the size of the attention values (total_value_depth).
  Let T be the total number of transforms (num_transforms).

  Computes the queries, keys, and values for attention.
  * For each node N_i in the graph, a query Q_i of size K is computed. This
    query is used to determine the relative weights to give to each of the
    node's incoming edges.
  * For each node N_j and edge type t, a key K_jt of size K is computed. When an
    edge of type t goes from node N_j to any other node, K_jt is the key that is
    in the attention process.
  * For each node N_j and edge type t, a value V_jt of size V is computed. When
    an edge of type t goes from node N_j to node N_i, Attention(Q_i, K_jt)
    produces a weight w_ijt. The message sent along this edge is w_ijt * V_jt.

  Args:
    node_states: A Tensor with shape [B, N, D].
    total_key_depth: an integer (K).
    total_value_depth: an integer (V).
    num_transforms: a integer specifying number of transforms (T). This is
      typically the number of edge types.
  Returns:
    q: The attention queries for each destination node (shape [B, N, K]).
    k: The attention keys for each node and edge type (shape [B, N*T, K]).
    v: The attention values for each node and edge type (shape [B, N*T, V]).
  """

  # node_states is initially a tensor with shape [B, N, D]. The call to dense
  # creates a D x K kernel that serves as a fully-connected layer.
  #
  # For each possible batch b and node n in the first two dimensions of
  # node_states, the corresponding size-D vector (the third dimension of
  # node_states) is the hidden state for node n in batch b. Each of these size-D
  # vectors is multiplied by the kernel to produce an attention query of size K.
  # The result is a tensor of size [B, N, K] containing the attention queries
  # for each node in each batch.
  q = common_layers.dense(
      node_states, total_key_depth, use_bias=False, name="q_mpnn")

  # Creates the attention keys in a manner similar to the process of creating
  # the attention queries. One key is created for each type of outgoing edge the
  # corresponding node might have, meaning k will have shape [B, N, K*T].
  k = _compute_edge_transforms(node_states,
                               total_key_depth,
                               num_transforms,
                               name="k_mpnn")
  v = _compute_edge_transforms(node_states,
                               total_value_depth,
                               num_transforms,
                               name="v_mpnn")

  return q, k, v


def sparse_message_pass_batched(node_states,
                                adjacency_matrices,
                                num_edge_types,
                                hidden_size,
                                use_bias=True,
                                average_aggregation=False,
                                name="sparse_ggnn_batched"):
  """Identical to sparse_ggnn except that each input has a batch dimension.

  B = The batch size.
  N = The number of nodes in each batch.
  H = The size of the hidden states.
  T = The number of edge types.

  Args:
    node_states: Initial states of each node in the graph. Shape: [B, N, H]
    adjacency_matrices: Adjacency matrices of directed edges for each edge
      type and batch. Shape: [B, N, N, T] (sparse).
    num_edge_types: The number of edge types. T.
    hidden_size: The size of the hidden layer. H.
    use_bias: Whether to use bias in the hidden layer.
    average_aggregation: How to aggregate the incoming node messages. If
      average_aggregation is true, the messages are averaged. If it is false,
      they are summed.
    name: (optional) The scope within which tf variables should be created.

  Returns:
    The result of one round of message-passing of shape [B, N, H].
  """

  b, n = tf.shape(node_states)[0], tf.shape(node_states)[1]

  # Flatten the batch dimension of the node states.
  node_states = tf.reshape(node_states, [b*n, hidden_size])

  # Flatten the batch dimension of the adjacency matrices.
  indices = adjacency_matrices.indices
  new_index2 = indices[:, 3]  # The edge type dimension.

  # Offset N x N adjacency matrix by the batch number in which it appears.
  new_index0 = indices[:, 1] + indices[:, 0] * tf.cast(n, tf.int64)
  new_index1 = indices[:, 2] + indices[:, 0] * tf.cast(n, tf.int64)

  # Combine these indices as triples.
  new_indices = tf.stack([new_index0, new_index1, new_index2], axis=1)

  # Build the new sparse matrix.
  new_shape = [tf.cast(b*n, tf.int64), tf.cast(b*n, tf.int64), num_edge_types]
  adjacency_matrices = tf.SparseTensor(indices=new_indices,
                                       values=adjacency_matrices.values,
                                       dense_shape=new_shape)

  # Run a message-passing step and return the result with the batch dimension.
  node_states = sparse_message_pass(
      node_states,
      adjacency_matrices,
      num_edge_types,
      hidden_size,
      use_bias=use_bias,
      average_aggregation=average_aggregation,
      name=name)
  return tf.reshape(node_states, [b, n, hidden_size])


def sparse_message_pass(node_states,
                        adjacency_matrices,
                        num_edge_types,
                        hidden_size,
                        use_bias=True,
                        average_aggregation=False,
                        name="sparse_ggnn"):
  """One message-passing step for a GNN with a sparse adjacency matrix.

  Implements equation 2 (the message passing step) in
  [Li et al. 2015](https://arxiv.org/abs/1511.05493).

  N = The number of nodes in each batch.
  H = The size of the hidden states.
  T = The number of edge types.

  Args:
    node_states: Initial states of each node in the graph. Shape is [N, H].
    adjacency_matrices: Adjacency matrix of directed edges for each edge
      type. Shape is [N, N, T] (sparse tensor).
    num_edge_types: The number of edge types. T.
    hidden_size: The size of the hidden state. H.
    use_bias: Whether to use bias in the hidden layer.
    average_aggregation: How to aggregate the incoming node messages. If
      average_aggregation is true, the messages are averaged. If it is false,
      they are summed.
    name: (optional) The scope within which tf variables should be created.

  Returns:
    The result of one step of Gated Graph Neural Network (GGNN) message passing.
    Shape: [N, H]
  """
  n = tf.shape(node_states)[0]
  t = num_edge_types
  incoming_edges_per_type = tf.sparse_reduce_sum(adjacency_matrices, axis=1)

  # Convert the adjacency matrix into shape [T, N, N] - one [N, N] adjacency
  # matrix for each edge type. Since sparse tensor multiplication only supports
  # two-dimensional tensors, we actually convert the adjacency matrix into a
  # [T * N, N] tensor.
  adjacency_matrices = tf.sparse_transpose(adjacency_matrices, [2, 0, 1])
  adjacency_matrices = tf.sparse_reshape(adjacency_matrices, [t * n, n])

  # Multiply the adjacency matrix by the node states, producing a [T * N, H]
  # tensor. For each (edge type, node) pair, this tensor stores the sum of
  # the hidden states of the node's neighbors over incoming edges of that type.
  messages = tf.sparse_tensor_dense_matmul(adjacency_matrices, node_states)

  # Rearrange this tensor to have shape [N, T * H]. The incoming states of each
  # nodes neighbors are summed by edge type and then concatenated together into
  # a single T * H vector.
  messages = tf.reshape(messages, [t, n, hidden_size])
  messages = tf.transpose(messages, [1, 0, 2])
  messages = tf.reshape(messages, [n, t * hidden_size])

  # Run each of those T * H vectors through a linear layer that produces
  # a vector of size H. This process is equivalent to running each H-sized
  # vector through a separate linear layer for each edge type and then adding
  # the results together.
  #
  # Note that, earlier on, we added together all of the states of neighbors
  # that were connected by edges of the same edge type. Since addition and
  # multiplying by a linear layer are commutative, this process was equivalent
  # to running each incoming edge through a linear layer separately and then
  # adding everything at the end.
  with tf.variable_scope(name, default_name="sparse_ggnn"):
    final_node_states = common_layers.dense(
        messages, hidden_size, use_bias=False)

    # Multiply the bias by for each edge type by the number of incoming nodes
    # of that edge type.
    if use_bias:
      bias = tf.get_variable("bias", initializer=tf.zeros([t, hidden_size]))
      final_node_states += tf.matmul(incoming_edges_per_type, bias)

    if average_aggregation:
      incoming_edges = tf.reduce_sum(incoming_edges_per_type, -1, keepdims=True)
      incoming_edges = tf.tile(incoming_edges, [1, hidden_size])
      final_node_states /= incoming_edges + 1e-7

  return final_node_states


def multihead_mpnn_attention(node_states,
                             total_key_depth,
                             total_value_depth,
                             output_depth,
                             num_heads,
                             adjacency_matrix=None,
                             num_edge_types=5,
                             num_transforms=None,
                             use_weighted_sum=False,
                             name="mpnn_attention"):
  """Multihead scaled-dot-product attention with input/output transformations.

  Let B be the number of batches.
  Let N be the number of nodes in the graph.
  Let D be the size of the node hidden states.
  Let K be the size of the attention keys/queries (total_key_depth).
  Let V be the size of the attention values (total_value_depth).
  Let O be the size of the attention output (output_depth).
  Let H be the number of heads (num_heads).
  Let T be the total number of transforms (num_transforms).

  The key and value depths are split across all of the heads. For example, if
  the key depth is 6 and there are three heads, then the key for each head has
  depth 2.

  Args:
    node_states: A Tensor with shape [B, N, D]
    total_key_depth: An integer (K).
    total_value_depth: An integer (V).
    output_depth: An integer (O).
    num_heads: An integer (H).
    adjacency_matrix: An Tensor of ints with shape [B, T, N, N]. If there is an
      edge from node j to node i in batch b, then adjacency_matrix[b, i, j]
      contains the type of that edge as an integer. Otherwise, it contains 0.
    num_edge_types: An integer indicating number of edge types.
    num_transforms: An integer indicating number of transforms (T). If None,
      then num_transforms will be equal to num_edge_types.
    use_weighted_sum: If False, will only use a single transform per edge type.
      Otherwise, use a learned weighted sum of transforms per edge type.
    name: A string.

  Returns:
    The result of the attention transformation. The output shape is [B, N, O].

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
      name, default_name="multihead_mpnn_attention", values=[node_states]):
    # If not explicitly set, use num_transforms set to num_edge_types.
    num_transforms = (
        num_edge_types if num_transforms is None else num_transforms)

    # Create the query for each node's incoming edges.
    # Create the keys/values for each node for each possible outgoing edge type.
    q, k, v = compute_mpnn_qkv(
        node_states,
        total_key_depth,
        total_value_depth,
        num_transforms)

    q_shape = tf.shape(q)  # As above, q_shape is [B, N, K].

    # Divides each query/key/value into separate heads. Specifically, the
    # query/key/value for each (batch, node) pair (i.e., the third dimensions
    # of q, k, and v) are broken into H separate pieces. These pieces are used
    # as the separate attention heads. The resulting tensors have shape
    # [B, H, N, ?/H], where ? = K, K*T or V*T as appropriate.
    q = common_attention.split_heads(q, num_heads)  # Shape [B, H, N, K/H].
    k = common_attention.split_heads(k, num_heads)  # Shape [B, H, N, K*T/H].
    v = common_attention.split_heads(v, num_heads)  # Shape [B, H, N, V*T/H].
    key_depth_per_head = total_key_depth // num_heads

    # Ensures that the logits don't have too large of a magnitude.
    q *= key_depth_per_head**-0.5

    # Rearrange the dimensions so that the head is first. This will make
    # subsequent steps easier (we loop over the head).
    q = tf.transpose(q, [1, 0, 2, 3])  # Shape [H, B, N, K/H].
    k = tf.transpose(k, [1, 0, 2, 3])  # Shape [H, B, N, K*T/H].
    v = tf.transpose(v, [1, 0, 2, 3])  # Shape [H, B, N, V*T/H].

    # Split the keys and values into separate per-edge-type keys and values.
    k = tf.reshape(k, [
        num_heads, q_shape[0], q_shape[1], num_transforms,
        total_key_depth // num_heads
    ])  # Shape [H, B, N, T, K/H].
    k = tf.transpose(k, [0, 1, 3, 2, 4])  # Shape [H, B, T, N, K/H].

    v = tf.reshape(v, [
        num_heads, q_shape[0], q_shape[1], num_transforms,
        total_value_depth // num_heads
    ])  # Shape [H, B, N, T, V/H].
    v = tf.transpose(v, [0, 1, 3, 2, 4])  # Shape [H, B, T, N, V/H].

    # Perform attention for each head and combine the results into a list.
    # head_outputs stores a list of tensors, each with shape [1, B, N, V/H].
    # The last dimension contains the values computed for each attention head.
    # Each value was determined by computing attention over all of the
    # incoming edges for node n, weighting the incoming values accordingly,
    # and adding those weighted values together.
    head_outputs = []
    for head_id in range(num_heads):
      output = dot_product_mpnn_attention(
          q[head_id],
          k[head_id],
          v[head_id],
          adjacency_matrix,
          num_edge_types,
          num_transforms=num_transforms,
          use_weighted_sum=use_weighted_sum)

      # Store this result in the list of attention results for each head.
      # The call to expand_dims gives output shape [1, B, N, V/H], which will
      # come in handy when we combine the heads together.
      head_outputs.append(tf.expand_dims(output, axis=0))

    # Combine the heads together into one tensor and rearrange the dimensions.
    x = tf.concat(head_outputs, axis=0)  # Shape [H, B, N, V/H].
    x = tf.transpose(x, [1, 0, 2, 3])  # Shape [B, H, N, V/H].

    # Concatenate the values produced by each head together into one vector.
    x = common_attention.combine_heads(x)  # Shape [B, N, V].

    # A fully-connected linear layer to convert from the value vectors of size V
    # to output vectors of length O (the appropriate output length).
    x = common_layers.dense(
        x, output_depth, use_bias=False, name="output_transform")
    return x


def dot_product_mpnn_attention(q,
                               k,
                               v,
                               adjacency_matrix,
                               num_edge_types,
                               num_transforms=None,
                               use_weighted_sum=False,
                               name=None):
  """Dot product attention with edge vectors.

  Let B be the number of batches.
  Let N be the number of nodes in the graph.
  Let K be the size of the attention keys/queries.
  Let V be the size of the attention values.
  Let T be the total number of transforms (num_transforms).

  Args:
    q: The query Tensor of shape [B, N, K].
    k: The key Tensor of shape [B, T, N, K].
    v: The value Tensor of shape [B, T, N, V].
    adjacency_matrix: A Tensor of shape [B, N, N, T]. An entry at
      indices b, i, j, k is the indicator of the edge
      from node j to node i in batch b. A standard adjacency matrix will only
      have one edge type while a mutigraph will have multiple edge types.
    num_edge_types: An integer specifying number of edge types.
    num_transforms: An integer indicating number of transforms (T). If None,
      then num_transforms will be equal to num_edge_types.
    use_weighted_sum: If False, will only use a single transform per edge type.
      Otherwise, use a learned weighted sum of transforms per edge type.
    name: A string.

  Returns:
    A Tensor of shape [B, N, V] storing the result of computing attention
    weights using the queries and keys and combining the values according to
    those weights.

  Raises:
    ValueError: if num_transforms doesn't equal num_edge_types and not using
      weighted sum.
  """
  with tf.variable_scope(
      name,
      default_name="dot_product_mpnn_attention",
      values=[q, k, v, adjacency_matrix, num_edge_types]):
    # If not explicitly set, use num_transforms set to num_edge_types.
    num_transforms = (
        num_edge_types if num_transforms is None else num_transforms)

    if not use_weighted_sum and num_transforms != num_edge_types:
      raise ValueError("num_transforms must equal num_edge_types unless "
                       "use_weighted_sum is True")

    # Computes the raw dot-product attention values between each query and
    # the corresponding keys it needs to consider.
    #
    # This operation takes the dot product of (the query for
    # each node) and (the key for each node for each possible edge type),
    # creating an N x N matrix for each edge type. The entry at index (i, j)
    # is the dot-product for the edge from node i to node j of the appropriate
    # type. These dot products will eventually become attention weights
    # specifying how much node i weights an edge of that type coming from node
    # j.
    all_edge_logits = tf.matmul(
        tf.tile(tf.expand_dims(q, axis=1), [1, num_edge_types, 1, 1]),
        k,
        transpose_b=True)

    # The adjacency matrix assumes there is only one directed edge (i <- j) for
    # each pair of nodes. If such an edge exists, it contains the integer
    # type of that edge at position (i, j) of the adjacency matrix.
    #
    # Construct edge_vectors of shape [B, N, N, T].
    if use_weighted_sum:
      # Use dense representation for edge vectors.
      edge_vectors = make_edge_vectors(
          adjacency_matrix,
          num_edge_types,
          num_transforms)
    else:
      # Generate one-hot vectors based on edge types.
      # If there is an edge from node j to node i of type t, then index t of the
      # last dimension is 1 for entry (i, j) of the second and third dimensions.
      edge_vectors = tf.one_hot(adjacency_matrix, num_transforms)

    # Rearranging the dimensions to match the shape of all_edge_logits.
    edge_vectors = tf.transpose(edge_vectors, [0, 3, 1, 2])

    # Element-wise multiplies all_edge_logits and edge_vectors.
    #
    # In other words: all_edge_logits contains N x N matrices of query-key
    # products. This element-wise multiplication zeroes out entries that do not
    # correspond to actual edges in the graph of the appropriate edge type.
    # all_edge_logits retains shape [B, T, N, N].
    all_edge_logits *= edge_vectors

    # Since there can only be one edge from node A to node B, we can collapse
    # the T different adjacency matrices containing key-query pairs into one
    # adjacency matrix. logits is [B, N, N].
    # TODO(dbieber): Use a reshape instead of reduce sum to attend over all
    # edges instead of over all neighboring nodes to handle the multigraph case.
    logits = tf.reduce_sum(all_edge_logits, axis=1)

    # For pairs of nodes with no edges between them, add a large negative bias
    # to each location without an edge so that the softmax of entries with the
    # value 0 become a small negative number instead.
    bias = 0
    bias = tf.to_float(tf.equal(
        tf.reduce_sum(adjacency_matrix, axis=-1), 0)) * -1e9
    logits += bias

    # Turn the raw key-query products into a probability distribution (or,
    # in terms of attention, weights). The softmax is computed across the
    # last dimension of logits.
    compatibility = tf.nn.softmax(logits)  # Shape [B, N, N].

    # Computes a summary showing the attention matrix as an image. Does not do
    # any work toward actually performing attention.
    common_attention.attention_image_summary(
        tf.expand_dims(compatibility, axis=1), None)

    # Repeats the attention matrix T times for each batch, producing
    # a tensor with shape [B, T, N, N] where the [N, N] component is T
    # repeats of the values found in compatibility.
    edge_compatibility = tf.tile(
        tf.expand_dims(compatibility, axis=1), [1, num_edge_types, 1, 1])

    # Zeroes out the entries in edge_compatibility that do not correspond to
    # actual edges.
    edge_compatibility *= edge_vectors  # Shape [B, T, N, N].

    output = compute_values(edge_compatibility, v)
    return output


def ggnn_fast_dense(node_states,
                    adjacency_matrix,
                    num_edge_types,
                    total_value_depth,
                    name=None):
  """ggnn version of the MPNN from Gilmer et al.

  Let B be the number of batches.
  Let D be the size of the node hidden states.
  Let K be the size of the attention keys/queries.
  Let V be the size of the output of the ggnn.
  Let T be the number of transforms / edge types.

  Args:
    node_states: The value Tensor of shape [B, T, N, D].
    adjacency_matrix: A Tensor of shape [B, N, N, T]. An entry at
      indices b, i, j, k is the indicator of the edge from node j to node i in
      batch b. A standard adjacency matrix will only have values of one, while a
      mutigraph may have larger integer values.
    num_edge_types: An integer specifying number of edge types.
    total_value_depth: An integer (V)
    name: A string.

  Returns:
    A Tensor of shape [B, N, V] storing the result of computing attention
    weights using the queries and keys and combining the values according to
    those weights.

  Raises:
    ValueError: if num_transforms doesn't equal num_edge_types and not using
      weighted sum.
  """
  # between the same nodes (with only one edge of each type. adjacency_matrix
  # will need to be converted to shape [B, T, N, N].
  with tf.variable_scope(
      name,
      default_name="ggnn_fast_dense",
      values=[node_states, adjacency_matrix, num_edge_types]):
    nodes_shape = common_layers.shape_list(node_states)
    v = _compute_edge_transforms(node_states,
                                 total_value_depth,
                                 num_edge_types,
                                 name="v_mpnn")
    v = tf.reshape(v, [nodes_shape[0], nodes_shape[1], num_edge_types,
                       total_value_depth
                      ])  # Shape [B, N, T, V].
    v = tf.transpose(v, [0, 2, 1, 3])  # Shape [B, T, N, V].

    # Rearranging the dimensions to match the shape of all_edge_logits.
    edge_vectors = tf.transpose(adjacency_matrix, [0, 3, 1, 2])
    output = compute_values(edge_vectors, v)
    return output


def compute_values(edge_compatibility, v):
  """Compute values. If edge compatibilities is just adjacency, we get ggnn.

  Args:
    edge_compatibility: A tensor of shape [batch, num_transforms, length, depth]
    v: A tensor of shape [batch, num_transforms, length, depth]

  Returns:
    output: A [batch, length, depth] tensor
  """

  # Computes the incoming value vectors for each node by weighting them
  # according to the attention weights. These values are still segregated by
  # edge type.
  # Shape = [B, T, N, V].
  all_edge_values = tf.matmul(tf.to_float(edge_compatibility), v)

  # Combines the weighted value vectors together across edge types into a
  # single N x V matrix for each batch.
  output = tf.reduce_sum(all_edge_values, axis=1)  # Shape [B, N, V].
  return output


def precompute_edge_matrices(adjacency, hparams):
  """Precompute the a_in and a_out tensors.

  (we don't want to add to the graph everytime _fprop is called)
  Args:
    adjacency: placeholder of real valued vectors of shape [B, L, L, E]
    hparams: tf.HParams object
  Returns:
    edge_matrices: [batch, L * D, L * D] the dense matrix for message passing
    viewed as a block matrix (L,L) blocks of size (D,D). Each plot is a function
    of the edge vector of the adjacency matrix at that spot.
  """
  batch_size, num_nodes, _, edge_dim = common_layers.shape_list(adjacency)

  # build the edge_network for incoming edges
  with tf.variable_scope("edge_network"):
    x = tf.reshape(
        adjacency, [batch_size * num_nodes * num_nodes, edge_dim],
        name="adj_reshape_in")

    for ip_layer in range(hparams.edge_network_layers):
      name = "edge_network_layer_%d"%ip_layer
      x = tf.layers.dense(common_layers.layer_preprocess(x, hparams),
                          hparams.edge_network_hidden_size,
                          activation=tf.nn.relu,
                          name=name)
    x = tf.layers.dense(common_layers.layer_preprocess(x, hparams),
                        hparams.hidden_size**2,
                        activation=None,
                        name="edge_network_output")

  # x = [batch * l * l, d *d]
  edge_matrices_flat = tf.reshape(x, [batch_size, num_nodes,
                                      num_nodes, hparams.hidden_size,
                                      hparams.hidden_size])

  # reshape to [batch, l * d, l *d]
  edge_matrices = tf.reshape(
      tf.transpose(edge_matrices_flat, [0, 1, 3, 2, 4]), [
          -1, num_nodes * hparams.hidden_size,
          num_nodes * hparams.hidden_size
      ],
      name="edge_matrices")

  return edge_matrices


def dense_message_pass(node_states, edge_matrices):
  """Computes a_t from h_{t-1}, see bottom of page 3 in the paper.

  Args:
    node_states: [B, L, D] tensor (h_{t-1})
    edge_matrices (tf.float32): [B, L*D, L*D]

  Returns:
    messages (tf.float32): [B, L, D] For each pair
      of nodes in the graph a message is sent along both the incoming and
      outgoing edge.
  """
  batch_size, num_nodes, node_dim = common_layers.shape_list(node_states)

  # Stack the nodes as a big column vector.
  h_flat = tf.reshape(
      node_states, [batch_size, num_nodes * node_dim, 1], name="h_flat")

  messages = tf.reshape(
      tf.matmul(edge_matrices, h_flat), [batch_size * num_nodes, node_dim],
      name="messages_matmul")

  message_bias = tf.get_variable("message_bias", shape=node_dim)
  messages = messages + message_bias
  messages = tf.reshape(messages, [batch_size, num_nodes, node_dim])
  return messages
