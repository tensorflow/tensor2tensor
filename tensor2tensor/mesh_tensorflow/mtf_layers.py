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
"""Layers for mesh tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
import tensorflow as tf


def dense(x, output_dim, reduced_dims=None, expert_dims=None,
          use_bias=True, activation=None, name=None):
  """Dense layer doing (kernel*x + bias) computation.

  Args:
    x: a mtf.Tensor of shape [..., reduced_dims].
    output_dim: a mtf.Dimension
    reduced_dims: an optional list of mtf.Dimensions of x to be reduced. If
      omitted, we reduce the last dimension.
    expert_dims: an optional list of mtf.Dimension which represent different
      experts. Different experts get different weights.
    use_bias: a boolean, whether to add bias.
    activation: an optional function from mtf.Tensor to mtf.Tensor
    name: a string. variable scope.

  Returns:
    a mtf.Tensor of shape [..., output_dim].
  """
  if expert_dims is None:
    expert_dims = []
  if reduced_dims is None:
    reduced_dims = x.shape.dims[-1:]
  w_shape = mtf.Shape(expert_dims + reduced_dims + [output_dim])
  output_shape = mtf.Shape(
      [d for d in x.shape.dims if d not in reduced_dims] + [output_dim])
  with tf.variable_scope(name, default_name="dense"):
    stddev = mtf.list_product(d.size for d in reduced_dims) ** -0.5
    w = mtf.get_variable(
        x.mesh,
        "kernel",
        w_shape,
        initializer=tf.random_normal_initializer(stddev=stddev),
        activation_dtype=x.dtype)
    y = mtf.matmul(x, w, output_shape=output_shape)
    if use_bias:
      b = mtf.get_variable(
          x.mesh,
          "bias",
          mtf.Shape(expert_dims + [output_dim]),
          initializer=tf.zeros_initializer(),
          activation_dtype=x.dtype)
      y += b
    if activation is not None:
      y = activation(y)
    return y


def layer_norm(x, dim, epsilon=1e-6, name="layer_prepostprocess"):
  """Layer normalization over dimension dim.

  Args:
    x: a mtf.Tensor whose shape contains dim.
    dim: a mtf.Dimension
    epsilon: a floating point number
    name: a string. variable scope.

  Returns:
    a mtf.Tensor with same shape as x.
  """
  with tf.variable_scope(name + "/layer_norm"):
    scale = mtf.get_variable(
        x.mesh,
        "layer_norm_scale",
        mtf.Shape([dim]),
        initializer=tf.ones_initializer(),
        activation_dtype=x.dtype)
    bias = mtf.get_variable(
        x.mesh,
        "layer_norm_bias",
        mtf.Shape([dim]),
        initializer=tf.zeros_initializer(),
        activation_dtype=x.dtype)
    reduced_shape = x.shape - dim
    mean = mtf.reduce_mean(x, output_shape=reduced_shape)
    variance = mtf.reduce_mean(mtf.square(x - mean), output_shape=reduced_shape)
    norm_x = (x - mean) * mtf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def softmax_cross_entropy_with_logits(logits, targets, vocab_dim):
  """Per-example softmax loss.

  Args:
    logits: a mtf.Tensor whose shape contains vocab_dim
    targets: a mtf.Tensor with the same shape as logits
    vocab_dim: a mtf.Dimension

  Returns:
    a mtf.Tensor whose shape is equal to logits.shape - vocab_dim

  Raises:
    ValueError: if the shapes do not match.
  """
  if logits.shape != targets.shape:
    raise ValueError(
        "logits shape must equal targets shape"
        "logits=%s targets=%s" % (logits.to_string, targets.to_string))
  if vocab_dim not in logits.shape.dims:
    raise ValueError("vocab_dim must be in logits.shape.dims")
  log_softmax = mtf.log_softmax(logits, vocab_dim)
  return mtf.negative(
      mtf.reduce_sum(log_softmax * targets, reduced_dim=vocab_dim))


def weights_nonzero(targets, dtype=tf.float32):
  def my_fn(x):
    return tf.cast(tf.not_equal(x, 0), dtype)
  return mtf.cwise(my_fn, [targets], output_dtype=dtype, name="weights_nonzero")


def dense_relu_dense(x,
                     hidden_channels,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     name=None):
  """Hidden layer with ReLU activation followed by linear projection.

  The output has the same number of channels as the input.

  Args:
    x: a mtf.Tensor
    hidden_channels: a mtf.Dimension - channels in the hidden layer
    dropout: an optional float
    dropout_broadcast_dims: an optional list of mtf.Dimension
    name: an optional string

  Returns:
    a mtf.Tensor with the same shape as x.
  """
  with tf.variable_scope(name, default_name="dense_relu_dense"):
    io_channels = x.shape.dims[-1]
    stddev = (hidden_channels.size * io_channels.size) ** -0.25
    io = mtf.Dimension("io", 2)
    w = mtf.get_variable(
        x.mesh,
        "kernel",
        mtf.Shape([io, io_channels, hidden_channels]),
        initializer=tf.random_normal_initializer(stddev=stddev),
        activation_dtype=x.dtype)
    wi, wo = mtf.unstack(w, io)
    h = mtf.relu(mtf.einsum([x, wi]))
    if dropout != 0.0:
      h = mtf.dropout(h, 1.0 - dropout,
                      noise_shape=h.shape - dropout_broadcast_dims)
    return mtf.einsum([h, wo])


def masked_local_attention_1d(query_antecedent,
                              memory_antecedent,
                              kv_channels,
                              heads,
                              block_length=128,
                              name=None):
  """Attention to the source position and a neighborhood to the left of it.

  The sequence is divided into blocks of length block_size.
  Attention for a given query position can only see memory positions
  less than or equal to the query position, in the corresponding block
  and the previous block.

  Args:
    query_antecedent: a mtf.Tensor with shape [batch, query_length, io_channels]
    memory_antecedent: a mtf.Tensor with shape
      [batch, memory_length, io_channels] (optional). Currently, memory_length
      must have the same size as query_length, but a different name.
    kv_channels: a mtf.Dimension (the size of the key and value vectors)
    heads: a mtf.Dimension (the number of heads)
    block_length: an integer, representing receptive fields for attention.
    name: an optional string.

  Returns:
    a Tensor of shape [batch, query_length, io_channels]

  Raises:
    ValueError: if channels or depth don't match.
  """
  with tf.variable_scope(
      name, default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):

    batch, query_length, io_channels = query_antecedent.shape.dims
    q_var, k_var, v_var, o_var = multihead_attention_vars(
        query_antecedent.mesh, heads, io_channels, kv_channels,
        query_antecedent.dtype)

    if memory_antecedent is None:
      memory_antecedent = rename_length_to_memory_length(
          query_antecedent, query_length.name)
    memory_batch, memory_length, memory_channels = memory_antecedent.shape.dims
    if memory_batch != batch:
      raise ValueError("memory batch must equal query batch")
    if memory_channels != io_channels:
      raise ValueError("memory channels must equal query channels")

    # Get query q, keys k and values v.
    q = mtf.einsum(
        [query_antecedent, q_var],
        mtf.Shape([batch, heads, query_length, kv_channels]))
    k = mtf.einsum(
        [memory_antecedent, k_var],
        mtf.Shape([batch, heads, memory_length, kv_channels]))
    v = mtf.einsum(
        [memory_antecedent, v_var],
        mtf.Shape([batch, heads, memory_length, kv_channels]))

    # Let's assume for now we don't have padding and the block length equally
    # divides the memory length.
    block_length = (query_length.size
                    if query_length.size < block_length * 2 else block_length)
    blength = mtf.Dimension("block_length", block_length)
    mlength = mtf.Dimension("mem_block_length", block_length)
    num_blocks = mtf.Dimension("num_blocks", query_length.size // block_length)

    q = mtf.reshape(
        q, mtf.Shape([batch, heads, num_blocks, blength, kv_channels]))
    k = mtf.reshape(
        k, mtf.Shape([batch, heads, num_blocks, mlength, kv_channels]))
    v = mtf.reshape(
        v, mtf.Shape([batch, heads, num_blocks, mlength, kv_channels]))

    # compute attention for the first query block.
    def first_block_attention():
      """Compute attention for the first block."""
      first_q = mtf.slice(q, 0, 1, num_blocks.name)
      first_k = mtf.slice(k, 0, 1, num_blocks.name)
      first_v = mtf.slice(v, 0, 1, num_blocks.name)
      first_output = dot_product_attention(first_q,
                                           first_k,
                                           first_v,
                                           mask=None)
      return first_output

    # Attention for first block, since query_length = key_length.
    first_output = first_block_attention()

    # Concatenate two adjacent blocks to compute the overlapping memory block.
    def local(x):
      """Helper function to get memory blocks."""
      prev_block = mtf.slice(x, 0, num_blocks.size-1, num_blocks.name)
      cur_block = mtf.slice(x, 1, num_blocks.size-1, num_blocks.name)
      local_block = mtf.concat([prev_block, cur_block], mlength.name)
      return local_block

    local_k = local(k)
    local_v = local(v)
    # Calculate the causal mask to avoid peeking into the future. We compute
    # this once and reuse it for all blocks since the block_size is known.
    mlength = local_k.shape.dims[3]
    mask = attention_bias_local_block(query_antecedent.mesh,
                                      blength, mlength)

    # Remove the first block from q since we already computed that.
    tail_q = mtf.slice(q, 1, num_blocks.size-1, num_blocks.name)

    tail_output = dot_product_attention(tail_q,
                                        local_k,
                                        local_v,
                                        mask=mask)

    # Now concatenate the first and rest of the blocks.
    final_output = mtf.concat([first_output, tail_output], num_blocks.name)
    final_output = mtf.reshape(final_output, mtf.Shape(
        [batch, heads, query_length, kv_channels]))
    return mtf.einsum([final_output, o_var],
                      mtf.Shape([batch, query_length, io_channels]))


def rename_length_to_memory_length(
    x, length_name="length", memory_length_name="memory_length"):
  return mtf.rename_dimension(x, length_name, memory_length_name)


def multihead_attention_vars(
    mesh, heads, io_channels, kv_channels, activation_dtype):
  """Create Parameters for Multihead Attention.

  Args:
    mesh: a Mesh
    heads: a Dimension
    io_channels: a Dimension
    kv_channels: a Dimension
    activation_dtype: a tf.dtype

  Returns:
    q_var: a Tensor with shape [heads, io_channels, kv_channels]
    k_var: a Tensor with shape [heads, io_channels, kv_channels]
    v_var: a Tensor with shape [heads, io_channels, kv_channels]
    o_var: a Tensor with shape [heads, io_channels, kv_channels]
  """
  qkvo = mtf.Dimension("qkvo", 4)
  qk_stddev = (io_channels.size ** -0.5) * (kv_channels.size ** -0.25)
  v_stddev = io_channels.size ** -0.5
  o_stddev = (io_channels.size * heads.size) ** -0.5
  def qkvo_initializer(shape,
                       dtype=None,
                       partition_info=None,
                       verify_shape=None):
    del partition_info, verify_shape
    return tf.random_normal(shape, dtype=dtype) * tf.reshape(
        [qk_stddev, qk_stddev, v_stddev, o_stddev], [4, 1, 1, 1])
  var = mtf.get_variable(
      mesh, "qkvo", mtf.Shape([qkvo, heads, io_channels, kv_channels]),
      initializer=qkvo_initializer, activation_dtype=activation_dtype)
  q_var, k_var, v_var, o_var = mtf.unstack(var, qkvo)
  return q_var, k_var, v_var, o_var


def dot_product_attention(q,
                          k,
                          v,
                          mask,
                          dropout=0.0,
                          dropout_broadcast_dims=None):
  """Dot-product attention.

  Args:
    q: Tensor with shape [...., length_q, depth_k]. Typically leading dimensions
      are [batch, heads].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    mask: mask Tensor (see attention_mask())
    dropout: a float.
    dropout_broadcast_dims: an optional list of mtf.Dimension

  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  length_kv = k.shape.dims[-2]
  logits_shape = mtf.Shape(q.shape.dims[:-1] + [length_kv])
  logits = mtf.einsum([q, k], logits_shape)
  if mask is not None:
    logits += mask
  weights = mtf.softmax(logits, length_kv)
  if dropout != 0.0:
    weights = mtf.dropout(
        weights, 1.0 - dropout,
        noise_shape=weights.shape - dropout_broadcast_dims)
  depth_v = v.shape.dims[-1]
  outputs_shape = mtf.Shape(q.shape.dims[:-1] + [depth_v])
  outputs = mtf.einsum([weights, v], outputs_shape)
  return outputs


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        mask,
                        kv_channels,
                        heads,
                        dropout=0.0,
                        dropout_broadcast_dims=None,
                        name="multihead_attention"):
  """Multihead scaled-dot-product attention with input/output transformations.

  In order to use only one variable containing the four weight matrices
  packed together, we insist that the query and memory antecedents have the
  same dimensionality (io_channels) and that the keys and values have the
  same dimensionality (kv_channels).

  Args:
    query_antecedent: a mtf.Tensor with shape [batch, query_length, io_channels]
    memory_antecedent: a mtf.Tensor with shape
      [batch, memory_length, io_channels] (optional)
    mask: mask Tensor (see attention_mask())
    kv_channels: a mtf.Dimension (the size of the key and value vectors)
    heads: a mtf.Dimension (the number of heads)
    dropout: a floating point value
    dropout_broadcast_dims: an optional list of mtf.Dimension
    name: an optional string.

  Returns:
    A mtf.Tensor with shape [batch, query_length, io_channels]

  Raises:
    ValueError: if the dimensions do not match.
  """
  batch, query_length, io_channels = query_antecedent.shape.dims
  with tf.variable_scope(name,
                         default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):
    q_var, k_var, v_var, o_var = multihead_attention_vars(
        query_antecedent.mesh, heads, io_channels, kv_channels,
        query_antecedent.dtype)
    if memory_antecedent is None:
      memory_antecedent = rename_length_to_memory_length(
          query_antecedent, query_length.name)
    memory_batch, memory_length, memory_channels = memory_antecedent.shape.dims
    if memory_batch != batch:
      raise ValueError("memory batch must equal query batch")
    if memory_channels != io_channels:
      raise ValueError("memory channels must equal query channels")
    q = mtf.einsum(
        [query_antecedent, q_var],
        mtf.Shape([batch, heads, query_length, kv_channels]))
    k = mtf.einsum(
        [memory_antecedent, k_var],
        mtf.Shape([batch, heads, memory_length, kv_channels]))
    v = mtf.einsum(
        [memory_antecedent, v_var],
        mtf.Shape([batch, heads, memory_length, kv_channels]))
    o = dot_product_attention(
        q, k, v, mask, dropout, dropout_broadcast_dims)
    return mtf.einsum(
        [o, o_var], mtf.Shape([batch, query_length, io_channels]))


def multihead_self_attention_incremental(query_antecedent,
                                         prev_k,
                                         prev_v,
                                         step_num,
                                         name="multihead_attention"):
  """Incremental self-attention (one decode step).

  In order to use only one variable containing the four weight matrices
  packed together, we insist that the query and memory antecedents have the
  same dimensionality (io_channels) and that the keys and values have the
  same dimensionality (kv_channels).

  Args:
    query_antecedent: a mtf.Tensor with shape [batch..., io_channels]
    prev_k: mtf.Tensor with shape [batch..., heads, memory_length, kv_channels]
    prev_v: mtf.Tensor with shape [batch..., heads, memory_length, kv_channels]
    step_num: mtf Scalar with dtype tf.int32
    name: an optional string.

  Returns:
    y: A mtf.Tensor with shape [batch..., io_channels]
    new_k: mtf.Tensor with shape [batch..., heads, memory_length, kv_channels]
    new_v: mtf.Tensor with shape [batch..., heads, memory_length, kv_channels]

  Raises:
    ValueError: if the dimensions do not match.
  """
  batch_dims = query_antecedent.shape.dims[:-1]
  io_channels = query_antecedent.shape.dims[-1]
  heads, memory_length, kv_channels = prev_k.shape.dims[-3:]
  with tf.variable_scope(name, default_name="multihead_attention"):
    q_var, k_var, v_var, o_var = multihead_attention_vars(
        query_antecedent.mesh, heads, io_channels, kv_channels,
        query_antecedent.dtype)
    memory_antecedent = query_antecedent
    q = mtf.einsum(
        [query_antecedent, q_var],
        mtf.Shape(batch_dims + [heads, kv_channels]))
    k = mtf.einsum(
        [memory_antecedent, k_var],
        mtf.Shape(batch_dims + [heads, kv_channels]))
    v = mtf.einsum(
        [memory_antecedent, v_var],
        mtf.Shape(batch_dims + [heads, kv_channels]))
    k = prev_k + mtf.multiply(
        k, mtf.one_hot(step_num, memory_length), output_shape=prev_k.shape)
    v = prev_v + mtf.multiply(
        v, mtf.one_hot(step_num, memory_length), output_shape=prev_v.shape)

    mask = mtf.to_float(mtf.greater(mtf.range(
        query_antecedent.mesh, memory_length, dtype=tf.int32), step_num)
                       ) * -1e9
    o = dot_product_attention(q, k, v, mask)
    y = mtf.einsum([o, o_var], query_antecedent.shape)
    return y, k, v


def multihead_encdec_attention_incremental(query_antecedent,
                                           q_var, o_var, k, v,
                                           mask,
                                           name="multihead_attention"):
  """Incremental attention over encoder (one decode step).

  In order to use only one variable containing the four weight matrices
  packed together, we insist that the query and memory antecedents have the
  same dimensionality (io_channels) and that the keys and values have the
  same dimensionality (kv_channels).

  memory_dims is a subset of query_dims

  Args:
    query_antecedent: a mtf.Tensor with shape query_dims + [io_channels]
    q_var: a mtf.Tensor with shape [heads, io_channels, kv_channels]
    o_var: a mtf.Tensor with shape [heads, io_channels, kv_channels]
    k: memory_dims + [heads, memory_length, kv_channels]
    v: memory_dims + [heads, memory_length, kv_channels]
    mask: mask Tensor (see attention_mask())
    name: an optional string.

  Returns:
    A mtf.Tensor with shape [batch, qlen, io_channels]
  """
  heads, _, kv_channels = k.shape.dims[-3:]
  query_dims = query_antecedent.shape.dims[:-1]
  with tf.variable_scope(name, default_name="multihead_attention"):
    q = mtf.einsum(
        [query_antecedent, q_var],
        mtf.Shape(query_dims + [heads, kv_channels]))
    o = dot_product_attention(q, k, v, mask)
    return mtf.einsum([o, o_var], query_antecedent.shape)


def attention_mask_ignore_padding(inputs, dtype=tf.float32):
  """Bias for encoder-decoder attention.

  Args:
    inputs: a mtf.Tensor with shape [..., length_dim]
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [..., memory_length_dim]
  """
  inputs = rename_length_to_memory_length(inputs)
  return mtf.cast(mtf.equal(inputs, 0), dtype) * -1e9


def attention_mask_autoregressive(query_pos, dtype=tf.float32):
  """Bias for self-attention where attention to the right is disallowed.

  Args:
    query_pos: a mtf.Tensor with shape [..., length_dim]
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [..., length_dim, memory_length_dim]
  """
  memory_pos = rename_length_to_memory_length(query_pos)
  return mtf.cast(mtf.less(query_pos, memory_pos), dtype) * -1e9


def attention_mask_same_segment(
    query_segment, memory_segment=None, dtype=tf.float32):
  """Bias for attention where attention between segments is disallowed.

  Args:
    query_segment: a mtf.Tensor with shape [..., length_dim]
    memory_segment: a mtf.Tensor with shape [..., memory_length_dim]
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [..., length_dim, memory_length_dim]
  """
  memory_segment = rename_length_to_memory_length(
      memory_segment or query_segment)
  return mtf.cast(mtf.not_equal(query_segment, memory_segment), dtype) * -1e9


def attention_bias_local_block(mesh, block_length, memory_length,
                               dtype=tf.int32):
  """Bias for attention for local blocks where attention to right is disallowed.

  Args:
    mesh: a MeshTensorflow object
    block_length: a mtf.Dimension
    memory_length: a mtf.Dimension
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [rows, cols]
  """
  mask = mtf.cast(mtf.less(mtf.range(mesh, block_length, dtype=dtype),
                           mtf.range(mesh, memory_length, dtype=dtype)),
                  dtype=dtype)
  mask = mtf.cast(mask, dtype=tf.float32)  * -1e9
  return mask
