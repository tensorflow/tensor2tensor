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
"""Some customization of common_attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

import tensorflow as tf


from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_152
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152  # pylint: disable=unused-import
from tensorflow.python.ops import inplace_ops


def summarize_tensors(tensor_dict, tag=None):
  """Summarize the tensors.

  Args:
    tensor_dict: a dictionary of tensors.
    tag: name scope of the summary; defaults to tensors/.
  """
  if tag is None:
    tag = "tensors/"

  for t_name in list(tensor_dict):
    t = tensor_dict[t_name]
    tf.summary.histogram(tag + t_name, t)


def image_embedding(images,
                    model_fn=resnet_v1_152,
                    trainable=True,
                    is_training=True,
                    weight_decay=0.0001,
                    batch_norm_decay=0.997,
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True,
                    add_summaries=False,
                    reuse=False):
  """Extract image features from pretrained resnet model."""

  is_resnet_training = trainable and is_training

  batch_norm_params = {
      "is_training": is_resnet_training,
      "trainable": trainable,
      "decay": batch_norm_decay,
      "epsilon": batch_norm_epsilon,
      "scale": batch_norm_scale,
  }

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(model_fn.__name__, [images], reuse=reuse) as scope:
    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm],
                            is_training=is_resnet_training,
                            trainable=trainable):
          with slim.arg_scope([slim.max_pool2d], padding="SAME"):
            net, end_points = model_fn(
                images, num_classes=None, global_pool=False,
                is_training=is_resnet_training,
                reuse=reuse, scope=scope)

  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        shared_rel=False,
                        max_relative_position=None,
                        image_shapes=None,
                        attention_type="dot_product",
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        max_length=None,
                        vars_3d=False,
                        scale_dotproduct=True,
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
    shared_rel: boolean to share relative embeddings
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    max_length: an integer - needed by relative attention
    vars_3d: use 3-dimensional variables for input/output transformations
    scale_dotproduct: whether to normalize the attention product.
    **kwargs (dict): Parameters for the attention function

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.

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
  vars_3d_num_heads = num_heads if vars_3d else 0
  with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

    if cache is None or memory_antecedent is None:
      q, k, v = common_attention.compute_qkv(
          query_antecedent, memory_antecedent,
          total_key_depth, total_value_depth, q_filter_width,
          kv_filter_width, q_padding, kv_padding,
          vars_3d_num_heads=vars_3d_num_heads)
    if cache is not None:
      if attention_type != "dot_product":
        # TODO(petershaw): Support caching when using relative position
        # representations, i.e. "dot_product_relative" attention.
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = common_attention.compute_attention_component(
            query_antecedent, total_key_depth,
            q_filter_width, q_padding, "q",
            vars_3d_num_heads=vars_3d_num_heads)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = common_attention.split_heads(k, num_heads)
        v = common_attention.split_heads(v, num_heads)
        decode_loop_step = kwargs.get("decode_loop_step")
        if decode_loop_step is None:
          k = cache["k"] = tf.concat([cache["k"], k], axis=2)
          v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
          # Inplace update is required for inference on TPU.
          # Inplace_ops only supports inplace_update on the first dimension.
          # The performance of current implementation is better than updating
          # the tensor by adding the result of matmul(one_hot,
          # update_in_current_step)
          tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
          tmp_k = inplace_ops.alias_inplace_update(
              tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
          k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
          tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
          tmp_v = inplace_ops.alias_inplace_update(
              tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
          v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

    q = common_attention.split_heads(q, num_heads)
    if cache is None:
      k = common_attention.split_heads(k, num_heads)
      v = common_attention.split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      if scale_dotproduct:
        q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      x = common_attention.dot_product_attention(
          q, k, v, bias, dropout_rate, image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims)
    elif attention_type == "dot_product_relative":
      x = common_attention.dot_product_attention_relative(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          make_image_summary=make_image_summary)
    elif attention_type == "dot_product_relative_v2":
      x = common_attention.dot_product_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_length,
          dropout_rate,
          image_shapes,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims)
    elif attention_type == "local_within_block_mask_right":
      x = common_attention.masked_within_block_local_attention_1d(
          q, k, v, block_length=block_length)
    elif attention_type == "rel_local_mask_right":
      x = common_attention.masked_rel_local_attention_1d(
          q, k, v, block_length=block_length,
          make_image_summary=make_image_summary,
          dropout_rate=dropout_rate,
          share_rel_embed=shared_rel)
    elif attention_type == "local_mask_right":
      x = common_attention.masked_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary)
    elif attention_type == "local_unmasked":
      x = common_attention.local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = common_attention.masked_dilated_self_attention_1d(
          q, k, v, block_length, block_width,
          gap_size, num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = common_attention.dilated_self_attention_1d(
          q, k, v, block_length, block_width,
          gap_size, num_memory_blocks)
    x = common_attention.combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.cast(o_var, x.dtype)
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = common_layers.dense(
          x, output_depth, use_bias=False, name="output_transform")
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x
