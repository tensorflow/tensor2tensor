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

"""Utilities for area attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf


def lengths_to_area_mask(feature_length, length, max_area_size):
  """Generates a non-padding mask for areas based on lengths.

  Args:
    feature_length: a tensor of [batch_size]
    length: the length of the batch
    max_area_size: the maximum area size considered
  Returns:
    mask: a tensor in shape of [batch_size, num_areas]
  """

  paddings = tf.cast(tf.expand_dims(
      tf.logical_not(
          tf.sequence_mask(feature_length, maxlen=length)), 2), tf.float32)
  _, _, area_sum, _, _ = compute_area_features(paddings,
                                               max_area_width=max_area_size)
  mask = tf.squeeze(tf.logical_not(tf.cast(area_sum, tf.bool)), [2])
  return mask


def _pool_one_shape(features_2d, area_width, area_height, batch_size,
                    width, height, depth, fn=tf.reduce_max, name=None):
  """Pools for an area in features_2d.

  Args:
    features_2d: a Tensor in a shape of [batch_size, height, width, depth].
    area_width: the max width allowed for an area.
    area_height: the max height allowed for an area.
    batch_size: the batch size.
    width: the width of the memory.
    height: the height of the memory.
    depth: the depth of the features.
    fn: the TF function for the pooling.
    name: the op name.
  Returns:
    pool_tensor: A Tensor of shape [batch_size, num_areas, depth]
  """
  with tf.name_scope(name, default_name="pool_one_shape"):
    images = []
    for y_shift in range(area_height):
      image_height = tf.maximum(height - area_height + 1 + y_shift, 0)
      for x_shift in range(area_width):
        image_width = tf.maximum(width - area_width + 1 + x_shift, 0)
        area = features_2d[:, y_shift:image_height, x_shift:image_width, :]
        flatten_area = tf.reshape(area, [batch_size, -1, depth, 1])
        images.append(flatten_area)
    image_tensor = tf.concat(images, axis=3)
    max_tensor = fn(image_tensor, axis=3)
  return max_tensor


def basic_pool(features, max_area_width, max_area_height=1, height=1,
               fn=tf.reduce_max, name=None):
  """Pools for each area based on a given pooling function (fn).

  Args:
    features: a Tensor in a shape of [batch_size, height * width, depth].
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    height: the height of the image.
    fn: the TF function for the pooling.
    name: the namescope.
  Returns:
    pool_results: A Tensor of shape [batch_size, num_areas, depth]
    area_heights: A Tensor of shape [batch_size, num_areas, 1]
    area_widths: A Tensor of shape [batch_size, num_areas, 1]
  """
  with tf.name_scope(name, default_name="basic_pool"):
    feature_shape = common_layers.shape_list(features)
    batch_size = feature_shape[0]
    length = feature_shape[-2]
    depth = feature_shape[-1]
    width = length // height
    features_2d = tf.reshape(features, [batch_size, height, width, depth])
    height_list = []
    width_list = []
    pool_list = []
    size_tensor = tf.ones_like(features_2d[:, :, :, 0], dtype=tf.int32)
    for area_height in range(max_area_height):
      for area_width in range(max_area_width):
        pool_tensor = _pool_one_shape(features_2d,
                                      area_width=area_width + 1,
                                      area_height=area_height + 1,
                                      batch_size=batch_size,
                                      width=width,
                                      height=height,
                                      depth=depth,
                                      fn=fn)
        pool_list.append(
            tf.reshape(pool_tensor, [batch_size, -1, depth]))
        height_list.append(
            tf.reshape(
                size_tensor[:, area_height:, area_width:] *\
                (area_height + 1), [batch_size, -1]))
        width_list.append(
            tf.reshape(
                size_tensor[:, area_height:, area_width:] *\
                (area_width + 1), [batch_size, -1]))
    pool_results = tf.concat(pool_list, axis=1)
    area_heights = tf.expand_dims(tf.concat(height_list, axis=1), 2)
    area_widths = tf.expand_dims(tf.concat(width_list, axis=1), 2)
  return pool_results, area_heights, area_widths


def _compute_sum_image(features, max_area_width, max_area_height=1, height=1,
                       name=None):
  """Computes area sums for features.

  Args:
    features: a Tensor in a shape of [batch_size, height * width, depth].
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    height: the height of the image.
    name: the namescope.
  Returns:
    sum_image: A Tensor of shape [batch_size, num_areas, depth]
    area_heights: A Tensor of shape [batch_size, num_areas, 1]
    area_widths: A Tensor of shape [batch_size, num_areas, 1]
  """
  with tf.name_scope(name, default_name="compute_sum_image"):
    feature_shape = common_layers.shape_list(features)
    batch_size = feature_shape[0]
    length = feature_shape[-2]
    depth = feature_shape[-1]
    width = length // height
    features_2d = tf.reshape(features, [batch_size, height, width, depth])
    width_cum = tf.cumsum(features_2d, axis=-2, name="compute_integral_h")
    integral_image = tf.cumsum(width_cum, axis=-3, name="compute_integral_v")
    padded_image = tf.pad(
        integral_image, [[0, 0], [1, 0], [1, 0], [0, 0]], constant_values=0)
    height_list = []
    width_list = []
    dst_images = []
    src_images_diag = []
    src_images_h = []
    src_images_v = []
    size_tensor = tf.ones_like(padded_image[:, :, :, 0],
                               dtype=tf.int32)
    for area_height in range(max_area_height):
      for area_width in range(max_area_width):
        dst_images.append(
            tf.reshape(
                padded_image[:, area_height + 1:, area_width + 1:, :],
                [batch_size, -1, depth]))
        src_images_diag.append(
            tf.reshape(
                padded_image[:, :-area_height - 1, :-area_width - 1, :],
                [batch_size, -1, depth]))
        src_images_h.append(
            tf.reshape(
                padded_image[:, area_height + 1:, :-area_width - 1, :],
                [batch_size, -1, depth]))
        src_images_v.append(
            tf.reshape(
                padded_image[:, :-area_height - 1, area_width + 1:, :],
                [batch_size, -1, depth]))
        height_list.append(
            tf.reshape(
                size_tensor[:, area_height + 1:, area_width + 1:] *\
                (area_height + 1), [batch_size, -1]))
        width_list.append(
            tf.reshape(
                size_tensor[:, area_height + 1:, area_width + 1:] *\
                (area_width + 1), [batch_size, -1]))
    sum_image = tf.subtract(
        tf.concat(dst_images, axis=1) + tf.concat(src_images_diag, axis=1),
        tf.concat(src_images_v, axis=1) + tf.concat(src_images_h, axis=1))
    area_heights = tf.expand_dims(tf.concat(height_list, axis=1), 2)
    area_widths = tf.expand_dims(tf.concat(width_list, axis=1), 2)
  return sum_image, area_heights, area_widths


def compute_area_features(features, max_area_width, max_area_height=1, height=1,
                          epsilon=1e-6):
  """Computes features for each area.

  Args:
    features: a Tensor in a shape of [batch_size, height * width, depth].
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    height: the height of the image.
    epsilon: the epsilon added to the variance for computing standard deviation.
  Returns:
    area_mean: A Tensor of shape [batch_size, num_areas, depth]
    area_std: A Tensor of shape [batch_size, num_areas, depth]
    area_sum: A Tensor of shape [batch_size, num_areas, depth]
    area_heights: A Tensor of shape [batch_size, num_areas, 1]
    area_widths: A Tensor of shape [batch_size, num_areas, 1]
  """
  with tf.name_scope("compute_area_features"):
    tf.logging.info("area_attention compute_area_features: %d x %d",
                    max_area_height, max_area_width)
    area_sum, area_heights, area_widths = _compute_sum_image(
        features, max_area_width=max_area_width,
        max_area_height=max_area_height, height=height)
    area_squared_sum, _, _ = _compute_sum_image(
        tf.pow(features, 2), max_area_width=max_area_width,
        max_area_height=max_area_height, height=height)
    sizes = tf.multiply(area_heights, area_widths)
    float_area_sizes = tf.to_float(sizes)
    area_mean = tf.div(area_sum, float_area_sizes)
    s2_n = tf.div(area_squared_sum, float_area_sizes)
    area_variance = tf.subtract(s2_n, tf.pow(area_mean, 2))
    area_std = tf.sqrt(tf.abs(area_variance) + epsilon)
    return area_mean, area_std, area_sum, area_heights, area_widths


def compute_area_key(features, max_area_width, max_area_height=1, height=1,
                     mode="mean", training=True, name=None):
  """Computes the key for each area.

  Args:
    features: a Tensor in a shape of [batch_size, height * width, depth].
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    height: the height of the image.
    mode: whether to combine different area features or only use
        the vector mean of each area, which can be "mean", "concat", "sum",
        "sample_concat", and "sample_sum".
    training: indicating if it is in the training mode.
    name: the name for setting the variable scope.
  Returns:
    area_key: a Tensor in the shape of [batch_size, num_areas, depth]
  """

  tf.logging.info("area_attention mode=%s", mode)
  area_mean, area_std, _, area_heights, area_widths =\
      compute_area_features(features, max_area_width=max_area_width,
                            max_area_height=max_area_height, height=height)
  if mode == "mean":
    return area_mean
  elif mode == "max":
    area_max, _, _ = basic_pool(features, max_area_width=max_area_width,
                                max_area_height=max_area_height, height=height)
    return area_max
  elif mode == "sample":
    if training:
      area_mean += (area_std * tf.random_normal(tf.shape(area_std)))
    return area_mean
  with tf.variable_scope(
      name, default_name="combine_area_features",
      values=[area_mean, area_std, area_heights, area_widths]):
    depth = common_layers.shape_list(area_mean)[-1]
    height_embed = tf.nn.embedding_lookup(
        params=tf.get_variable("area_height_emb",
                               [max_area_height, depth // 2]),
        ids=area_heights[:, :, 0] - 1)
    width_embed = tf.nn.embedding_lookup(
        params=tf.get_variable("area_width_emb",
                               [max_area_width, depth // 2]),
        ids=area_widths[:, :, 0] - 1)
    size_embed = tf.concat([height_embed, width_embed], -1)
    if mode == "concat":
      feature_concat = tf.concat([area_mean, area_std, size_embed], -1)
    elif mode == "max_concat":
      area_max, _, _ = basic_pool(features, max_area_width=max_area_width,
                                  max_area_height=max_area_height,
                                  height=height)
      feature_concat = tf.concat([area_max, size_embed], -1)
    elif mode == "sum":
      feature_concat = size_embed + area_mean + area_std
    elif mode == "sample_concat":
      if training:
        area_mean += (area_std * tf.random_normal(tf.shape(area_std)))
      feature_concat = tf.concat([area_mean, size_embed], -1)
    elif mode == "sample_sum":
      if training:
        area_mean += (area_std * tf.random_normal(tf.shape(area_std)))
      feature_concat = area_mean + size_embed
    else:
      raise ValueError("Unsupported area key mode=%s" % mode)
    feature_hidden = tf.layers.dense(inputs=feature_concat,
                                     units=depth,
                                     activation=tf.nn.relu)
    area_key = tf.layers.dense(feature_hidden, units=depth)
    return area_key


def dot_product_area_attention(q,
                               k,
                               v,
                               bias,
                               dropout_rate=0.0,
                               image_shapes=None,
                               name=None,
                               attention_image_summary=None,
                               save_weights_to=None,
                               dropout_broadcast_dims=None,
                               max_area_width=1,
                               max_area_height=1,
                               memory_height=1,
                               area_key_mode="mean",
                               area_value_mode="sum",
                               top_k_areas=0,
                               area_temperature=1.0,
                               training=True):
  """Dot-product area attention.

  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    attention_image_summary: the callback for making image summary of attention.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    memory_height: the height of the memory.
    area_key_mode: the mode for computing area keys, which can be "mean",
      "concat", "sum", "sample_concat", and "sample_sum".
    area_value_mode: the mode for computing area values, which can be either
      "mean", or "sum".
    top_k_areas: Use the top key areas for attention.
    area_temperature: the temperature for attention softmax.
    training: indicating if it is in the training mode.
  Returns:
    Tensor with shape [..., length_q, depth_v].
  """

  tf.logging.info("dot_product_area_attention: "
                  "area_h=%d, area_w=%d, mem_h=%d, "
                  "area_key_mode=%s, area_value_mode=%s, "
                  "area_temperature=%f",
                  max_area_height, max_area_width, memory_height,
                  area_key_mode, area_value_mode,
                  area_temperature)
  with tf.variable_scope(
      name, default_name="dot_product_area_attention",
      values=[q, k, v]) as scope:
    mem_shape = common_layers.shape_list(k)
    batch_size = mem_shape[0]
    head_size = mem_shape[1]
    length = mem_shape[2]
    depth = mem_shape[3]
    k_area = compute_area_key(
        tf.reshape(k, [-1, length, depth]),
        max_area_width=max_area_width,
        max_area_height=max_area_height,
        height=memory_height,
        mode=area_key_mode,
        training=training)
    if area_value_mode == "mean":
      v_area, _, _, _, _ = compute_area_features(
          tf.reshape(v, [-1, length, depth]), max_area_width=max_area_width,
          max_area_height=max_area_height, height=memory_height)
    elif area_value_mode == "max":
      v_area, _, _ = basic_pool(tf.reshape(v, [-1, length, depth]),
                                max_area_width=max_area_width,
                                max_area_height=max_area_height,
                                height=memory_height,
                                fn=tf.reduce_max)
    elif area_value_mode == "sum":
      _, _, v_area, _, _ = compute_area_features(
          tf.reshape(v, [-1, length, depth]), max_area_width=max_area_width,
          max_area_height=max_area_height, height=memory_height)
    else:
      raise ValueError("Unsupported area value mode=%s" % area_value_mode)
    k = tf.reshape(k_area, [batch_size, head_size, -1, depth])
    v = tf.reshape(v_area, [batch_size, head_size, -1, depth])
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      with tf.name_scope("compute_area_att_bias", values=[bias]):
        bias_shape = common_layers.shape_list(bias)
        mem_length = bias_shape[-1]
        bias_values = tf.reshape(
            tf.to_float(tf.less(bias, -1)), [-1, mem_length, 1])
        _, _, padding_sum, _, _ = compute_area_features(
            bias_values, max_area_width=max_area_width,
            max_area_height=max_area_height, height=memory_height)
        bias = tf.where(
            tf.cast(tf.to_int32(padding_sum), tf.bool),
            tf.fill(tf.shape(padding_sum), -np.inf),
            tf.zeros_like(padding_sum, dtype=tf.float32))
        bias = tf.reshape(bias,
                          [bias_shape[0], bias_shape[1],
                           bias_shape[2], -1])
      logits += bias
    logits = logits / area_temperature
    weights = tf.nn.softmax(logits, name="attention_weights")
    if top_k_areas > 0:
      tf.logging.info("area_attention top_k_areas=%d", top_k_areas)
      top_k = tf.minimum(common_layers.shape_list(weights)[-1], top_k_areas)
      top_weights, _ = tf.nn.top_k(weights, k=top_k)
      min_values = tf.reduce_min(top_weights, -1, keepdims=True)
      weights = tf.where(tf.greater_equal(weights, min_values),
                         weights, tf.zeros_like(weights))
      weights = tf.div(weights, tf.reduce_sum(weights, -1, keepdims=True))
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    # Drop out attention links for each head.
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and attention_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)
