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
"""Utils for attention mechanism for images."""

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils

import tensorflow as tf


class AttentionType(object):
  """Types of attention type used in cia."""
  LOCAL_1D = "local_1d"
  LOCAL_2D = "local_2d"
  GLOBAL = "global"
  GLOCAL = "global_local"
  DILATED = "dilated"
  MOE_LOCAL_1D = "moe_local1d"
  LOCAL_BLOCK = "local_block"
  NON_CAUSAL_1D = "local_1d_noncausal"

  @staticmethod
  def get_choices():
    return [
        AttentionType.GLOBAL,
        AttentionType.GLOCAL,
        AttentionType.MOE_LOCAL_1D,
        AttentionType.LOCAL_1D,
        AttentionType.LOCAL_2D,
        AttentionType.LOCAL_BLOCK,
        AttentionType.DILATED,
        AttentionType.NON_CAUSAL_1D,
    ]


def maybe_reshape_4d_to_3d(x):
  """Reshape input from 4D to 3D if necessary."""
  x_shape = common_layers.shape_list(x)
  is_4d = False
  if len(x_shape) == 4:
    x = tf.reshape(x, [x_shape[0], x_shape[1]*x_shape[2], x_shape[3]])
    is_4d = True
  return x, x_shape, is_4d


def local_attention_2d(x, hparams, attention_type="local_attention_2d"):
  """Local 2d, self attention layer."""
  # self-attention
  with tf.variable_scope("local_2d_self_att"):
    y = common_attention.multihead_attention_2d(
        x,
        None,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        attention_type=attention_type,
        query_shape=hparams.query_shape,
        memory_flange=hparams.memory_flange,
        name="self_attention")
  return y


def local_within_block_attention(x,
                                 self_attention_bias,
                                 hparams,
                                 attention_type="local_within_block_mask_right",
                                 q_padding="VALID",
                                 kv_padding="VALID"):
  """Local within block self attention."""
  x_new, x_shape, is_4d = maybe_reshape_4d_to_3d(x)
  with tf.variable_scope("local_within_block"):
    y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x_new, hparams),
        None,
        self_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=attention_type,
        block_width=hparams.block_width,
        block_length=hparams.block_length,
        q_padding=q_padding,
        kv_padding=kv_padding,
        q_filter_width=hparams.q_filter_width,
        kv_filter_width=hparams.kv_filter_width,
        name="local_within_block")
    if is_4d:
      y = tf.reshape(y, x_shape)
    return y


def local_attention_1d(x,
                       hparams,
                       attention_type="local_unmasked",
                       q_padding="VALID",
                       kv_padding="VALID"):
  """Local 1d self attention."""
  # self-attention
  x, x_shape, is_4d = maybe_reshape_4d_to_3d(x)
  with tf.variable_scope("local_1d_self_att"):
    y = common_attention.multihead_attention(
        x,
        None,
        None,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=attention_type,
        block_width=hparams.block_width,
        block_length=hparams.block_length,
        q_padding=q_padding,
        kv_padding=kv_padding,
        q_filter_width=hparams.q_filter_width,
        kv_filter_width=hparams.kv_filter_width,
        make_image_summary=False,
        name="self_attention")
    if is_4d:
      y = tf.reshape(y, x_shape)
    return y


def dilated_attention_1d(x,
                         hparams,
                         attention_type="masked_dilated_1d",
                         q_padding="VALID",
                         kv_padding="VALID",
                         gap_size=2):
  """Dilated 1d self attention."""
  # self-attention
  x, x_shape, is_4d = maybe_reshape_4d_to_3d(x)
  with tf.variable_scope("masked_dilated_1d"):
    y = common_attention.multihead_attention(
        x,
        None,
        None,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        attention_type=attention_type,
        block_width=hparams.block_width,
        block_length=hparams.block_length,
        q_padding=q_padding,
        kv_padding=kv_padding,
        q_filter_width=hparams.q_filter_width,
        kv_filter_width=hparams.kv_filter_width,
        gap_size=gap_size,
        num_memory_blocks=hparams.num_memory_blocks,
        name="self_attention")
    if is_4d:
      y = tf.reshape(y, x_shape)
      y.set_shape([None, None, None, hparams.hidden_size])
    return y


def local_global_attention(x,
                           self_attention_bias,
                           hparams,
                           q_padding="LEFT",
                           kv_padding="LEFT"):
  """Local and global 1d self attention."""
  with tf.variable_scope("self_local_global_att"):
    [x_global, x_local] = tf.split(x, 2, axis=-1)
    split_hidden_size = int(hparams.hidden_size / 2)
    split_heads = int(hparams.num_heads / 2)
    if self_attention_bias is not None:
      self_attention_bias = get_self_attention_bias(x)
    y_global = common_attention.multihead_attention(
        x_global,
        None,
        self_attention_bias,
        hparams.attention_key_channels or split_hidden_size,
        hparams.attention_value_channels or split_hidden_size,
        split_hidden_size,
        split_heads,
        hparams.attention_dropout,
        q_filter_width=hparams.q_filter_width,
        kv_filter_width=hparams.kv_filter_width,
        q_padding=q_padding,
        kv_padding=kv_padding,
        name="global_self_att")
    y_local = common_attention.multihead_attention(
        x_local,
        None,
        None,
        hparams.attention_key_channels or split_hidden_size,
        hparams.attention_value_channels or split_hidden_size,
        split_hidden_size,
        split_heads,
        hparams.attention_dropout,
        attention_type="local_masked",
        block_length=hparams.block_length,
        block_width=hparams.block_width,
        q_filter_width=hparams.q_filter_width,
        kv_filter_width=hparams.kv_filter_width,
        q_padding=q_padding,
        kv_padding=kv_padding,
        name="local_self_att")
    y = tf.concat([y_global, y_local], axis=-1)
    return y


def full_self_attention(x,
                        self_attention_bias,
                        hparams,
                        q_padding="LEFT",
                        kv_padding="LEFT"):
  """Full self-attention layer."""
  x, x_shape, is_4d = maybe_reshape_4d_to_3d(x)
  if self_attention_bias is not None:
    self_attention_bias = get_self_attention_bias(x)
  with tf.variable_scope("self_att"):
    y = common_attention.multihead_attention(
        x,
        None,
        self_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        q_filter_width=hparams.q_filter_width,
        kv_filter_width=hparams.kv_filter_width,
        q_padding=q_padding,
        kv_padding=kv_padding,
        name="self_att")
    if is_4d:
      y = tf.reshape(y, [x_shape[0], x_shape[1], x_shape[2], x_shape[3]])
      y.set_shape([None, None, None, hparams.hidden_size])
    return y


def encdec_attention_1d(x,
                        encoder_output,
                        encoder_decoder_attention_bias,
                        hparams):
  """Local 1d self attention."""
  x, x_shape, is_4d = maybe_reshape_4d_to_3d(x)
  encoder_output, _, _ = maybe_reshape_4d_to_3d(encoder_output)
  with tf.variable_scope("encdec_attention"):
    # Encoder Decoder attention
    y = common_attention.multihead_attention(
        x,
        encoder_output,
        encoder_decoder_attention_bias,
        hparams.attention_key_channels or hparams.hidden_size,
        hparams.attention_value_channels or hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        name="encdec_attention")
  if is_4d:
    y = tf.reshape(y, x_shape)
    y.set_shape([None, None, None, hparams.hidden_size])
  return y


def transformer_decoder_layers(inputs,
                               encoder_output,
                               num_layers,
                               hparams,
                               self_attention_bias=None,
                               encoder_decoder_attention_bias=None,
                               attention_type=AttentionType.LOCAL_2D,
                               losses=None,
                               name="transformer"):
  """Multi layer transformer."""
  x = inputs
  x = tf.nn.dropout(x, 1.0 - hparams.layer_prepostprocess_dropout)
  if attention_type == AttentionType.DILATED:
    assert len(hparams.gap_sizes) == num_layers
  for layer in range(num_layers):
    with tf.variable_scope("%s_layer_%d" % (name, layer)):
      # self-attention + skip connections
      if attention_type == AttentionType.LOCAL_2D:
        y = local_attention_2d(common_layers.layer_preprocess(x, hparams),
                               hparams,
                               attention_type="masked_local_attention_2d")
      elif attention_type == AttentionType.LOCAL_1D:
        y = local_attention_1d(common_layers.layer_preprocess(x, hparams),
                               hparams,
                               attention_type="local_mask_right",
                               q_padding="LEFT", kv_padding="LEFT")
      elif attention_type == AttentionType.NON_CAUSAL_1D:
        y = local_attention_1d(common_layers.layer_preprocess(x, hparams),
                               hparams,
                               attention_type="local_unmasked",
                               q_padding="VALID", kv_padding="VALID")
      elif attention_type == AttentionType.LOCAL_BLOCK:
        y = local_within_block_attention(
            common_layers.layer_preprocess(x, hparams),
            self_attention_bias, hparams,
            attention_type="local_within_block_mask_right",
            q_padding="LEFT", kv_padding="LEFT")
      elif attention_type == AttentionType.GLOCAL:
        y = local_global_attention(common_layers.layer_preprocess(x, hparams),
                                   self_attention_bias, hparams,
                                   q_padding="LEFT", kv_padding="LEFT")
      elif attention_type == AttentionType.DILATED:
        y = dilated_attention_1d(common_layers.layer_preprocess(x, hparams),
                                 hparams, q_padding="LEFT",
                                 kv_padding="LEFT",
                                 gap_size=hparams.gap_sizes[layer])
      elif attention_type == AttentionType.GLOBAL:
        y = full_self_attention(common_layers.layer_preprocess(x, hparams),
                                self_attention_bias, hparams,
                                q_padding="LEFT", kv_padding="LEFT")
      x = common_layers.layer_postprocess(x, y, hparams)
      # enc-dec attention + skip connections
      if encoder_output is not None:
        y = encdec_attention_1d(common_layers.layer_preprocess(x, hparams),
                                encoder_output,
                                encoder_decoder_attention_bias,
                                hparams)
        x = common_layers.layer_postprocess(x, y, hparams)
      # feed-fwd layers + skip connections
      y = ffn_layer(common_layers.layer_preprocess(x, hparams), hparams,
                    losses=losses)
      x = common_layers.layer_postprocess(x, y, hparams)
  return common_layers.layer_preprocess(x, hparams)


def transformer_encoder_layers(inputs,
                               num_layers,
                               hparams,
                               attention_type=AttentionType.GLOBAL,
                               self_attention_bias=None,
                               q_padding="VALID",
                               kv_padding="VALID",
                               name="transformer"):
  """Multi layer transformer encoder."""
  x = inputs
  x = tf.nn.dropout(x, 1.0 - hparams.layer_prepostprocess_dropout)

  for layer in range(num_layers):
    # attention layers + skip connections
    with tf.variable_scope("%s_layer_%d" % (name, layer)):
      if attention_type == AttentionType.LOCAL_2D:
        y = local_attention_2d(common_layers.layer_preprocess(x, hparams),
                               hparams,
                               attention_type="local_attention_2d")
      elif attention_type == AttentionType.LOCAL_1D:
        y = local_attention_1d(common_layers.layer_preprocess(x, hparams),
                               hparams,
                               attention_type="local_unmasked",
                               q_padding=q_padding, kv_padding=kv_padding)
      elif attention_type == AttentionType.GLOBAL:
        y = full_self_attention(common_layers.layer_preprocess(x, hparams),
                                self_attention_bias, hparams,
                                q_padding=q_padding, kv_padding=kv_padding)
      x = common_layers.layer_postprocess(x, y, hparams)
      # feed-fwd layer + skip connections
      y = ffn_layer(common_layers.layer_preprocess(x, hparams), hparams)
      x = common_layers.layer_postprocess(x, y, hparams)
  return common_layers.layer_preprocess(x, hparams)


def ffn_layer(x, hparams, losses=None):
  """ffn layer transformer."""
  with tf.variable_scope("ffn"):
    if hparams.ffn_layer == "none":
      return x
    if hparams.ffn_layer == "conv_hidden_relu":
      y = common_layers.dense_relu_dense(
          x,
          hparams.filter_size,
          hparams.hidden_size,
          dropout=hparams.relu_dropout)
    elif hparams.ffn_layer == "normed_conv_hidden_relu":
      y = common_layers.normed_conv_hidden_relu(
          x,
          hparams.norm_type,
          hparams.layer_norm_epsilon,
          hparams.filter_size,
          hparams.hidden_size,
          dropout=hparams.relu_dropout,
          norm_name="convnorm")
    elif hparams.ffn_layer == "self_attention_ffn":
      x_shape = tf.shape(x)
      x = tf.reshape(x, [x_shape[0], -1, hparams.hidden_size])
      y = common_attention.ffn_self_attention_layer(
          x, hparams.filter_size, hparams.hidden_size, hparams.num_parts,
          hparams.attention_dropout, hparams.share_kv)
      y = tf.reshape(y, x_shape)
    elif hparams.ffn_layer == "local_moe_tpu":
      overhead = (hparams.moe_overhead_train
                  if hparams.mode == tf.estimator.ModeKeys.TRAIN
                  else hparams.moe_overhead_eval)
      x, x_shape, is_4d = maybe_reshape_4d_to_3d(x)
      y, loss = expert_utils.local_moe_tpu(
          x, hparams.filter_size // 2,
          hparams.hidden_size,
          hparams.moe_num_experts, overhead=overhead,
          loss_coef=hparams.moe_loss_coef)
      if is_4d:
        y = tf.reshape(y, x_shape)
      if losses is None:
        raise ValueError(
            "transformer_ffn_layer with type local_moe_tpu must pass in "
            "a losses list")
      losses.append(loss)
    else:
      assert hparams.ffn_layer == "glu_ffn"
      y = common_layers.gated_linear_unit_layer(x)
    return y


def get_self_attention_bias(x):
  """Creates masked self attention bias.

  Args:
    x: A tensor of shape [batch, length, depth]

  Returns:
    self_attention_bias: A tensor of shape [length, length, 1]
  """

  x_shape = common_layers.shape_list(x)
  self_attention_bias = common_attention.attention_bias_lower_triangle(
      x_shape[1])
  return self_attention_bias


def transformer_layers_sharded(dp,
                               ps_devices,
                               inputs,
                               num_layers,
                               hparams,
                               self_attention_bias=None,
                               enc_output=None,
                               attention_type=AttentionType.GLOBAL,
                               name="transformer"):
  """Multi layer transformer, sharded by the data parallelism dp."""
  x = inputs
  extra_loss = tf.constant(0.0)
  moe_hidden_sizes = [int(s) for s in hparams.moe_hidden_sizes.split(",")]
  expert_fn = expert_utils.ffn_expert_fn(
      hparams.hidden_size, moe_hidden_sizes, hparams.hidden_size)
  x = dp(tf.nn.dropout, x, 1.0 - hparams.layer_prepostprocess_dropout)
  for layer in range(num_layers):
    with tf.variable_scope("%s_layer_%d" % (name, layer)):
      # self-attention
      if attention_type == AttentionType.LOCAL_2D:
        y = dp(local_attention_2d(common_layers.layer_preprocess(x, hparams),
                                  hparams,
                                  attention_type="masked_local_attention_2d"))
      elif attention_type == AttentionType.LOCAL_1D:
        y = dp(local_attention_1d(common_layers.layer_preprocess(x, hparams),
                                  hparams,
                                  attention_type="local_mask_right",
                                  q_padding="LEFT", kv_padding="LEFT"))
      elif attention_type == AttentionType.GLOCAL:
        y = dp(local_global_attention(
            common_layers.layer_preprocess(x, hparams), self_attention_bias,
            hparams, q_padding="LEFT", kv_padding="LEFT"))
      elif attention_type == AttentionType.GLOBAL:
        self_attention_bias = dp(get_self_attention_bias(x))
        y = dp(full_self_attention(common_layers.layer_preprocess(x, hparams),
                                   self_attention_bias, hparams,
                                   q_padding="LEFT", kv_padding="LEFT"))
      x = common_layers.layer_postprocess(x, y, hparams)
      if enc_output is not None:
        y = dp(encdec_attention_1d(common_layers.layer_preprocess(x, hparams),
                                   enc_output, None, hparams))
        x = dp(common_layers.layer_postprocess, x, y, hparams)
      with tf.variable_scope("ffn"):
        if str(layer) in hparams.moe_layers_decoder.split(","):
          y, loss = expert_utils.distributed_moe(
              dp,
              ps_devices,
              common_layers.layer_preprocess(x, hparams),
              hparams.mode == tf.estimator.ModeKeys.TRAIN,
              input_size=hparams.hidden_size,
              expert_fn=expert_fn,
              num_experts=hparams.moe_num_experts,
              k=hparams.moe_k,
              loss_coef=hparams.moe_loss_coef)
          extra_loss += loss
          x = dp(common_layers.layer_postprocess, x, y, hparams)
        else:
          y = dp(ffn_layer, common_layers.layer_preprocess(x, hparams), hparams)
          x = dp(common_layers.layer_postprocess, x, y, hparams)
  return dp(common_layers.layer_preprocess, x, hparams), extra_loss


def postprocess_image(x, rows, cols, hparams):
  """Postprocessing after decoding."""
  batch = common_layers.shape_list(x)[0]
  channels = 256
  x = tf.reshape(x, [batch, rows, cols, hparams.hidden_size])
  targets = tf.layers.dense(x, 256, use_bias=True, activation=None,
                            name="output_conv")
  if (hparams.mode == tf.contrib.learn.ModeKeys.INFER and
      hparams.block_raster_scan):
    y = targets
    y = tf.reshape(y, [batch, -1, hparams.img_len*3, channels])
    yshape = common_layers.shape_list(y)
    block_length = hparams.query_shape[0]
    block_width = hparams.query_shape[1]

    # Break into block row wise.
    y = tf.reshape(y,
                   [batch, yshape[1] // block_length,
                    block_length,
                    yshape[2], channels])
    yshape = common_layers.shape_list(y)
    # Break into blocks width wise.
    y_blocks = tf.reshape(y,
                          [batch, yshape[1], yshape[2],
                           yshape[3] // block_width,
                           block_width, channels])

    # Reshape targets as [batch_size, num_blocks_rows, num_block_cols,
    # block_length, block_width, channels]
    targets = tf.transpose(y_blocks, [0, 1, 3, 2, 4, 5])

  return targets


def prepare_encoder(inputs, hparams, attention_type="local_1d"):
  """Prepare encoder for images."""
  x = prepare_image(inputs, hparams, name="enc_channels")
  # Add position signals.
  x = add_pos_signals(x, hparams, "enc_pos")
  x_shape = common_layers.shape_list(x)
  if attention_type == "local_1d":
    x = tf.reshape(x, [x_shape[0], x_shape[1]*x_shape[2], hparams.hidden_size])
    x.set_shape([None, None, hparams.hidden_size])
  elif attention_type == "local_2d":
    x.set_shape([None, None, None, hparams.hidden_size])
  return x


def prepare_decoder(targets, hparams):
  """Prepare decoder for images."""
  targets_shape = common_layers.shape_list(targets)
  channels = hparams.num_channels
  curr_infer_length = None

  # during training, images are [batch, IMG_LEN, IMG_LEN, 3].
  # At inference, they are [batch, curr_infer_length, 1, 1]
  if hparams.mode == tf.contrib.learn.ModeKeys.INFER:
    curr_infer_length = targets_shape[1]
    if hparams.block_raster_scan:
      assert hparams.img_len*channels % hparams.query_shape[1] == 0
      assert hparams.img_len % hparams.query_shape[0] == 0
      total_block_width = hparams.img_len*channels
      # Decoding is in block raster scan order. We divide the image into
      # hparams.query_shape blocks and then decode each block in raster scan.
      # To make that compatible with our inference pipeline, pad the target so
      # that rows is a multiple of query_shape and columns is a multiple of
      # hparams.img_len*channels
      curr_infer_length = targets_shape[1]
      block_padding_factor = total_block_width * hparams.query_shape[0]
      targets = tf.pad(targets, [
          [0, 0], [0, -curr_infer_length % block_padding_factor],
          [0, 0], [0, 0]])

      num_blocks = total_block_width // hparams.query_shape[1]
      # Reshape the image to represent blocks
      target_blocks = tf.reshape(
          targets, [targets_shape[0], -1, num_blocks, hparams.query_shape[0],
                    hparams.query_shape[1]])
      # Transpose to read the image in 2D fashion.
      targets = tf.transpose(target_blocks, [0, 1, 3, 2, 4])
    else:
      # add padding to make sure the size of targets is a multiple of img_height
      # times number of channels. This is  needed for positional encodings and
      # for doing the RGB lookup.
      padding_factor = channels * hparams.img_len
      targets = tf.pad(targets, [
          [0, 0], [0, -curr_infer_length % padding_factor], [0, 0], [0, 0]])
    targets = tf.reshape(targets,
                         [targets_shape[0], -1, hparams.img_len, channels])
  # Preprocess image
  x = prepare_image(targets, hparams, name="dec_channels")
  x_shape = common_layers.shape_list(x)
  if (hparams.dec_attention_type == AttentionType.LOCAL_2D or
      hparams.dec_attention_type == AttentionType.LOCAL_BLOCK):
    x = common_attention.right_shift_blockwise(x, hparams.query_shape)
    x = add_pos_signals(x, hparams, "dec_pos")
  else:
    # Add position signals
    x = tf.reshape(x, [targets_shape[0],
                       x_shape[1]*x_shape[2], hparams.hidden_size])
    x = common_layers.shift_right_3d(x)
    x = tf.reshape(x, [targets_shape[0],
                       x_shape[1], x_shape[2], hparams.hidden_size])
    x = add_pos_signals(x, hparams, "dec_pos")
  x = common_layers.cast_like(x, targets)
  return x, x_shape[1], x_shape[2]


def prepare_image(inputs, hparams, name=None):
  """Prepare image."""
  inputs_shape = common_layers.shape_list(inputs)
  batch = inputs_shape[0]
  orig_rows = inputs_shape[1]
  orig_cols = inputs_shape[2]
  channels = hparams.num_channels

  hidden_size = hparams.hidden_size
  # Only do lookup if the modality is identity
  if hparams.target_modality == "image:identity":
    inputs = tf.to_int32(inputs)
    x = get_channel_embeddings(channels, inputs, hidden_size, name=name)
  else:
    x = inputs
  x = tf.reshape(x, [batch, orig_rows, orig_cols * channels, hidden_size])

  return x


def create_output(decoder_output, rows, cols, targets, hparams):
  """Create output from decoder output and vars."""
  decoded_image = postprocess_image(decoder_output, rows, cols, hparams)
  targets_shape = common_layers.shape_list(targets)
  if hparams.mode == tf.estimator.ModeKeys.PREDICT:
    # Hardcoding that the number of intensity values is 256.
    y = tf.reshape(decoded_image, [targets_shape[0], -1, 1, 1, 256])
    output = y[:, :targets_shape[1], :, :, :]
  else:
    output = tf.reshape(decoded_image, [
        targets_shape[0], targets_shape[1], targets_shape[2],
        targets_shape[3], 256
    ])
  return output


def get_channel_embeddings(io_depth, targets, hidden_size, name="channel"):
  """Get separate embedding for each of the channels."""
  targets_split = tf.split(targets, io_depth, axis=3)
  rgb_embedding_var = tf.get_variable("rgb_target_emb_%s" % name,
                                      [256 * io_depth, hidden_size])
  rgb_embedding_var = tf.identity(rgb_embedding_var)
  rgb_embedding_var *= float(hidden_size)**0.5
  channel_target_embs = []
  for i in range(io_depth):
    # Adding the channel offsets to get the right embedding since the
    # embedding tensor has shape 256 * io_depth, hidden_size
    target_ids = tf.squeeze(targets_split[i], axis=3) + i * 256
    target_embs = common_layers.gather(rgb_embedding_var, target_ids)
    channel_target_embs.append(target_embs)

  return tf.concat(channel_target_embs, axis=-1)


def add_pos_signals(x, hparams, name="pos_emb"):
  with tf.variable_scope(name, reuse=False):
    if hparams.pos == "timing":
      x = common_attention.add_timing_signal_nd(x)
    else:
      assert hparams.pos == "emb"
      x = common_attention.add_positional_embedding_nd(
          x, hparams.max_length, name)
  return x
