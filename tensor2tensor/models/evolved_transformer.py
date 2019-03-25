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

"""Evolved Transformer model.

This implements the model described in arxiv.org/abs/1901.11117 .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class EvolvedTransformer(transformer.Transformer):
  """The Evolved Transformer from arxiv.org/abs/1901.11117 ."""

  def __init__(self, *args, **kwargs):
    super(EvolvedTransformer, self).__init__(*args, **kwargs)
    self._encoder_function = evolved_transformer_encoder
    self._decoder_function = evolved_transformer_decoder

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha,
                   use_tpu):
    """Forced slow beam decode because cache is not supported.

    Args:
      features: an map of string to `Tensor`.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: Whether or not TPU is being used.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length].
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1).
      }
    """
    return self._beam_decode_slow(features, decode_length, beam_size, top_beams,
                                  alpha, use_tpu)


def evolved_transformer_encoder(encoder_input,
                                encoder_self_attention_bias,
                                hparams,
                                name="encoder",
                                nonpadding=None,
                                save_weights_to=None,
                                make_image_summary=True,
                                losses=None,
                                attn_bias_for_padding=None):
  """Evolved Transformer encoder. See arxiv.org/abs/1901.11117 for more details.

  Note: Pad remover is not supported.

  Args:
    encoder_input: a Tensor.
    encoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias()).
    hparams: hyperparameters for model.
    name: a string.
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be passed in,
      which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used for
      pad_remover(efficiency) and to mask out padding in convolutional layers.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: Not used.
    attn_bias_for_padding: Padded attention bias in case a unidirectional
      encoder is being used where future attention is masked.

  Returns:
    Tensor encoder output.
  """
  del losses

  hidden_state = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      attention_bias = encoder_self_attention_bias
      if attn_bias_for_padding is not None:
        attention_bias = attn_bias_for_padding
      padding = common_attention.attention_bias_to_padding(attention_bias)
      nonpadding = 1.0 - padding

    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):

        with tf.variable_scope("gated_linear_unit"):

          residual_state = hidden_state
          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

          values = common_layers.layers().Dense(
              hparams.hidden_size)(hidden_state)
          gates = common_layers.layers().Dense(
              hparams.hidden_size, activation=tf.nn.sigmoid)(hidden_state)
          hidden_state = values * gates

          hidden_state = common_layers.layer_postprocess(
              residual_state, hidden_state, hparams)

        with tf.variable_scope("conv_branches"):

          residual_state = hidden_state
          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)
          # Mask padding from conv layers.
          mask = tf.tile(
              tf.expand_dims(nonpadding, 2), [1, 1, hparams.hidden_size])
          hidden_state *= mask

          left_output_dim = int(hparams.hidden_size * 4)
          left_state = common_layers.layers().Dense(
              left_output_dim, activation=tf.nn.relu)(hidden_state)
          left_state = tf.nn.dropout(left_state,
                                     1 - hparams.layer_prepostprocess_dropout)

          right_output_dim = int(hparams.hidden_size / 2)
          right_state = common_layers.layers().Conv1D(
              right_output_dim,
              3,
              padding="SAME",
              name="standard_conv_3x1",
              activation=tf.nn.relu)(hidden_state)
          right_state = tf.nn.dropout(right_state,
                                      1 - hparams.layer_prepostprocess_dropout)

          right_state = tf.pad(
              right_state,
              [[0, 0], [0, 0], [0, left_output_dim - right_output_dim]],
              constant_values=0)
          hidden_state = left_state + right_state

          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)
          # Mask padding from conv layer.
          mask = tf.tile(tf.expand_dims(nonpadding, 2), [1, 1, left_output_dim])
          hidden_state *= mask

          separable_conv_9x1 = common_layers.layers().SeparableConv1D(
              right_output_dim, 9, padding="SAME", name="separable_conv_9x1")
          hidden_state = separable_conv_9x1(hidden_state)
          hidden_state = tf.pad(
              hidden_state,
              [[0, 0], [0, 0], [0, hparams.hidden_size - right_output_dim]],
              constant_values=0)

          hidden_state = common_layers.layer_postprocess(
              residual_state, hidden_state, hparams)

        with tf.variable_scope("self_attention"):
          residual_state = hidden_state
          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

          hidden_state = common_attention.multihead_attention(
              hidden_state,
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"))

          hidden_state = common_layers.layer_postprocess(
              residual_state, hidden_state, hparams)

        with tf.variable_scope("dense_layers"):
          residual_state = hidden_state
          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

          hidden_state = common_layers.layers().Dense(
              int(hparams.hidden_size * 4), activation=tf.nn.relu)(hidden_state)
          hidden_state = tf.nn.dropout(hidden_state,
                                       1 - hparams.layer_prepostprocess_dropout)

          hidden_state = common_layers.layers().Dense(
              hparams.hidden_size)(hidden_state)
          hidden_state = common_layers.layer_postprocess(
              residual_state, hidden_state, hparams)

    # If normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(hidden_state, hparams)


def evolved_transformer_decoder(decoder_input,
                                encoder_output,
                                decoder_self_attention_bias,
                                encoder_decoder_attention_bias,
                                hparams,
                                cache=None,
                                decode_loop_step=None,
                                name="decoder",
                                nonpadding=None,
                                save_weights_to=None,
                                make_image_summary=True,
                                losses=None):
  """Evolved Transformer decoder. See arxiv.org/abs/1901.11117 for more details.

  Args:
    decoder_input: a Tensor.
    encoder_output: a Tensor.
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias()).
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias()).
    hparams: hyperparameters for model.
    cache: Not supported.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string.
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: Not supported.

  Returns:
    Decoder output tensor.
  """
  del cache, losses

  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  with tf.variable_scope(name):
    hidden_state = decoder_input
    layer_cache = None

    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):

        with tf.variable_scope("16_head_self_attention"):
          residual_state = hidden_state
          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

          # Attention with at least 16 heads.
          left_state = common_attention.multihead_attention(
              hidden_state,
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              max(16, hparams.num_heads),
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              decode_loop_step=decode_loop_step,
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"))

        if encoder_output is not None:
          with tf.variable_scope("first_attend_to_encoder"):
            right_state = common_attention.multihead_attention(
                hidden_state,
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                max_relative_position=hparams.max_relative_position,
                heads_share_relative_embedding=(
                    hparams.heads_share_relative_embedding),
                add_relative_to_values=hparams.add_relative_to_values,
                save_weights_to=save_weights_to,
                cache=layer_cache,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length"),
                vars_3d=hparams.get("attention_variables_3d"),
                activation_dtype=hparams.get("activation_dtype", "float32"),
                weight_dtype=hparams.get("weight_dtype", "float32"))

            left_state = tf.nn.dropout(left_state,
                                       1 - hparams.layer_prepostprocess_dropout)
            right_state = tf.nn.dropout(
                right_state, 1 - hparams.layer_prepostprocess_dropout)

            hidden_state = residual_state + left_state + right_state

        else:
          hidden_state = common_layers.layer_postprocess(
              residual_state, left_state, hparams)

        with tf.variable_scope("conv_branches"):
          residual_state = hidden_state
          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

          if nonpadding is not None:
            # Mask padding from conv layers.
            mask = tf.tile(
                tf.expand_dims(nonpadding, 2), [1, 1, hparams.hidden_size])
            hidden_state *= mask

          # Shift inputs so that future tokens cannot be seen.
          left_state = tf.pad(hidden_state, paddings=[[0, 0], [10, 0], [0, 0]])
          left_output_dim = int(hparams.hidden_size * 2)
          separable_conv_11x1 = tf.layers.SeparableConv1D(
              left_output_dim,
              11,
              padding="VALID",
              name="separable_conv11x1",
              activation=tf.nn.relu)
          left_state = separable_conv_11x1.apply(left_state)
          left_state = tf.nn.dropout(left_state,
                                     1 - hparams.layer_prepostprocess_dropout)

          right_state = tf.pad(hidden_state, paddings=[[0, 0], [6, 0], [0, 0]])
          right_output_dim = int(hparams.hidden_size / 2)
          separable_conv_7x1_1 = tf.layers.SeparableConv1D(
              right_output_dim, 7, padding="VALID", name="separable_conv_7x1_1")
          right_state = separable_conv_7x1_1.apply(right_state)
          right_state = tf.nn.dropout(right_state,
                                      1 - hparams.layer_prepostprocess_dropout)
          right_state = tf.pad(
              right_state,
              [[0, 0], [0, 0], [0, left_output_dim - right_output_dim]],
              constant_values=0)

          hidden_state = left_state + right_state

          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)
          if nonpadding is not None:
            # Mask padding from conv layers.
            mask = tf.tile(
                tf.expand_dims(nonpadding, 2), [1, 1, hparams.hidden_size * 2])
            hidden_state *= mask

          hidden_state = tf.pad(hidden_state, paddings=[[0, 0], [6, 0], [0, 0]])
          separable_conv_7x1_2 = tf.layers.SeparableConv1D(
              hparams.hidden_size,
              7,
              padding="VALID",
              name="separable_conv_7x1_2")
          hidden_state = separable_conv_7x1_2.apply(hidden_state)

          hidden_state = common_layers.layer_postprocess(
              residual_state, hidden_state, hparams)

        with tf.variable_scope("self_attention"):
          residual_state = hidden_state
          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

          hidden_state = common_attention.multihead_attention(
              hidden_state,
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              decode_loop_step=decode_loop_step,
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"))
          hidden_state = common_layers.layer_postprocess(
              residual_state, hidden_state, hparams)

        if encoder_output is not None:
          with tf.variable_scope("second_attend_to_encoder"):
            residual_state = hidden_state
            hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

            hidden_state = common_attention.multihead_attention(
                hidden_state,
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                max_relative_position=hparams.max_relative_position,
                heads_share_relative_embedding=(
                    hparams.heads_share_relative_embedding),
                add_relative_to_values=hparams.add_relative_to_values,
                save_weights_to=save_weights_to,
                cache=layer_cache,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length"),
                vars_3d=hparams.get("attention_variables_3d"),
                activation_dtype=hparams.get("activation_dtype", "float32"),
                weight_dtype=hparams.get("weight_dtype", "float32"))
            hidden_state = common_layers.layer_postprocess(
                residual_state, hidden_state, hparams)

        with tf.variable_scope("dense_layers"):
          residual_state = hidden_state
          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

          hidden_state = tf.layers.dense(
              hidden_state,
              int(hparams.hidden_size * 4),
              activation=tf.nn.swish)
          hidden_state = tf.nn.dropout(hidden_state,
                                       1 - hparams.layer_prepostprocess_dropout)

          hidden_state = common_layers.layer_preprocess(hidden_state, hparams)

          hidden_state = tf.layers.dense(hidden_state, hparams.hidden_size)
          hidden_state = common_layers.layer_postprocess(
              residual_state, hidden_state, hparams)

    return common_layers.layer_preprocess(hidden_state, hparams)


# TODO(davidso): Update optimizer, learning rate, and decay to match paper.
def add_evolved_transformer_hparams(hparams):
  """Add Evolved Transformer hparams.

  Note: These are for the Adam optimizer, not the Adafactor optimizer used in
  the paper.

  Args:
    hparams: Current hparams.

  Returns:
    hparams updated with Evolved Transformer values.
  """
  # Evolved Transformer "layers" are twice as deep as Transformer, so roughly
  # halve the number that we use. These numbers are taken from
  # arxiv.org/abs/1901.11117 .
  hparams.num_encoder_layers = 3
  hparams.num_decoder_layers = 4

  # Learning rate and decay scheme that mimics the transformer Adam config,
  # but with cosine decay instead of rsqrt.
  hparams.learning_rate_constant /= hparams.learning_rate_warmup_steps ** 0.5
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*single_cycle_cos_decay*rsqrt_hidden_size")
  # The current infrastructure does not support exposing
  # `train_steps` to the decay functions, and so we are hard coding the decay
  # steps here to match the default number of train steps used in `t2t_trainer`.
  # TODO(davidso): Thread `train_steps` through to decay functions so we do not
  # have to worry about a `learning_rate_decay_steps` mismatch.
  hparams.learning_rate_decay_steps = 250000
  return hparams


@registry.register_hparams
def evolved_transformer_base():
  """Base parameters for Evolved Transformer model."""
  return add_evolved_transformer_hparams(transformer.transformer_base())


@registry.register_hparams
def evolved_transformer_big():
  """Big parameters for Evolved Transformer model on WMT."""
  return add_evolved_transformer_hparams(transformer.transformer_big())


@registry.register_hparams
def evolved_transformer_base_tpu():
  """Base parameters for Evolved Transformer model on TPU."""
  hparams = add_evolved_transformer_hparams(transformer.transformer_tpu())
  hparams.learning_rate_constant = 1 / hparams.learning_rate_warmup_steps ** 0.5
  hparams.learning_rate_schedule = (
      "constant*single_cycle_cos_decay")
  return hparams


@registry.register_hparams
def evolved_transformer_big_tpu():
  """Big parameters for Evolved Transformer model on TPU."""
  hparams = add_evolved_transformer_hparams(transformer.transformer_big_tpu())
  hparams.learning_rate_constant = 1 / hparams.learning_rate_warmup_steps ** 0.5
  hparams.learning_rate_schedule = (
      "constant*single_cycle_cos_decay")
  return hparams
