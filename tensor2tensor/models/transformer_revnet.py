# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Reversible Residual Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import rev_block
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class TransformerRevnet(transformer.Transformer):
  """Reversible Residual Transformer.

  Layers are reversible and are recomputed on the backward pass.

  y1 = x1 + f(x2)
  y2 = x2 + g(y1)

  f: Attention
  g: Feed-forward
  """

  def body(self, features):
    hparams = self._hparams
    targets = features["targets"]
    inputs = features["inputs"]
    target_space = features["target_space_id"]

    inputs = common_layers.flatten4d3d(inputs)
    targets = common_layers.flatten4d3d(targets)

    (encoder_input, encoder_self_attention_bias,
     encoder_decoder_attention_bias) = (transformer.transformer_prepare_encoder(
         inputs, target_space, hparams))
    (decoder_input,
     decoder_self_attention_bias) = transformer.transformer_prepare_decoder(
         targets, hparams)

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output = transformer_revnet_encoder(
        encoder_input, encoder_self_attention_bias, hparams)

    decoder_output = transformer_revnet_decoder(
        decoder_input, encoder_output, decoder_self_attention_bias,
        encoder_decoder_attention_bias, hparams)
    decoder_output = tf.expand_dims(decoder_output, 2)

    return decoder_output


def transformer_revnet_encoder(encoder_input,
                               encoder_self_attention_bias,
                               hparams,
                               name="encoder"):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """

  def f(x, side_input):
    """f(x) for reversible layer, self-attention layer."""
    encoder_self_attention_bias = side_input[0]

    old_hid_size = hparams.hidden_size
    hparams.hidden_size = old_hid_size // 2

    with tf.variable_scope("self_attention"):
      y = common_attention.multihead_attention(
          common_layers.layer_preprocess(
              x, hparams), None, encoder_self_attention_bias,
          hparams.attention_key_channels or hparams.hidden_size,
          hparams.attention_value_channels or hparams.hidden_size,
          hparams.hidden_size, hparams.num_heads, hparams.attention_dropout)
      y = common_layers.layer_postprocess(x, y, hparams)
    hparams.hidden_size = old_hid_size
    return y

  def g(x):
    """g(x) for reversible layer, feed-forward layer."""
    old_hid_size = hparams.hidden_size
    hparams.hidden_size = old_hid_size // 2

    with tf.variable_scope("ffn"):
      y = transformer.transformer_ffn_layer(
          common_layers.layer_preprocess(x, hparams), hparams)
      y = common_layers.layer_postprocess(x, y, hparams)
    hparams.hidden_size = old_hid_size
    return y

  x1, x2 = tf.split(encoder_input, 2, axis=-1)

  with tf.variable_scope(name):
    y1, y2 = rev_block.rev_block(
        x1,
        x2,
        f,
        g,
        num_layers=hparams.num_hidden_layers,
        f_side_input=[encoder_self_attention_bias],
        is_training=hparams.mode == tf.estimator.ModeKeys.TRAIN)
    y = tf.concat([y1, y2], axis=-1)

  return common_layers.layer_preprocess(y, hparams)


def transformer_revnet_decoder(decoder_input,
                               encoder_output,
                               decoder_self_attention_bias,
                               encoder_decoder_attention_bias,
                               hparams,
                               name="decoder"):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """

  def f(x, side_input):
    """f(x) for reversible layer, self-attention and enc-dec attention."""
    decoder_self_attention_bias = side_input[0]
    encoder_decoder_attention_bias = side_input[1]
    encoder_output = side_input[2]

    old_hid_size = hparams.hidden_size
    hparams.hidden_size = old_hid_size // 2

    with tf.variable_scope("self_attention"):
      y = common_attention.multihead_attention(
          common_layers.layer_preprocess(
              x, hparams), None, decoder_self_attention_bias,
          hparams.attention_key_channels or hparams.hidden_size,
          hparams.attention_value_channels or hparams.hidden_size,
          hparams.hidden_size, hparams.num_heads, hparams.attention_dropout)
      y = common_layers.layer_postprocess(x, y, hparams)
      if encoder_output is not None:
        with tf.variable_scope("encdec_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(
                  x, hparams), encoder_output, encoder_decoder_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size, hparams.num_heads, hparams.attention_dropout)
          y = common_layers.layer_postprocess(x, y, hparams)
    hparams.hidden_size = old_hid_size
    return y

  def g(x):
    """g(x) for reversible layer, feed-forward layer."""
    old_hid_size = hparams.hidden_size
    hparams.hidden_size = old_hid_size // 2
    with tf.variable_scope("ffn"):
      y = transformer.transformer_ffn_layer(
          common_layers.layer_preprocess(x, hparams), hparams)
      y = common_layers.layer_postprocess(x, y, hparams)
    hparams.hidden_size = old_hid_size
    return y

  x1, x2 = tf.split(decoder_input, 2, axis=-1)

  with tf.variable_scope(name):
    y1, y2 = rev_block.rev_block(
        x1,
        x2,
        f,
        g,
        num_layers=hparams.num_hidden_layers,
        f_side_input=[
            decoder_self_attention_bias, encoder_decoder_attention_bias,
            encoder_output
        ],
        is_training=hparams.mode == tf.estimator.ModeKeys.TRAIN)
    y = tf.concat([y1, y2], axis=-1)
    return common_layers.layer_preprocess(y, hparams)


@registry.register_hparams
def transformer_revnet_base():
  """Base hparams for TransformerRevnet."""
  hparams = transformer.transformer_big()

  # Use settings from transformer_n_da
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.learning_rate = 0.4

  return hparams


@registry.register_hparams
def transformer_revnet_big():
  """Base hparams for TransformerRevnet."""
  hparams = transformer_revnet_base()

  # The TransformerRevnet uses significantly less memory than the Transformer.
  # Increase batch size and model size.
  hparams.batch_size *= 2
  hparams.hidden_size *= 2
  hparams.num_heads *= 2
  hparams.num_hidden_layers += 1
  return hparams
