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

"""Alternative transformer network.

Using different layer types to demonstrate alternatives to self attention.

Code is mostly copied from original Transformer source.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class TransformerAlt(t2t_model.T2TModel):

  def model_fn_body(self, features):
    hparams = self._hparams
    targets = features["targets"]
    inputs = features.get("inputs")
    target_space = features.get("target_space_id")

    inputs = common_layers.flatten4d3d(inputs)
    targets = common_layers.flatten4d3d(targets)

    (encoder_input,
     encoder_attention_bias, _) = (transformer.transformer_prepare_encoder(
         inputs, target_space, hparams))
    (decoder_input, _) = (transformer.transformer_prepare_decoder(
        targets, hparams))

    encoder_mask = bias_to_mask(encoder_attention_bias)

    def residual_fn(x, y):
      return common_layers.layer_norm(x + tf.nn.dropout(
          y, 1.0 - hparams.residual_dropout))

    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.residual_dropout)
    decoder_input = tf.nn.dropout(decoder_input, 1.0 - hparams.residual_dropout)

    encoder_output = alt_transformer_encoder(encoder_input, residual_fn,
                                             encoder_mask, hparams)

    decoder_output = alt_transformer_decoder(decoder_input, encoder_output,
                                             residual_fn,
                                             encoder_attention_bias, hparams)

    decoder_output = tf.expand_dims(decoder_output, 2)

    return decoder_output


def composite_layer(inputs, mask, hparams, for_output=False):
  """Composite layer."""
  x = inputs

  # Applies ravanbakhsh on top of each other.
  if hparams.composite_layer_type == "ravanbakhsh":
    for layer in xrange(hparams.layers_per_layer):
      with tf.variable_scope(".%d" % layer):
        x = common_layers.ravanbakhsh_set_layer(
            hparams.hidden_size,
            x,
            mask=mask,
            sequential=for_output,
            dropout=hparams.relu_dropout)

  # Transforms elements to get a context, and then uses this in a final layer.
  elif hparams.composite_layer_type == "reembedding":
    # Transform elements n times and then pool.
    for layer in xrange(hparams.layers_per_layer):
      with tf.variable_scope("sub_layer_%d" % layer):
        x = common_layers.linear_set_layer(
            hparams.hidden_size, x, dropout=hparams.relu_dropout)
        if for_output:
          context = common_layers.running_global_pool_1d(x)
        else:
          context = common_layers.global_pool_1d(x, mask=mask)
    # Final layer.
    x = common_layers.linear_set_layer(
        hparams.hidden_size, x, context=context, dropout=hparams.relu_dropout)
  return x


def alt_transformer_encoder(encoder_input,
                            residual_fn,
                            mask,
                            hparams,
                            name="encoder"):
  """Alternative encoder."""
  x = encoder_input
  with tf.variable_scope(name):
    x = encoder_input
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = residual_fn(x, composite_layer(x, mask, hparams))
  return x


def alt_transformer_decoder(decoder_input,
                            encoder_output,
                            residual_fn,
                            encoder_decoder_attention_bias,
                            hparams,
                            name="decoder"):
  """Alternative decoder."""
  with tf.variable_scope(name):
    x = decoder_input
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        x_ = common_attention.multihead_attention(
            x,
            encoder_output,
            encoder_decoder_attention_bias,
            hparams.attention_key_channels or hparams.hidden_size,
            hparams.attention_value_channels or hparams.hidden_size,
            hparams.hidden_size,
            hparams.num_heads,
            hparams.attention_dropout,
            name="encdec_attention")

        x_ = residual_fn(x_, composite_layer(
            x_, None, hparams, for_output=True))
        x = residual_fn(x, x_)
  return x


def bias_to_mask(bias):
  # We need masks of the form batch size x input sequences
  # Biases are of the form batch_size x num_heads x input sequences x
  #  output sequences. Squeeze out dim one, and get the first element of
  #  each vector.
  bias = tf.squeeze(bias, [1])[:, :, 0]
  bias = -tf.clip_by_value(bias, -1.0, 1.0)
  mask = 1 - bias
  return mask


@registry.register_hparams
def transformer_alt():
  """Set of hyperparameters."""
  hparams = transformer.transformer_base()
  hparams.batch_size = 2048
  hparams.num_hidden_layers = 10
  hparams.add_hparam("layers_per_layer", 4)
  # Composite layer: ravanbakhsh or reembedding.
  hparams.add_hparam("composite_layer_type", "ravanbakhsh")
  return hparams
