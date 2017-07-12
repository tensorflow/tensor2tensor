# Copyright 2017 Google Inc.
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

from tensor2tensor.models import common_attention
from tensor2tensor.models import common_layers
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

    (encoder_input, encoder_attention_bias,
     _) = transformer.transformer_prepare_encoder(inputs, target_space, hparams)
    (decoder_input,
     decoder_self_attention_bias) = transformer.transformer_prepare_decoder(
         targets, hparams)

    # We need masks of the form batch size x input sequences
    # Biases seem to be of the form batch_size x 1 x input sequences x vec dim
    #  Squeeze out dim one, and get the first element of each vector.
    encoder_mask = tf.squeeze(encoder_attention_bias, [1])[:, :, 0]
    decoder_mask = tf.squeeze(decoder_self_attention_bias, [1])[:, :, 0]

    def residual_fn(x, y):
      return common_layers.layer_norm(x + tf.nn.dropout(
          y, 1.0 - hparams.residual_dropout))

    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.residual_dropout)
    decoder_input = tf.nn.dropout(decoder_input, 1.0 - hparams.residual_dropout)
    encoder_output = alt_transformer_encoder(
        encoder_input, residual_fn, encoder_mask, hparams)

    decoder_output = alt_transformer_decoder(
        decoder_input, encoder_output, residual_fn, decoder_mask,
        encoder_attention_bias, hparams)

    decoder_output = tf.expand_dims(decoder_output, 2)

    return decoder_output


def composite_layer(inputs, mask, hparams):
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
            dropout=0.0)

  # Transforms elements to get a context, and then uses this in a final layer.
  elif hparams.composite_layer_type == "reembedding":
    # Transform elements n times and then pool.
    for layer in xrange(hparams.layers_per_layer):
      with tf.variable_scope(".%d" % layer):
        x = common_layers.linear_set_layer(
            hparams.hidden_size,
            x,
            dropout=0.0)
    context = common_layers.global_pool_1d(x, mask=mask)

    # Final layer.
    x = common_layers.linear_set_layer(
        hparams.hidden_size,
        x,
        context=context,
        dropout=0.0)

  return x


def alt_transformer_encoder(encoder_input,
                            residual_fn,
                            mask,
                            hparams,
                            name="encoder"):
  """Alternative encoder."""
  x = encoder_input

  with tf.variable_scope(name):
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = residual_fn(x, composite_layer(x, mask, hparams))

  return x


def alt_transformer_decoder(decoder_input,
                            encoder_output,
                            residual_fn,
                            mask,
                            encoder_decoder_attention_bias,
                            hparams,
                            name="decoder"):
  """Alternative decoder."""
  x = decoder_input

  # Summaries don't work in multi-problem setting yet.
  summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
  with tf.variable_scope(name):
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
            summaries=summaries,
            name="encdec_attention")

        x_ = residual_fn(x_, composite_layer(x_, mask, hparams))
        x = residual_fn(x, x_)

  return x


@registry.register_hparams
def transformer_alt():
  """Set of hyperparameters."""
  hparams = transformer.transformer_base()
  hparams.batch_size = 64
  hparams.add_hparam("layers_per_layer", 4)
  hparams.add_hparam("composite_layer_type", "reembedding")
  return hparams
