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

"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import transformer_layers
from tensor2tensor.models import transformer
from tensor2tensor.v2 import keras_utils
import tensorflow as tf
import gin.tf


@gin.configurable(whitelist=["hidden_size", "filter_size"])
class Transformer(tf.keras.Model):
  """Transformer."""

  def __init__(self, features_info=None, input_names=None, target_names=None,
               hidden_size=512, filter_size=2048):
    super(Transformer, self).__init__()
    # TODO(lukaszkaiser): gin'ify and split into encoder/decoder classes.
    self._has_input = True if input_names else False
    self._input_name = input_names[0]
    self._target_name = target_names[0]
    try:
      target_vocab_size = features_info[self._target_name].num_classes
    except AttributeError:
      target_vocab_size = features_info[self._target_name].encoder.vocab_size
    hparams = transformer.transformer_base()
    hparams.hidden_size = hidden_size
    hparams.filter_size = filter_size

    # Now the model.
    self._embedding = tf.keras.layers.Embedding(
        target_vocab_size, hidden_size, mask_zero=True)
    def transformer_encoder(inputs, features):
      return transformer.transformer_encode(
          transformer_layers.transformer_encoder, inputs, None,
          hparams, features=features)

    def transformer_prepare_decoder(targets, features):
      return transformer.transformer_prepare_decoder(targets, hparams, features)

    def transformer_decoder(decoder_input, encoder_output,
                            encoder_decoder_attention_bias,
                            decoder_self_attention_bias,
                            features):
      return transformer.transformer_decode(
          transformer.transformer_decoder,
          decoder_input,
          encoder_output,
          encoder_decoder_attention_bias,
          decoder_self_attention_bias,
          hparams,
          nonpadding=transformer.features_to_nonpadding(features, "targets"))

    if self._has_input:
      self._encoder = keras_utils.FunctionLayer(transformer_encoder)
    self._prepare_decoder = keras_utils.FunctionLayer(
        transformer_prepare_decoder)
    self._decoder = keras_utils.FunctionLayer(transformer_decoder)
    self._logits = tf.keras.layers.Dense(
        target_vocab_size, activation=None)

  def call(self, features, training=False):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
          "targets": Target decoder outputs. [batch_size, decoder_length, 1,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.
      training: Whether we are training or not.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    if self._has_input:
      inputs = features[self._input_name]
      inputs = tf.expand_dims(self._embedding(inputs), 2)
      encoder_output, encoder_decoder_attention_bias = self._encoder(
          inputs, features)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    targets = features[self._target_name]
    targets = self._embedding(targets)
    decoder_input, decoder_self_attention_bias = self._prepare_decoder(
        targets, features)
    decoder_output = self._decoder(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        features)

    return self._logits(tf.squeeze(decoder_output, axis=2))


def transformer_base_single_gpu():
  """Single-gpu set of parameters for Transformer."""
  gin.bind_parameter("T2TLearningRateSchedule.warmup_steps", 16000)
  gin.bind_parameter("preprocess_fn.max_target_length", 256)
  gin.bind_parameter("batch_fn.eval_batch_size", 8)
  return Transformer
