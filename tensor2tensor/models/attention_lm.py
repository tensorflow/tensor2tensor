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

"""Self-attention based language model.

Like transformer.py, but no encoder

decoder: [Self-Attention, Feed-forward] x n

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class AttentionLM(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def body(self, features):
    # Remove dropout if not training
    hparams = self._hparams
    targets = features["targets"]
    targets = tf.squeeze(targets, 2)

    (decoder_input, decoder_self_attention_bias) = attention_lm_prepare_decoder(
        targets, hparams)

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)
    decoder_output = attention_lm_decoder(decoder_input,
                                          decoder_self_attention_bias, hparams)
    decoder_output = tf.expand_dims(decoder_output, 2)

    return decoder_output


def attention_lm_prepare_decoder(targets, hparams):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a Tensor, containing large negative values
    to implement masked attention and possibly baises for diagonal alignments
  """
  if hparams.prepend_mode == "prepend_inputs_full_attention":
    decoder_self_attention_bias = (
        common_attention.attention_bias_prepended(
            common_attention.embedding_to_padding(targets)))
  else:
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(
            common_layers.shape_list(targets)[1]))
  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias)


def attention_lm_decoder(decoder_input,
                         decoder_self_attention_bias,
                         hparams,
                         name="decoder"):
  """A stack of attention_lm layers.

  Args:
    decoder_input: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """
  x = decoder_input
  with tf.variable_scope(name):
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(
                  x, hparams), None, decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size, hparams.num_heads, hparams.attention_dropout)
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = common_layers.conv_hidden_relu(
              common_layers.layer_preprocess(x, hparams),
              hparams.filter_size,
              hparams.hidden_size,
              dropout=hparams.relu_dropout)
          x = common_layers.layer_postprocess(x, y, hparams)
    return common_layers.layer_preprocess(x, hparams)


@registry.register_hparams
def attention_lm_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 1024
  hparams.batch_size = 8192
  hparams.max_length = 256
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 2000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.label_smoothing = 0.0
  hparams.shared_embedding_and_softmax_weights = False

  hparams.add_hparam("filter_size", 4096)  # Add new ones like this.
  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("encoder_full_attention", False)
  return hparams


@registry.register_hparams
def attention_lm_small():
  """Cheap model.

  on lm1b_32k:
     45M params
     2 steps/sec on  [GeForce GTX TITAN X]

  Returns:
    an hparams object.
  """
  hparams = attention_lm_base()
  hparams.num_hidden_layers = 4
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.layer_prepostprocess_dropout = 0.5
  return hparams


@registry.register_hparams
def attention_lm_translation():
  """Version to use for seq2seq."""
  hparams = attention_lm_base()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.learning_rate = 0.4
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 512
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  return hparams


@registry.register_hparams
def attention_lm_translation_l12():
  """Version to use for seq2seq."""
  hparams = attention_lm_translation()
  hparams.batch_size = 4096
  hparams.num_hidden_layers = 12
  return hparams


@registry.register_hparams
def attention_lm_translation_full_attention():
  """Version to use for seq2seq."""
  hparams = attention_lm_translation()
  hparams.prepend_mode = "prepend_inputs_full_attention"
  return hparams
