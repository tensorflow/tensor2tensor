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

"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.models import common_attention
from tensor2tensor.models import common_hparams
from tensor2tensor.models import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class Transformer(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def model_fn_body(self, features, train):
    # Remove dropout if not training
    hparams = copy.copy(self._hparams)
    if not train:
      hparams.attention_dropout = 0.
      hparams.relu_dropout = 0.
      hparams.residual_dropout = 0.
    targets = features["targets"]
    inputs = features.get("inputs")
    target_space = features.get("target_space_id")

    inputs = tf.squeeze(inputs, 2)
    targets = tf.squeeze(targets, 2)

    (encoder_input, encoder_attention_bias, _) = (transformer_prepare_encoder(
        inputs, target_space, hparams))
    (decoder_input, decoder_self_attention_bias) = transformer_prepare_decoder(
        targets, hparams)

    def residual_fn(x, y):
      return common_layers.layer_norm(x + tf.nn.dropout(
          y, 1.0 - hparams.residual_dropout))

    # encoder_input = tf.squeeze(encoder_input, 2)
    # decoder_input = tf.squeeze(decoder_input, 2)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.residual_dropout)
    decoder_input = tf.nn.dropout(decoder_input, 1.0 - hparams.residual_dropout)
    encoder_output = transformer_encoder(encoder_input, residual_fn,
                                         encoder_attention_bias, hparams)

    decoder_output = transformer_decoder(
        decoder_input, encoder_output, residual_fn, decoder_self_attention_bias,
        encoder_attention_bias, hparams)
    decoder_output = tf.expand_dims(decoder_output, 2)

    return decoder_output


def transformer_prepare_encoder(inputs, target_space, hparams):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a Tensor, containing large negative values
      to implement masked attention and possibly baises for diagonal
      alignments
    encoder_padding: a Tensor
  """
  # Flatten inputs.
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  encoder_padding = common_attention.embedding_to_padding(encoder_input)
  encoder_self_attention_bias = common_attention.attention_bias_ignore_padding(
      encoder_padding)
  # Append target_space_id embedding to inputs.
  emb_target_space = common_layers.embedding(
      target_space, 32, ishape_static[-1], name="target_space_embedding")
  emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
  encoder_input += emb_target_space
  if hparams.pos == "timing":
    encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  return (encoder_input, encoder_self_attention_bias, encoder_padding)


def transformer_prepare_decoder(targets, hparams):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a Tensor, containing large negative values
    to implement masked attention and possibly baises for diagonal alignments
  """
  decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
  decoder_input = common_layers.shift_left_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias)


def transformer_encoder(encoder_input,
                        residual_fn,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder"):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    residual_fn: a function from (layer_input, layer_output) -> combined_output
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """
  x = encoder_input
  # Summaries don't work in multi-problem setting yet.
  summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
  with tf.variable_scope(name):
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = residual_fn(
            x,
            common_attention.multihead_attention(
                x,
                None,
                encoder_self_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                summaries=summaries,
                name="encoder_self_attention"))
        x = residual_fn(x,
                        common_layers.conv_hidden_relu(
                            x,
                            hparams.filter_size,
                            hparams.hidden_size,
                            dropout=hparams.relu_dropout))
  return x


def transformer_decoder(decoder_input,
                        encoder_output,
                        residual_fn,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        name="decoder"):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    residual_fn: a function from (layer_input, layer_output) -> combined_output
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """
  x = decoder_input
  # Summaries don't work in multi-problem setting yet.
  summaries = "problems" not in hparams.values() or len(hparams.problems) == 1
  with tf.variable_scope(name):
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        x = residual_fn(
            x,
            common_attention.multihead_attention(
                x,
                None,
                decoder_self_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                summaries=summaries,
                name="decoder_self_attention"))
        x = residual_fn(
            x,
            common_attention.multihead_attention(
                x,
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                summaries=summaries,
                name="encdec_attention"))
        x = residual_fn(x,
                        common_layers.conv_hidden_relu(
                            x,
                            hparams.filter_size,
                            hparams.hidden_size,
                            dropout=hparams.relu_dropout))
  return x


@registry.register_hparams
def transformer_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.batch_size = 4096
  hparams.max_length = 256
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = int(True)

  hparams.add_hparam("filter_size", 2048)  # Add new ones like this.
  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("residual_dropout", 0.1)
  hparams.add_hparam("nbr_decoder_problems", 1)
  return hparams


@registry.register_hparams
def transformer_single_gpu():
  hparams = transformer_base()
  hparams.batch_size = 8192
  hparams.learning_rate_warmup_steps = 16000
  hparams.batching_mantissa_bits = 2
  return hparams


@registry.register_hparams
def transformer_tiny():
  hparams = transformer_base()
  hparams.hidden_size = 64
  hparams.filter_size = 128
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_l2():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  return hparams


@registry.register_hparams
def transformer_l4():
  hparams = transformer_base()
  hparams.num_hidden_layers = 4
  return hparams


@registry.register_hparams
def transformer_l8():
  hparams = transformer_base()
  hparams.num_hidden_layers = 8
  return hparams


@registry.register_hparams
def transformer_h1():
  hparams = transformer_base()
  hparams.num_heads = 1
  return hparams


@registry.register_hparams
def transformer_h4():
  hparams = transformer_base()
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_h16():
  hparams = transformer_base()
  hparams.num_heads = 16
  return hparams


@registry.register_hparams
def transformer_h32():
  hparams = transformer_base()
  hparams.num_heads = 32
  return hparams


@registry.register_hparams
def transformer_k128():
  hparams = transformer_base()
  hparams.attention_key_channels = 128
  return hparams


@registry.register_hparams
def transformer_k256():
  hparams = transformer_base()
  hparams.attention_key_channels = 256
  return hparams


@registry.register_hparams
def transformer_ff1024():
  hparams = transformer_base()
  hparams.filter_size = 1024
  return hparams


@registry.register_hparams
def transformer_ff4096():
  hparams = transformer_base()
  hparams.filter_size = 4096
  return hparams


@registry.register_hparams
def transformer_dr0():
  hparams = transformer_base()
  hparams.residual_dropout = 0.0
  return hparams


@registry.register_hparams
def transformer_dr2():
  hparams = transformer_base()
  hparams.residual_dropout = 0.2
  return hparams


@registry.register_hparams
def transformer_ls0():
  hparams = transformer_base()
  hparams.label_smoothing = 0.0
  return hparams


@registry.register_hparams
def transformer_ls2():
  hparams = transformer_base()
  hparams.label_smoothing = 0.2
  return hparams


@registry.register_hparams
def transformer_hs256():
  hparams = transformer_base()
  hparams.hidden_size = 256
  return hparams


@registry.register_hparams
def transformer_hs1024():
  hparams = transformer_base()
  hparams.hidden_size = 1024
  return hparams


@registry.register_hparams
def transformer_big_dr1():
  hparams = transformer_base()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.residual_dropout = 0.1
  hparams.batching_mantissa_bits = 2
  return hparams


@registry.register_hparams
def transformer_big_enfr():
  hparams = transformer_big_dr1()
  hparams.shared_embedding_and_softmax_weights = int(False)
  hparams.filter_size = 8192
  hparams.residual_dropout = 0.1
  return hparams


@registry.register_hparams
def transformer_big_dr2():
  hparams = transformer_big_dr1()
  hparams.residual_dropout = 0.2
  return hparams


@registry.register_hparams
def transformer_big_dr3():
  hparams = transformer_big_dr1()
  hparams.residual_dropout = 0.3
  return hparams


@registry.register_hparams
def transformer_big_single_gpu():
  hparams = transformer_big_dr1()
  hparams.learning_rate_warmup_steps = 16000
  hparams.optimizer_adam_beta2 = 0.998
  hparams.batching_mantissa_bits = 3
  return hparams


@registry.register_hparams
def transformer_parsing_base_dr6():
  """hparams for parsing on wsj only."""
  hparams = transformer_base()
  hparams.attention_dropout = 0.2
  hparams.residual_dropout = 0.2
  hparams.max_length = 512
  hparams.learning_rate_warmup_steps = 16000
  hparams.hidden_size = 1024
  hparams.learning_rate = 0.5
  hparams.shared_embedding_and_softmax_weights = int(False)
  return hparams


@registry.register_hparams
def transformer_parsing_big():
  """HParams for parsing on wsj semi-supervised."""
  hparams = transformer_big_dr1()
  hparams.max_length = 512
  hparams.shared_source_target_embedding = int(False)
  hparams.learning_rate_warmup_steps = 4000
  hparams.batch_size = 2048
  hparams.learning_rate = 0.5
  return hparams


@registry.register_ranged_hparams("transformer_big_single_gpu")
def transformer_range1(rhp):
  """Small range of hyperparameters."""
  hparams = transformer_big_single_gpu()
  common_hparams.fill_ranged_hparams_from_hparams(hparams, rhp)

  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_float("initializer_gain", 0.5, 2.0)
  rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
  rhp.set_float("weight_decay", 0.0, 2.0)
