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
"""Transformers with depthwise recurrency (go/r-transformer).


A high-level explanation on the idea and the architecture:

The vanilla Transformer model has no recurrence and struggles with some tasks
that a fully recurrent model can easily solve. Instead of incorporating
recurrence in time (which has a dependency on sequence length T),
we apply recurrence in depth (which we can set to some fixed length D << T),
and apply self-attention instead of sequential processing to enable the model
to incorporate long-range dependencies.

Structure of the code is explained in r_transformer_util.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models.research import r_transformer_util
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class RTransformer(transformer.Transformer):
  """R-Transformer: Depth-wise recurrent transoformer model."""

  def encode(self, inputs, target_space, hparams, features=None):
    """Encode r-transformer inputs.

    It is similar to "transformer.encode", but it uses
    "r_transformer_util.r_transformer_encoder" instead of
    "transformer.transformer_encoder".

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """

    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer.transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    (encoder_output,
     encoder_extra_output) = r_transformer_util.r_transformer_encoder(
         encoder_input,
         self_attention_bias,
         hparams,
         nonpadding=transformer.features_to_nonpadding(features, "inputs"),
         save_weights_to=self.attention_weights)

    return encoder_output, encoder_decoder_attention_bias, encoder_extra_output

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             nonpadding=None):
    """Decode R-Transformer outputs from encoder representation.

    It is similar to "transformer.decode", but it uses
    "r_transformer_util.r_transformer_decoder" instead of
    "transformer.transformer_decoder".

    Args:
      decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
        hidden_dim]
      encoder_output: Encoder representation. [batch_size, input_length,
        hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
        attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
        self-attention. [batch_size, decoder_length]
      hparams: hyperparmeters for model.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]

    Returns:
       Tuple of:
         Final decoder representation. [batch_size, decoder_length,
            hidden_dim]
         encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)

    """

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    # No caching in r-transformers!
    decoder_output, dec_extra_output = r_transformer_util.r_transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        nonpadding=nonpadding,
        save_weights_to=self.attention_weights)

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2), dec_extra_output

  def body(self, features):
    """R-Transformer main model_fn.


    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    if self.has_input:
      inputs = features["inputs"]
      target_space = features["target_space_id"]
      (encoder_output, encoder_decoder_attention_bias,
       enc_extra_output) = self.encode(
           inputs, target_space, hparams, features=features)
    else:
      (encoder_output, encoder_decoder_attention_bias,
       enc_extra_output) = (None, None, (None, None))

    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)

    (decoder_input,
     decoder_self_attention_bias) = transformer.transformer_prepare_decoder(
         targets, hparams, features=features)

    decoder_output, dec_extra_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=transformer.features_to_nonpadding(features, "targets"))

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    if hparams.recurrence_type == "act" and hparams.act_loss_weight != 0:
      if self.has_input:
        enc_ponder_times, enc_remainders = enc_extra_output
        enc_act_loss = (
            hparams.act_loss_weight *
            tf.reduce_mean(enc_ponder_times + enc_remainders))
      else:
        enc_act_loss = 0.0

      (dec_ponder_times, dec_remainders) = dec_extra_output
      dec_act_loss = (
          hparams.act_loss_weight *
          tf.reduce_mean(dec_ponder_times + dec_remainders))
      act_loss = enc_act_loss + dec_act_loss
      tf.summary.scalar("act_loss", act_loss)
      return decoder_output, {"act_loss": act_loss}

    return decoder_output


@registry.register_model
class RTransformerEncoder(transformer.Transformer):
  """R-Transformer Encoder: Depth-wise recurrent transoformer encoder-only."""

  def encode(self, inputs, target_space, hparams, features=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
    inputs = common_layers.flatten4d3d(inputs)

    (encoder_input, self_attention_bias, _) = (
        transformer.transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    (encoder_output,
     encoder_extra_output) = r_transformer_util.r_transformer_encoder(
         encoder_input,
         self_attention_bias,
         hparams,
         nonpadding=transformer.features_to_nonpadding(features, "inputs"),
         save_weights_to=self.attention_weights)

    return encoder_output, encoder_extra_output

  def body(self, features):
    """R-Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    assert self.has_input, ("r_transformer_encoder is applicable on problems"
                            "with inputs")

    inputs = features["inputs"]
    target_space = features["target_space_id"]
    encoder_output, enc_extra_output = self.encode(
        inputs, target_space, hparams, features=features)

    encoder_output = tf.expand_dims(encoder_output, 2)

    if hparams.recurrence_type == "act" and hparams.act_loss_weight != 0:
      ponder_times, remainders = enc_extra_output
      act_loss = hparams.act_loss_weight * tf.reduce_mean(ponder_times +
                                                          remainders)
      tf.summary.scalar("act_loss", act_loss)

      return encoder_output, {"act_loss": act_loss}
    return encoder_output


def update_hparams_for_r_transformer(hparams):
  """Adds deault hparams for all of the variants of the R-transformer.

  Args:
    hparams: default hparams (usually one of the standard hparams from
      transformer model (like "transformer_base")

  Returns:
    hparams with default values for R-Transformers hyper-parameters

  """
  # Type of recurrency:
  # None(no-recurrency) basic, highway, skip, dwa, act, rnn, gru, lstm.
  hparams.add_hparam("recurrence_type", "basic")

  # Number of steps (which is equivalent to num layer in transformer).
  hparams.add_hparam("num_rec_steps", hparams.num_hidden_layers)

  # Default ffn layer is separable convolution.
  hparams.add_hparam("transformer_ffn_type", "sep")

  # Transform bias (in models with highway or skip connection).
  hparams.add_hparam("transform_bias_init", -1.0)
  hparams.add_hparam("couple_carry_transform_gates", True)

  # Depth-wise attention (grid-transformer!) hparams:
  # Adds depth embedding, if true.
  hparams.add_hparam("depth_embedding", True)
  # Learns attention weights for elements (instead of positions), if true.
  hparams.add_hparam("dwa_elements", True)

  # Type of ffn_layer used for gate in skip, highway, etc.
  # "dense" or "dense_dropconnect".
  # With dense_relu_dense, the bias/kernel initializations will not be applied.
  hparams.add_hparam("gate_ffn_layer", "dense")

  # Config for all rnn style recurrencies (rnn, lstm, gru):
  # Input of the gate functions: i:input/s:state/t:transformed state.
  # or any combination: e.g. is, ts, ist, etc.
  hparams.add_hparam("gates_inputs", "i")

  # LSTEM forget bias.
  hparams.add_hparam("lstm_forget_bias", 1.0)

  # How to combine state and input in each step:
  # "mh_attention_ffn_add" or "add_mh_attention_ffn" or "dense_mh_attention"
  # or "mh_attention_dense".
  # Interpretation for e.g. "mh_attention_ffn_add":
  # Apply transformer attention then transformer ffn, then add.
  hparams.add_hparam("inputs_states_combination", "mh_attention_ffn_add")

  # Config for gru_style recurrency:
  # What to transform in gru: state/output/candidate/combination of them.
  hparams.add_hparam("gru_transformation", ["state_transformation"])

  # Config for lstm_style Recurrency:
  # What to transform in lstm: state/modulated_input/memory.
  hparams.add_hparam("lstm_transformation", ["state_transformation"])
  # Uses the mememory at the last step as the final touput, if true.
  hparams.add_hparam("use_memory_as_final_state", False)

  # Type of act: basic/accumulated/global (instead of position-wise!)/random.
  hparams.add_hparam("act_type", "basic")
  # Max number of steps (forces halting at this step).
  hparams.add_hparam("act_max_steps", 2 * hparams.num_hidden_layers)
  hparams.add_hparam("act_halting_bias_init", 1.0)
  hparams.add_hparam("act_epsilon", 0.01)
  hparams.add_hparam("act_loss_weight", 0.01)

  return hparams


@registry.register_hparams
def r_transformer_big():
  hparams = transformer.transformer_big()
  hparams = update_hparams_for_r_transformer(hparams)
  return hparams


@registry.register_hparams
def r_transformer_base():
  hparams = transformer.transformer_base()
  hparams = update_hparams_for_r_transformer(hparams)
  return hparams


@registry.register_hparams
def r_transformer_tiny():
  hparams = transformer.transformer_tiny()
  hparams = update_hparams_for_r_transformer(hparams)
  hparams.num_rec_steps = 8
  return hparams


@registry.register_hparams
def transformer_teeny():
  hparams = transformer.transformer_base()
  hparams.num_rec_steps = 2
  hparams.hidden_size = 128
  hparams.filter_size = 128
  hparams.num_heads = 2
  return hparams


@registry.register_hparams
def r_transformer_teeny():
  hparams = transformer_teeny()
  hparams = update_hparams_for_r_transformer(hparams)
  hparams.num_rec_steps = 10
  return hparams


@registry.register_hparams
def r_transformer_base_dropconnect():
  hparams = r_transformer_base()
  hparams.gate_ffn_layer = "dense_dropconnect"
  hparams.add_hparam("dropconnect_dropout", 0.5)
  return hparams


@registry.register_hparams
def r_transformer_act_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  return hparams


@registry.register_hparams
def r_transformer_act_tiny():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "act"
  return hparams


@registry.register_hparams
def r_transformer_act_big():
  hparams = r_transformer_big()
  hparams.recurrence_type = "act"
  return hparams


@registry.register_hparams
def r_transformer_act_random_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.act_type = "random"
  return hparams


@registry.register_hparams
def r_transformer_act_accumulated_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.act_type = "accumulated"
  return hparams


@registry.register_hparams
def r_transformer_act_global_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.act_type = "global"
  return hparams


@registry.register_hparams
def r_transformer_act_accumulated_tiny():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.act_type = "accumulated"
  return hparams


@registry.register_hparams
def r_transformer_act_global_tiny():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.act_type = "global"
  return hparams


@registry.register_hparams
def r_transformer_act_random_tiny():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.act_type = "random"
  return hparams


@registry.register_hparams
def r_transformer_act_base_sb():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.batch_size = 2048
  return hparams


@registry.register_hparams
def r_transformer_act_large():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.hidden_size = 1024
  hparams.batch_size = 2048
  hparams.filter_size = 2048
  return hparams


@registry.register_hparams
def r_transformer_act_tall():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.num_hidden_layers = 16
  hparams.batch_size = 1024
  hparams.act_max_steps = 24
  return hparams


@registry.register_hparams
def r_transformer_act_tall_actlossw0():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.num_hidden_layers = 16
  hparams.batch_size = 1024
  hparams.act_max_steps = 24
  return hparams


@registry.register_hparams
def r_transformer_act_tall_actlossw001():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.num_hidden_layers = 16
  hparams.batch_size = 1024
  hparams.act_max_steps = 24
  return hparams


@registry.register_hparams
def r_transformer_act_base_d03():
  hparams = r_transformer_base()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.attention_dropout = 0.3
  hparams.relu_dropout = 0.3
  return hparams


@registry.register_hparams
def r_transformer_act_big_d03():
  hparams = r_transformer_big()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.attention_dropout = 0.3
  hparams.relu_dropout = 0.3
  return hparams


@registry.register_hparams
def r_transformer_act_tiny_d02():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.attention_dropout = 0.2
  hparams.relu_dropout = 0.2
  return hparams


@registry.register_hparams
def r_transformer_act_tiny_d02_sb():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.attention_dropout = 0.2
  hparams.relu_dropout = 0.2
  hparams.batch_size = 2048
  return hparams


@registry.register_hparams
def r_transformer_act_tiny_sb():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.batch_size = 2048
  return hparams


@registry.register_hparams
def r_transformer_act_tiny_d05():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.5
  hparams.attention_dropout = 0.5
  hparams.relu_dropout = 0.5
  return hparams


@registry.register_hparams
def r_transformer_base_sb():
  hparams = r_transformer_base()
  hparams.batch_size = 2048
  return hparams


@registry.register_hparams
def r_transformer_skip_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "skip"
  return hparams


@registry.register_hparams
def r_transformer_skip_tiny():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "skip"
  return hparams


@registry.register_hparams
def r_transformer_highway_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "highway"
  return hparams


@registry.register_hparams
def r_transformer_highway_tiny():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "highway"
  return hparams


@registry.register_hparams
def r_transformer_dwa_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "dwa"
  return hparams


@registry.register_hparams
def r_transformer_dwa_tiny():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "dwa"
  return hparams


@registry.register_hparams
def r_transformer_dwa_tiny_test():
  hparams = r_transformer_tiny()
  hparams.recurrence_type = "dwa"
  return hparams


@registry.register_hparams
def r_transformer_rnn_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "rnn"
  return hparams


@registry.register_hparams
def r_transformer_gru_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "gru"
  return hparams


@registry.register_hparams
def r_transformer_lstm_base():
  hparams = r_transformer_base()
  hparams.recurrence_type = "lstm"
  return hparams
