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
"""Universal Transformers.


Universal Transformer is recurrent in depth while employing self-attention
to combine information from different parts of sequences.
In contrast to the Transformer, given enough memory its recurrence in depth
makes the Universal Transformer computationally universal.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models.research import universal_transformer_util
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class UniversalTransformer(transformer.Transformer):
  """Universal Transformer: Depth-wise recurrent transformer model."""

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode Universal Transformer inputs.

    It is similar to "transformer.encode", but it uses
    "universal_transformer_util.universal_transformer_encoder" instead of
    "transformer.transformer_encoder".

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: Unused.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
    del losses

    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer.transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    (encoder_output, encoder_extra_output) = (
        universal_transformer_util.universal_transformer_encoder(
            encoder_input,
            self_attention_bias,
            hparams,
            nonpadding=transformer.features_to_nonpadding(features, "inputs"),
            save_weights_to=self.attention_weights))

    return encoder_output, encoder_decoder_attention_bias, encoder_extra_output

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             nonpadding=None,
             losses=None):
    """Decode Universal Transformer outputs from encoder representation.

    It is similar to "transformer.decode", but it uses
    "universal_transformer_util.universal_transformer_decoder" instead of
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
      cache: Unimplemented.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]
      losses: Unused.

    Returns:
       Tuple of:
         Final decoder representation. [batch_size, decoder_length,
            hidden_dim]
         encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)

    """
    del losses
    # TODO(dehghani): enable caching.
    del cache

    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    # No caching in Universal Transformers!
    (decoder_output, dec_extra_output) = (
        universal_transformer_util.universal_transformer_decoder(
            decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            hparams,
            nonpadding=nonpadding,
            save_weights_to=self.attention_weights))

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2), dec_extra_output

  def body(self, features):
    """Universal Transformer main model_fn.


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
    if hparams.add_position_timing_signal:
      # Turning off addition of positional embedding in the encoder/decoder
      # preparation as we do it in the beginning of each step.
      hparams.pos = None

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
      tf.contrib.summary.scalar("act_loss", act_loss)
      return decoder_output, {"act_loss": act_loss}

    return decoder_output

  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: bool, whether to use the TPU codepath.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    return (self._slow_greedy_infer_tpu(features, decode_length) if use_tpu else
            self._slow_greedy_infer(features, decode_length))

  def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
    # Caching is not ebabled in Universal Transformer
    # TODO(dehghani): Support fast decoding for Universal Transformer
    return self._beam_decode_slow(features, decode_length, beam_size,
                                  top_beams, alpha)


@registry.register_model
class UniversalTransformerEncoder(transformer.Transformer):
  """Universal Transformer Encoder: Has no decoder (e.g.for classification)."""

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, input_height,
        hidden_dim] which will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.
      losses: Unused.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_extra_output: which is extra encoder output used in some
            variants of the model (e.g. in ACT, to pass the ponder-time to body)
    """
    del losses
    inputs = common_layers.flatten4d3d(inputs)

    (encoder_input, self_attention_bias, _) = (
        transformer.transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    (encoder_output, encoder_extra_output) = (
        universal_transformer_util.universal_transformer_encoder(
            encoder_input,
            self_attention_bias,
            hparams,
            nonpadding=transformer.features_to_nonpadding(features, "inputs"),
            save_weights_to=self.attention_weights))

    return encoder_output, encoder_extra_output

  def body(self, features):
    """Universal Transformer main model_fn.

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

    assert self.has_input, ("universal_transformer_encoder is applicable on "
                            "problems with inputs")

    inputs = features["inputs"]
    target_space = features["target_space_id"]
    encoder_output, enc_extra_output = self.encode(
        inputs, target_space, hparams, features=features)

    encoder_output = tf.expand_dims(encoder_output, 2)

    if hparams.recurrence_type == "act" and hparams.act_loss_weight != 0:
      ponder_times, remainders = enc_extra_output
      act_loss = hparams.act_loss_weight * tf.reduce_mean(ponder_times +
                                                          remainders)
      tf.contrib.summary.scalar("act_loss", act_loss)

      return encoder_output, {"act_loss": act_loss}
    return encoder_output


def update_hparams_for_universal_transformer(hparams):
  """Adds deault hparams for all of the variants of the Universal Transformer.

  Args:
    hparams: default hparams (usually one of the standard hparams from
      transformer model (like "transformer_base")

  Returns:
    hparams with default values for Universal Transformers hyper-parameters

  """
  # If not None, mixes vanilla transformer with Universal Transformer.
  # Options: None, "before_ut", and "after_ut".
  hparams.add_hparam("mix_with_transformer", None)

  # Number of vanilla transformer layers used to be mixed with u-transofmer.
  hparams.add_hparam("num_mixedin_layers", 2)

  # Type of recurrency:
  # basic, highway, skip, dwa, act, rnn, gru, lstm.
  hparams.add_hparam("recurrence_type", "basic")

  # Number of steps (which is equivalent to num layer in transformer).
  hparams.add_hparam("num_rec_steps", hparams.num_hidden_layers)

  # Add the positional mebedding at each step(horisontal timing)
  hparams.add_hparam("add_position_timing_signal", True)
  if hparams.add_position_timing_signal:
    hparams.pos = None
  # Logic of position shifting when using timing signal:
  # None, "random", "step"
  hparams.add_hparam("position_start_index", None)

  # Add an step embedding at each step (vertical timing)
  hparams.add_hparam("add_step_timing_signal", True)
  # Either "learned" or "sinusoid"
  hparams.add_hparam("step_timing_signal_type", "learned")

  # Add or concat the timing signal (applied both on position and step timing).
  # Options: "add" and "concat".
  hparams.add_hparam("add_or_concat_timing_signal", "add")

  # Add SRU at the beginning of each Universal Transformer step.
  # This can be considered as a position timing signal
  hparams.add_hparam("add_sru", False)

  # Default ffn layer is separable convolution.
  # Options: "fc" and "sepconv".
  hparams.add_hparam("transformer_ffn_type", "sepconv")

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
def universal_transformer_base():
  hparams = transformer.transformer_big()
  hparams = update_hparams_for_universal_transformer(hparams)
  return hparams


@registry.register_hparams
def universal_transformer_big():
  hparams = transformer.transformer_big()
  hparams = update_hparams_for_universal_transformer(hparams)
  hparams.hidden_size = 2048
  hparams.filter_size = 8192
  return hparams


@registry.register_hparams
def universal_transformer_small():
  hparams = transformer.transformer_base()
  hparams = update_hparams_for_universal_transformer(hparams)
  return hparams


@registry.register_hparams
def universal_transformer_tiny():
  hparams = transformer.transformer_tiny()
  hparams = update_hparams_for_universal_transformer(hparams)
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
def universal_transformer_teeny():
  hparams = transformer_teeny()
  hparams = update_hparams_for_universal_transformer(hparams)
  hparams.num_rec_steps = 10
  return hparams


@registry.register_hparams
def universal_transformer_small_dropconnect():
  hparams = universal_transformer_small()
  hparams.gate_ffn_layer = "dense_dropconnect"
  hparams.add_hparam("dropconnect_dropout", 0.5)
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_base():
  hparams = universal_transformer_base()
  hparams.recurrence_type = "act"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_random_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.act_type = "random"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_accumulated_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.act_type = "accumulated"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_global_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.act_type = "global"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_accumulated_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.act_type = "accumulated"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_global_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.act_type = "global"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_random_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.act_type = "random"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_small_sb():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.batch_size = 2048
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_large():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.hidden_size = 1024
  hparams.batch_size = 2048
  hparams.filter_size = 2048
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_tall():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.num_hidden_layers = 16
  hparams.batch_size = 1024
  hparams.act_max_steps = 24
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_tall_actlossw0():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.num_hidden_layers = 16
  hparams.batch_size = 1024
  hparams.act_max_steps = 24
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_tall_actlossw001():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.num_hidden_layers = 16
  hparams.batch_size = 1024
  hparams.act_max_steps = 24
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_small_d03():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.attention_dropout = 0.3
  hparams.relu_dropout = 0.3
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_base_d03():
  hparams = universal_transformer_base()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.attention_dropout = 0.3
  hparams.relu_dropout = 0.3
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_tiny_d02():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.attention_dropout = 0.2
  hparams.relu_dropout = 0.2
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_tiny_d02_sb():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.attention_dropout = 0.2
  hparams.relu_dropout = 0.2
  hparams.batch_size = 2048
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_tiny_sb():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.batch_size = 2048
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_tiny_d05():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.layer_prepostprocess_dropout = 0.5
  hparams.attention_dropout = 0.5
  hparams.relu_dropout = 0.5
  return hparams


@registry.register_hparams
def universal_transformer_small_sb():
  hparams = universal_transformer_small()
  hparams.batch_size = 2048
  return hparams


@registry.register_hparams
def universal_transformer_skip_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "skip"
  return hparams


@registry.register_hparams
def universal_transformer_skip_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "skip"
  return hparams


@registry.register_hparams
def universal_transformer_highway_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "highway"
  return hparams


@registry.register_hparams
def universal_transformer_highway_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "highway"
  return hparams


@registry.register_hparams
def universal_transformer_dwa_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "dwa"
  return hparams


@registry.register_hparams
def universal_transformer_dwa_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "dwa"
  return hparams


@registry.register_hparams
def universal_transformer_dwa_tiny_test():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "dwa"
  return hparams


@registry.register_hparams
def universal_transformer_rnn_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "rnn"
  return hparams


@registry.register_hparams
def universal_transformer_gru_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "gru"
  return hparams


@registry.register_hparams
def universal_transformer_lstm_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "lstm"
  return hparams


@registry.register_hparams
def universal_transformer_position_random_timing_small():
  hparams = universal_transformer_small()
  hparams.position_start_index = "random"
  return hparams


@registry.register_hparams
def universal_transformer_position_random_timing_tiny():
  hparams = universal_transformer_tiny()
  hparams.position_start_index = "random"
  return hparams


@registry.register_hparams
def universal_transformer_position_step_timing_tiny():
  hparams = universal_transformer_tiny()
  hparams.position_start_index = "step"
  return hparams


@registry.register_hparams
def universal_transformer_step_sinusoid_timing_tiny():
  hparams = universal_transformer_tiny()
  hparams.step_timing_signal_type = "sinusoid"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_position_random_timing_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.position_start_index = "random"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_position_step_timing_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.position_start_index = "step"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_step_sinusoid_timing_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.step_timing_signal_type = "sinusoid"
  return hparams


@registry.register_hparams
def universal_transformer_mix_after_ut_small():
  hparams = universal_transformer_small()
  hparams.mix_with_transformer = "before_ut"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_mix_before_ut_small():
  hparams = universal_transformer_small()
  hparams.mix_with_transformer = "before_ut"
  hparams.recurrence_type = "act"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_mix_after_ut_small():
  hparams = universal_transformer_small()
  hparams.mix_with_transformer = "after_ut"
  hparams.recurrence_type = "act"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_concat_tiny():
  hparams = universal_transformer_tiny()
  hparams.recurrence_type = "act"
  hparams.add_or_concat_timing_signal = "concat"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_concat_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.add_or_concat_timing_signal = "concat"
  return hparams


@registry.register_hparams
def adaptive_universal_transformer_with_sru_small():
  hparams = universal_transformer_small()
  hparams.recurrence_type = "act"
  hparams.add_sru = True
  return hparams


@registry.register_hparams
def universal_transformer_fc_small():
  hparams = universal_transformer_small()
  hparams.transformer_ffn_type = "fc"
  return hparams


@registry.register_hparams
def universal_transformer_fc_base():
  hparams = universal_transformer_base()
  hparams.transformer_ffn_type = "fc"
  return hparams


@registry.register_hparams
def universal_transformer_fc_big():
  hparams = universal_transformer_big()
  hparams.transformer_ffn_type = "fc"
  return hparams


@registry.register_ranged_hparams
def universal_transformer_base_range(rhp):
  """Small range of hyperparameters."""
  # After starting from base, set intervals for some parameters.
  rhp.set_discrete("num_rec_steps", [6, 8, 10])
  rhp.set_discrete("hidden_size", [1024, 2048, 4096])
  rhp.set_discrete("filter_size", [2048, 4096, 8192])
  rhp.set_discrete("num_heads", [8, 16, 32])
  rhp.set_discrete("transformer_ffn_type", ["sepconv", "fc"])
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_float("weight_decay", 0.0, 2.0)


@registry.register_ranged_hparams
def adaptive_universal_transformer_base_range(rhp):
  """Small range of hyperparameters."""
  # After starting from base, set intervals for some parameters.
  rhp.set_discrete("act_max_steps", [8, 16, 32])
  rhp.set_float("act_loss_weight", 0.0, 0.5)
  rhp.set_discrete("hidden_size", [1024, 2048, 4096])
  rhp.set_discrete("filter_size", [2048, 4096, 8192])
  rhp.set_discrete("num_heads", [8, 16, 32])
  rhp.set_discrete("transformer_ffn_type", ["sepconv", "fc"])
  rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
  rhp.set_float("weight_decay", 0.0, 2.0)
