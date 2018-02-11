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

"""transformer (attention seq-seq model) with mixtures of experts.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


# The transformer architecture can be defined using the layer_types hparams.
# If not defined, the default types and num_hidden_layers are used as fallback
# values.
#
# Examples of usage:
# "a/a/a/a/a/a": Original base transformer (6 encoder and decoder layers of
# multihead full attention)
# "a/a/a-moe/a": 4 layers with 1 moe at layer 3
# "loc/red/loc/red": Alternate between local and memory compressed attention
# "a/a/a#": Encoder only model (3 layers)
# "#a/a/a": Decoder only model (3 layers)
# "a/a-moe#a/a/a": Encoder (2 layers with 1 moe), decoder (3 layers)
# Note that all combinaisons are not necessarily possibles (some attention
# types are not necessarily compatible with the encoder, or can't accept certain
# types of masking)

SEP_ENCODEC = "#"
SEP_LAYER = "/"
SEP_FF = "-"


@registry.register_model
class TransformerMoe(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  @property
  def use_body_sharded(self):
    return True

  def body_sharded(self, sharded_features):
    # ========= Prepare the input and target =========

    hparams = self._hparams
    dp = self._data_parallelism
    targets = sharded_features["targets"]
    inputs = sharded_features["inputs"]
    target_space = sharded_features["target_space_id"]

    inputs = dp(common_layers.flatten4d3d, inputs)
    targets = dp(common_layers.flatten4d3d, targets)

    def dp_preprocess(x):
      return dp(common_layers.layer_preprocess, x, hparams)

    def dp_postprocess(x, y):
      return dp(common_layers.layer_postprocess, x, y, hparams)

    (encoder_input, encoder_self_attention_bias,
     encoder_decoder_attention_bias) = dp(
         transformer.transformer_prepare_encoder,
         inputs, target_space, hparams)
    (decoder_input, decoder_self_attention_bias) = dp(
        transformer.transformer_prepare_decoder, targets, hparams)
    encoder_input = dp(tf.nn.dropout, encoder_input,
                       1.0 - hparams.layer_prepostprocess_dropout)
    decoder_input = dp(tf.nn.dropout, decoder_input,
                       1.0 - hparams.layer_prepostprocess_dropout)

    cache = dict(extra_loss=0.0)

    def prepostprocess(fct):
      """Apply processing and capture the extra loss."""
      @expert_utils.add_var_scope()
      def decorated(x, *args, **kwargs):
        x = dp_preprocess(x)
        y, loss = fct(x, *args, **kwargs)
        cache["extra_loss"] += loss
        return dp_postprocess(x, y)
      return decorated

    # ========= Compute the transformer architecture =========

    def extract_layer_types(layer_types):
      """Parse the layer string.

      Args:
        layer_types (str): String containing the network architecture. See
          top file comment for examples of format.

      Returns:
        list[tuple[str, str]]: Encoder layers: list of (attention, feed-forward)
        list[tuple[str, str, str]]: Decoder layers: list of (self-attention,
          enc-dec attention, feed-forward)
      """
      # If the architecture has not explicitly been set, we just construct a
      # standard transformer with the fallback values
      if not layer_types:
        layer_types = SEP_LAYER.join(
            [hparams.default_att] * hparams.num_hidden_layers)

      # If encoder not explicitly defined, the encoder will have the same
      # structure as the decoder
      layer_types = layer_types.split(SEP_ENCODEC)
      if len(layer_types) == 1:
        layer_types *= 2

      # Some models don't need the encoder (ex: language modeling)
      # TODO(epot): What are the other conditions (has_input ?)
      if hparams.prepend_mode != "none":
        layer_types[0] = ""

      # Extend the blocks and fill them with the default values if not specified
      final_layers = ([], [])
      for i, blocks_str in enumerate(layer_types):
        for blocks_str in blocks_str.split(SEP_LAYER):
          if not blocks_str:
            continue
          blocks_list = blocks_str.split(SEP_FF)
          # Eventually use the fallback values for the layer_types. If the
          # encoder is empty, do not use the enco-deco attention.
          self_att = blocks_list[0] or hparams.default_att
          ende_att = hparams.default_att if layer_types[0] else "_"
          ff = hparams.default_ff
          if len(blocks_list) > 1:
            ff = blocks_list[-1]
          if len(blocks_list) == 3:
            ende_att = blocks_list[1]
          if i == 0:  # Encoder
            blocks_tuple = (self_att, ff)
          elif i == 1:  # Decoder
            blocks_tuple = (self_att, ende_att, ff)
          final_layers[i].append(blocks_tuple)

      return final_layers

    # ========= Construct the transformer encoder and decoder =========

    encoder_layers, decoder_layers = extract_layer_types(hparams.layer_types)

    layers = common_attention.get_standardized_layers(
        hparams=hparams,
        dp=dp,
        ps_devices=self._ps_devices,
    )

    if hparams.mode == tf.estimator.ModeKeys.TRAIN:

      # Display the encoder-decoder architecture
      def print_layer(name, layers):
        tf.logging.info("{} architecture:".format(name))
        for i, l in enumerate(layers):
          tf.logging.info(" * Layer {}: {}".format(i, " - ".join(l)))
      print_layer("Encoder", encoder_layers)
      print_layer("Decoder", decoder_layers)

    encoder_outputs = []

    x = encoder_input
    with tf.variable_scope("encoder"):
      for layer_num, block_types in enumerate(encoder_layers):
        # Each encoder layers is composed of two blocks:
        # * self-attention block
        # * feed-forward block
        att_type, ff_type = block_types
        with tf.variable_scope("layer_{}".format(layer_num)):
          x = prepostprocess(layers[att_type])(
              x,
              bias=encoder_self_attention_bias,
              name="att_{}".format(att_type),
          )
          x = prepostprocess(layers[ff_type])(
              x,
              name="ff_{}".format(ff_type)
          )
        encoder_outputs.append(x)
      if encoder_outputs:
        encoder_outputs[-1] = dp_preprocess(x)

    x = decoder_input
    with tf.variable_scope("decoder"):
      for layer_num, block_types in enumerate(decoder_layers):
        # Each decoder layers is composed of three blocks:
        # * self-attention block
        # * enco-deco attention block (optional)
        # * feed-forward block
        self_att_type, att_ende_type, ff_type = block_types
        with tf.variable_scope("layer_{}".format(layer_num)):
          x = prepostprocess(layers[self_att_type])(
              x,
              bias=decoder_self_attention_bias,
              name="self_att_{}".format(self_att_type),
          )
          # Only add the enco-deco attention layer if there is an encoder
          if encoder_outputs:
            x = prepostprocess(layers[att_ende_type])(
                x,
                memory_antecedent=encoder_outputs[-1],
                bias=encoder_decoder_attention_bias,
                name="att_ende_{}".format(att_ende_type),
            )
          x = prepostprocess(layers[ff_type])(
              x,
              name="ff_{}".format(ff_type)
          )
      # If normalization is done in layer_preprocess, then it should also be
      # done on the output, since the output can grow very large, being the sum
      # of a whole stack of unnormalized layer outputs.
      x = dp_preprocess(x)

    decoder_output = dp(tf.expand_dims, x, 2)
    return decoder_output, cache["extra_loss"]


@registry.register_hparams
def transformer_moe_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"
  hparams.hidden_size = 512
  hparams.batch_size = 4096
  hparams.max_length = 2001
  hparams.max_input_seq_length = 2000
  hparams.max_target_seq_length = 2000
  hparams.dropout = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 5
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.0
  hparams.shared_embedding_and_softmax_weights = True
  # According to noam, ("n", "da") seems better for harder-to-learn models
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"

  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", False)

  hparams = common_attention.add_standard_attention_hparams(hparams)

  # Decoder layers type. If set, num_decoder_layers parameter will be ignored
  # and the number of decoder layer will be deduced from the string
  # See top file comment for example of usage
  hparams.add_hparam("layer_types", "")
  # Default attention type (ex: a, loc, red,...) and feed-forward type (ex: fc,
  # sep, moe,...)
  hparams.add_hparam("default_att", "a")
  hparams.add_hparam("default_ff", "fc")

  return hparams


@registry.register_hparams
def transformer_moe_8k():
  """Hyper parameters specifics for long sequence generation."""
  hparams = transformer_moe_base()

  hparams.batch_size = 8192
  hparams.max_length = 0  # max_length == batch_size
  hparams.eval_drop_long_sequences = True
  hparams.min_length_bucket = 256  # Avoid cyclic problems for big batches

  hparams.default_ff = "sep"
  hparams.hidden_size = 1024

  return hparams


@registry.register_hparams
def transformer_moe_8k_lm():
  """Language modeling params.

  Will have the following architecture by default:
  * No encoder.
  * Decoder architecture:
    * Layer 0: a - sepm  (masked self-attention/masked separable convolutions)
    * Layer 1: a - sepm
    * Layer 2: a - moe  (mixture of expert layers in the middle)
    * Layer 3: a - sepm
    * Layer 4: a - sepm

  Returns:
    hparams
  """
  hparams = transformer_moe_8k()

  # Use masked versions of local attention and separable convolution
  hparams.default_ff = "sepm"

  # hparams.layer_types contains the network architecture:
  # Start with '#' for decoder only architecture
  hparams.layer_types = "#a/a/a-moe/a/a"  # 5 full attention layers with 1 moe
  # For long sequences, if running out of memory, it's possible to use the
  # one of those two optimized versions instead:
  #  * Memory efficient multihead attention (slow):
  # hparams.layer_types = "#mem/mem/mem-moe/mem/mem"
  #  * Alternate between local/compressed attention layers (faster):
  # hparams.layer_types = "#locm/redm/locm-moe/redm/locm"

  return hparams


@registry.register_hparams
def transformer_moe_2k():
  """Base transformers model with moe.

  Will have the following architecture:
  * No encoder.
    * Layer 0: a - sep  (self-attention - unmasked separable convolutions)
    * Layer 1: a - sep
    * Layer 2: a - sep
    * Layer 3: a - sep
    * Layer 4: a - sep
  * Decoder architecture:
    * Layer 0: a - a - sepm  (self-attention - enco/deco-attention - masked sep)
    * Layer 1: a - a - sepm
    * Layer 2: a - a - moe  (mixture of expert layers in the middle)
    * Layer 3: a - a - sepm
    * Layer 4: a - a - sepm

  Returns:
    hparams
  """
  hparams = transformer_moe_8k()
  hparams.batch_size = 2048

  hparams.default_ff = "sep"

  # hparams.layer_types contains the network architecture:
  encoder_archi = "a/a/a/a/a"
  decoder_archi = "a-sepm/a-sepm/a-moe/a-sepm/a-sepm"
  hparams.layer_types = "{}#{}".format(encoder_archi, decoder_archi)

  return hparams


@registry.register_hparams
def transformer_moe_12k():
  """Hyper parameters specifics for long sequence generation."""
  hparams = transformer_moe_8k()
  hparams.batch_size = 12000
  # At 12k, the softmax become the memory bottleneck
  hparams.factored_logit = True
  return hparams


@registry.register_hparams
def transformer_moe_prepend_8k():
  """Model which formulate a seq2seq problem as language modeling."""
  hparams = transformer_moe_8k()
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.eval_drop_long_sequences = False
  hparams.max_input_seq_length = 7500,
  hparams.default_ff = "sepm"
  hparams.layer_types = "locm/redm/locm-moe/redm/locm"
  hparams.moe_num_experts = 256
  return hparams
