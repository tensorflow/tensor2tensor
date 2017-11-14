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

"""transformer (attention seq-seq model) with mixtures of experts.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

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


def partial(fct, *args, **kwargs):
  """Wrapper around functools.partial for Python 2 compatibility with wraps."""
  new_fct = functools.partial(fct, *args, **kwargs)
  new_fct = functools.wraps(fct)(new_fct)
  return new_fct


@registry.register_model
class TransformerMoe(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  @expert_utils.add_var_scope("transformer_moe")
  def model_fn_body_sharded(self, sharded_features):

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
    cache = dict(extra_loss=0)
    moe_hidden_sizes = [int(s) for s in hparams.moe_hidden_sizes.split(",")]
    expert_fn = expert_utils.ffn_expert_fn(
        hparams.hidden_size, moe_hidden_sizes, hparams.hidden_size)

    # ========= Define some utils decorators =========

    def prepostprocess(fct):
      """Add pre and post processing."""
      # WARNING: Should be applied after dp (pre/post-process use dp and
      # can be applied to function which doesn't use dp)
      @functools.wraps(fct)
      def decorated(x, *args, **kwargs):
        x = dp_preprocess(x)
        y = fct(x, *args, **kwargs)
        return dp_postprocess(x, y)
      return decorated

    def dp_wrapper(fct):
      """Encapsulate the function in a data parallelism object."""
      @functools.wraps(fct)
      def decorated(*args, **kwargs):
        return dp(fct, *args, **kwargs)
      return decorated

    def add_kwargs(
        fct,
        enco_kwargs=None,
        deco_kwargs=None,
        endeco_kwargs=None,  # Enco-deco attention: overwrite deco_kwargs
    ):
      """Allow to have different arguments for the encoder and decoder."""
      # WARNING: If this decorator is applied before dp_wrapper, the kwargs
      # may not be correctly dipatched across the devices.
      @functools.wraps(fct)
      def decorated(*args, **kwargs):
        current_scope = tf.contrib.framework.get_name_scope()
        if "/encoder/" in current_scope:
          kwargs.update(enco_kwargs or {})
        elif "/decoder/" in current_scope:
          kwargs.update(deco_kwargs or {})
          if "/att_ende_" in current_scope:
            kwargs.update(endeco_kwargs or {})
        return fct(*args, **kwargs)
      return decorated

    def capture_extra_loss(fct, loss_coef=1.0):
      """Capture the additional loss."""
      @functools.wraps(fct)
      def decorated(*args, **kwargs):
        y, loss = fct(*args, **kwargs)
        cache["extra_loss"] += loss * loss_coef
        return y
      return decorated

    def remove_kwargs(fct, extra_params):
      """Remove some unused parameters."""
      @functools.wraps(fct)
      def decorated(*args, **kwargs):
        for k in extra_params:  # Remove the extra params
          kwargs.pop(k, None)
        return fct(*args, **kwargs)
      return decorated

    # def pad_remover(fct):
    #   """Remove/restore the padding on the input."""
    #   @functools.wraps(fct)
    #   def decorated(x, *args, **kwargs):
    #     x = pad_remover.remove(x)
    #     x = fct(x, *args, **kwargs)
    #     x = pad_remover.restore(x)
    #     return x
    #   return decorated

    # ========= Define the available layers =========
    total_key_depth = hparams.attention_key_channels or hparams.hidden_size
    total_value_depth = hparams.attention_value_channels or hparams.hidden_size

    # Multi-head full attention layer
    multihead_attention = partial(
        common_attention.multihead_attention,
        total_key_depth=total_key_depth,
        total_value_depth=total_value_depth,
        output_depth=hparams.hidden_size,
        num_heads=hparams.num_heads,
        dropout_rate=hparams.attention_dropout,
    )
    multihead_attention = dp_wrapper(multihead_attention)
    multihead_attention = add_kwargs(  # After dp to correctly dispatch kwargs
        multihead_attention,
        enco_kwargs={"bias": encoder_self_attention_bias},
        deco_kwargs={"bias": decoder_self_attention_bias},
        endeco_kwargs={"bias": encoder_decoder_attention_bias},
    )
    multihead_attention = prepostprocess(multihead_attention)

    # Local attention layer
    # Reuse same parameters as multihead_attention (dp and pre/post-processing
    # already applied)
    # Only works for self attention. Always mask the future.
    local_attention = partial(
        multihead_attention,
        block_length=hparams.attention_loc_block_length,
        attention_type="local_mask_right",
    )

    # Memory-compressed multihead self attention layer
    # Only works for self attention. Always mask the future.
    compressed_attention = partial(
        common_attention.multihead_self_attention_reduced,
        factor=hparams.attention_red_factor,
        nonlinearity=hparams.attention_red_nonlinearity,
        reduction_type=hparams.attention_red_type,
        multihead_params=dict(
            total_key_depth=total_key_depth,
            total_value_depth=total_value_depth,
            num_heads=hparams.num_heads,
            dropout_rate=hparams.attention_dropout,
        )
    )
    compressed_attention = remove_kwargs(
        compressed_attention, ["memory_antecedent"])
    compressed_attention = dp_wrapper(compressed_attention)
    compressed_attention = prepostprocess(compressed_attention)

    # Mixture of expert layer
    distributed_moe = partial(
        expert_utils.distributed_moe,
        dp,
        self._ps_devices,
        train=hparams.mode == tf.estimator.ModeKeys.TRAIN,
        input_size=hparams.hidden_size,
        expert_fn=expert_fn,
        num_experts=hparams.moe_num_experts,
        k=hparams.moe_k,
        loss_coef=hparams.moe_loss_coef
    )
    distributed_moe = capture_extra_loss(distributed_moe)
    distributed_moe = prepostprocess(distributed_moe)

    # FC layer
    conv_hidden_relu = partial(
        common_layers.conv_hidden_relu,
        hidden_size=hparams.filter_size,
        output_size=hparams.hidden_size,
        dropout=hparams.relu_dropout,
    )
    conv_hidden_relu = dp_wrapper(conv_hidden_relu)
    conv_hidden_relu = prepostprocess(conv_hidden_relu)

    # Separable convolution layer
    # Reuse conv_hidden_relu (dp and pre/post-processing already applied)
    # Mask the future for the decoder only
    sep_conv_relu = partial(
        conv_hidden_relu,
        # Parameters copied from the transformer model, could add hparams
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
    )
    sep_conv_relu = add_kwargs(
        sep_conv_relu,
        enco_kwargs={"padding": "SAME"},
        deco_kwargs={"padding": "LEFT"},  # Mask future for decoder
    )

    # This dictionary contains the list of all available layers
    available_layers = dict(
        # Attention layers
        a=multihead_attention,  # Standard multihead full attention
        loc=local_attention,  # Local attention
        red=compressed_attention,  # Memory-compressed attention
        mem=None,  # Memory efficient
        # Feed-forward layers
        moe=distributed_moe,  # Mixture of expert layer
        sep=sep_conv_relu,  # Separable convolution
        fc=conv_hidden_relu,  # Fully connected
    )

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
          with tf.variable_scope("att_{}".format(att_type)):
            x = available_layers[att_type](
                x,
                memory_antecedent=None,
            )
          with tf.variable_scope("ff_{}".format(ff_type)):
            x = available_layers[ff_type](x)
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
          with tf.variable_scope("self_att_{}".format(self_att_type)):
            x = available_layers[self_att_type](
                x,
                memory_antecedent=None,
            )
          with tf.variable_scope("att_ende_{}".format(att_ende_type)):
            # Only add the enco-deco attention layer if there is an encoder
            if encoder_outputs:
              x = available_layers[att_ende_type](
                  x,
                  memory_antecedent=encoder_outputs[-1],
              )
          with tf.variable_scope("ff_{}".format(ff_type)):
            x = available_layers[ff_type](x)
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
  hparams.shared_embedding_and_softmax_weights = int(True)
  # According to noam, ("n", "da") seems better for harder-to-learn models
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"

  hparams.add_hparam("filter_size", 2048)  # Add new ones like this.
  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")
  # Other attention types params
  hparams.add_hparam("attention_loc_block_length", 256)
  hparams.add_hparam("attention_red_factor", 3)
  hparams.add_hparam("attention_red_type", "conv")
  hparams.add_hparam("attention_red_nonlinearity", "none")
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", int(False))

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
  hparams.eval_drop_long_sequences = int(True)
  hparams.min_length_bucket = 256  # Avoid cyclic problems for big batches

  hparams.default_ff = "sep"
  hparams.hidden_size = 1024

  return hparams


@registry.register_hparams
def transformer_moe_12k():
  """Hyper parameters specifics for long sequence generation."""
  hparams = transformer_moe_8k()
  hparams.batch_size = 12000
  # At 12k, the softmax become the memory bottleneck
  hparams.factored_logit = int(True)
  return hparams


@registry.register_hparams
def transformer_moe_prepend_8k():
  """Model which formulate a seq2seq problem as language modeling."""
  hparams = transformer_moe_8k()
  hparams.prepend_mode = "prepend_inputs_masked_attention",
  hparams.eval_drop_long_sequences = int(False),
  hparams.max_input_seq_length = 7500,
  hparams.layer_types = "loc/red/loc-moe/red/loc"
  hparams.moe_num_experts = 256
  return hparams


