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

"""Test of the SymShard programming model.

Symmetric model parallellism.

Each shard (device) has a similar structure with different weights.
Occasional allreduce (sum) across shards.

On TPU, we replicate the whole model on each core.  This is not the intended
use, but we can test the model quality.

Example problem: translate_ende_8k_packed

Preliminary results on languagemodel_lm1b8k_packed (200k steps 8 cores)
  transformer_tpu:             48M params   dev-log-ppl=-1.29   dev-BLEU=27.0
  transformer_symshard_sh4:    49M params   dev-log-ppl=-1.30   dev-BLEU=26.4
  transformer_symshard_base:   98M params   dev-log-ppl=-1.23   dev-BLEU=27.6

  transformer_symshard_base with different mixing fraction (default=0.5):
    mix_fraction=0.0    dev-log-ppl=-1.33
    mix_fraction=0.25   dev-log-ppl=-1.23
    mix_fraction=0.5    dev-log-ppl=-1.23
    mix_fraction=0.75   dev-log-ppl=-1.24
    mix_fraction=1.0    dev-log-ppl=-1.28

TODO(noam): Make sure no one is using super_lm, then delete it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class TransformerSymshard(t2t_model.T2TModel):
  """See file docstring."""

  def body(self, features):
    hparams = self._hparams
    ps_devices = self._ps_devices
    single_device = (len(ps_devices) == 1)
    assert hparams.num_model_shards % len(ps_devices) == 0
    shards_per_device = hparams.num_model_shards // len(ps_devices)
    model_devices = [ps_devices[i // shards_per_device]
                     for i in xrange(hparams.num_model_shards)]
    print("model_devices = %s" % model_devices)
    mp = expert_utils.Parallelism(model_devices, reuse=False)
    targets_vocab_size = self._problem_hparams.vocabulary["targets"].vocab_size
    # squeeze out channels, heights
    targets = tf.squeeze(features["targets_raw"], [2, 3])
    targets_embedding_var = mp(
        tf.get_variable, "embedding",
        [[targets_vocab_size, hparams.hidden_size]] * mp.n,
        initializer=tf.random_normal_initializer(
            0.0, hparams.hidden_size**-0.5))
    shifted_targets = common_layers.shift_right_2d(targets)
    # Bypass the symbol modality and use a different embedding on each shard.
    if single_device:
      targets_embedding_var_combined = tf.concat(targets_embedding_var, 1)
      decoder_input_combined = common_layers.embedding(
          shifted_targets, targets_vocab_size,
          hparams.hidden_size * mp.n,
          multiplier=hparams.hidden_size**0.5,
          embedding_var=targets_embedding_var_combined,
      )
      decoder_input = tf.split(decoder_input_combined, mp.n, axis=2)
    else:
      targets_embedding_var_combined = None
      decoder_input = mp(
          common_layers.embedding, shifted_targets, targets_vocab_size,
          hparams.hidden_size,
          multiplier=hparams.hidden_size**0.5,
          embedding_var=targets_embedding_var,
      )
    decoder_self_attention_bias = mp(
        common_attention.attention_bias_lower_triangle,
        tf.shape(targets)[1])
    if "targets_segmentation" in features:
      # "Packed" dataset - keep the examples from seeing each other.
      targets_segmentation = features["targets_segmentation"]
      targets_position = features["targets_position"]
      decoder_self_attention_bias = mp(
          tf.add, decoder_self_attention_bias,
          mp(common_attention.attention_bias_same_segment,
             targets_segmentation, targets_segmentation))
      decoder_input = mp(
          common_attention.add_timing_signal_1d_given_position,
          decoder_input, targets_position)
    else:
      targets_position = None
      decoder_self_attention_bias = mp(
          common_attention.attention_bias_lower_triangle,
          tf.shape(targets)[1])
      decoder_input = mp(common_attention.add_timing_signal_1d, decoder_input)

    if self.has_input:
      inputs = tf.squeeze(features["inputs_raw"], [2, 3])
      inputs_vocab_size = self._problem_hparams.vocabulary["inputs"].vocab_size
      # share everything for now
      share_inputs_and_targets_embedding = True
      if share_inputs_and_targets_embedding:
        assert inputs_vocab_size == targets_vocab_size
        inputs_embedding_var = targets_embedding_var
        inputs_embedding_var_combined = targets_embedding_var_combined
      if single_device:
        encoder_input_combined = common_layers.embedding(
            inputs, inputs_vocab_size,
            hparams.hidden_size * mp.n,
            multiplier=hparams.hidden_size**0.5,
            embedding_var=inputs_embedding_var_combined,
        )
        encoder_input = tf.split(encoder_input_combined, mp.n, axis=2)
      else:
        encoder_input = mp(
            common_layers.embedding, inputs, inputs_vocab_size,
            hparams.hidden_size,
            multiplier=hparams.hidden_size**0.5,
            embedding_var=inputs_embedding_var,
        )
      if "inputs_segmentation" in features:
        # "Packed" dataset - keep the examples from seeing each other.
        inputs_segmentation = features["inputs_segmentation"]
        inputs_position = features["inputs_position"]
        encoder_self_attention_bias = mp(
            common_attention.attention_bias_same_segment,
            inputs_segmentation, inputs_segmentation)
        encoder_decoder_attention_bias = mp(
            common_attention.attention_bias_same_segment,
            targets_segmentation, inputs_segmentation)
        encoder_input = mp(
            common_attention.add_timing_signal_1d_given_position,
            encoder_input, inputs_position)
      else:
        encoder_padding = tf.to_float(tf.equal(inputs, 0))
        ignore_padding = common_attention.attention_bias_ignore_padding(
            encoder_padding)
        encoder_self_attention_bias = ignore_padding
        encoder_decoder_attention_bias = ignore_padding
        inputs_position = None
        encoder_input = mp(common_attention.add_timing_signal_1d, encoder_input)

      # encoder stack here
      with tf.variable_scope("encoder"):
        encoder_input = mp(
            tf.nn.dropout, encoder_input,
            1.0 - hparams.layer_prepostprocess_dropout)
        encoder_output = _layer_stack(
            mp,
            encoder_input,
            encoder_self_attention_bias,
            hparams.encoder_layers,
            hparams)
    else:
      encoder_decoder_attention_bias = None
      encoder_output = None

    with tf.variable_scope("decoder"):
      decoder_input = mp(
          tf.nn.dropout, decoder_input,
          1.0 - hparams.layer_prepostprocess_dropout)
      decoder_output = _layer_stack(
          mp,
          decoder_input,
          decoder_self_attention_bias,
          layers=hparams.decoder_layers,
          hparams=hparams,
          encoder_output=encoder_output,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias)

    # Bypass the symbol modality and compute logits directly.
    # We compute a different set of logits on each shard, and sum them.
    # Share the weights with the target embedding.
    output_var = targets_embedding_var
    output_var_combined = targets_embedding_var_combined
    if single_device:
      decoder_output = tf.concat(decoder_output, 2)
      logits = tf.tensordot(decoder_output, output_var_combined, [[2], [1]])
      num, denom = common_layers.padded_cross_entropy(
          logits, targets, hparams.label_smoothing)
      training_loss = num / denom
    else:
      logits = mp(
          tf.tensordot, decoder_output, output_var, [[[2], [1]]] * mp.n)
      logits = common_layers.all_reduce_ring(logits, mp)
      # On each device, we compute the loss for a part of the batch.
      # This is faster than computing the whole loss on one shard.
      mp, logits = common_layers.reduce_by_device(mp, logits, lambda l: l[0])
      def _loss_for_shard(logits, targets, shard):
        logits = common_layers.approximate_split(logits, mp.n, 0)[shard]
        targets = common_layers.approximate_split(targets, mp.n, 0)[shard]
        return common_layers.padded_cross_entropy(
            logits, targets, hparams.label_smoothing)
      num, denom = mp(_loss_for_shard, logits, targets, range(mp.n))
      training_loss = tf.add_n(num) / tf.add_n(denom)
      logits = logits[0]
    logits = tf.expand_dims(tf.expand_dims(logits, 2), 3)
    # override training loss so that it is not computed externally.
    losses = {"training": training_loss}
    return logits, losses


def _layer_stack(mp,
                 inputs,
                 self_attention_bias,
                 layers,
                 hparams,
                 encoder_output=None,
                 encoder_decoder_attention_bias=None):
  """A stack of layers.

  Args:
    mp: a Parallelism object
    inputs: a list of Tensors
    self_attention_bias: list of bias Tensor for self-attention
      (see common_attention.attention_bias())
    layers: a string
    hparams: hyperparameters for model
    encoder_output: optional list of tensors
    encoder_decoder_attention_bias: optional list of tensors

  Returns:
    y: a list of Tensors
  """
  layers = layers.strip(",").split(",")

  # scaled_dot_product_attention_with_projections uses a 3d attention bias
  # (no heads), where multihead_attention uses 4d attention bias.
  self_attention_bias_3d = mp(tf.squeeze, self_attention_bias, 1)
  if encoder_decoder_attention_bias is not None:
    encoder_decoder_attention_bias_3d = mp(
        tf.squeeze, encoder_decoder_attention_bias, 1)
  relu_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "relu_dropout_broadcast_dims", "")))
  mix_size = int(hparams.mix_fraction * hparams.hidden_size)
  accumulator = inputs
  x = inputs
  for layer_num, layer_type in enumerate(layers):
    with tf.variable_scope("%s_%d" % (layer_type, layer_num)):
      tf.logging.info("%s_%d" % (layer_type, layer_num))
      if layer_type == "a":
        # accumulate
        accumulator = mp(tf.add, x, accumulator)
        x = accumulator
      elif layer_type == "n":
        # normalize
        x = mp(common_layers.apply_norm,
               x, hparams.norm_type, hparams.hidden_size, hparams.norm_epsilon)
      elif layer_type == "d":
        # dropout
        x = mp(tf.nn.dropout, x, 1.0 - hparams.layer_prepostprocess_dropout)
      elif layer_type == "m":
        if mix_size > 0:
          # mix across shards
          def _split(t):
            return tuple(tf.split(
                t, [mix_size, hparams.hidden_size - mix_size], 2))
          to_mix, to_keep = mp(_split, x)
          mixed = common_layers.all_reduce_ring(to_mix, mp)
          mixed = mp(tf.multiply, mixed, mp.n ** -0.5)
          x = mp(lambda a, b: tf.concat([a, b], 2), mixed, to_keep)
      elif layer_type == "att":
        # single-head attention
        q = mp(tf.layers.dense, x, hparams.hidden_size, use_bias=False,
               name="q_transform")
        x = mp(
            common_attention.scaled_dot_product_attention_simple,
            q, x, x, self_attention_bias_3d)
        x = mp(tf.layers.dense, x, hparams.hidden_size, use_bias=False,
               name="o_transform")
      elif layer_type == "enc-att":
        # single-head attention over encoder
        q = mp(tf.layers.dense, x, hparams.hidden_size, use_bias=False,
               name="q_transform")
        assert encoder_output is not None
        x = mp(
            common_attention.scaled_dot_product_attention_simple,
            q, encoder_output, encoder_output,
            encoder_decoder_attention_bias_3d)
        x = mp(tf.layers.dense, x, hparams.hidden_size, use_bias=False,
               name="o_transform")
      elif layer_type == "multihead-att":
        # multi-head attention
        x = mp(
            common_attention.multihead_attention,
            x,
            None,
            self_attention_bias,  # bias
            hparams.multihead_attention_key_channels or hparams.hidden_size,
            hparams.multihead_attention_value_channels or hparams.hidden_size,
            hparams.hidden_size,
            hparams.multihead_attention_num_heads,
            hparams.attention_dropout)
      elif layer_type == "enc-multihead-att":
        # multi-head attention
        x = mp(
            common_attention.multihead_attention,
            x,
            encoder_output,
            encoder_decoder_attention_bias,  # bias
            hparams.multihead_attention_key_channels or hparams.hidden_size,
            hparams.multihead_attention_value_channels or hparams.hidden_size,
            hparams.hidden_size,
            hparams.multihead_attention_num_heads,
            hparams.attention_dropout)
      elif layer_type == "ffn":
        x = mp(
            common_layers.dense_relu_dense, x,
            hparams.filter_size, hparams.hidden_size,
            dropout=hparams.relu_dropout,
            dropout_broadcast_dims=[relu_dropout_broadcast_dims] * mp.n)
      else:
        assert False, "unknown sublayer %s" % layer_type
  return x


@registry.register_hparams
def transformer_symshard_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 256
  hparams.batch_size = 2048
  hparams.max_length = 0
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.add_hparam("attention_dropout", 0.1)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("relu_dropout_broadcast_dims", "1")
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_schedule = "rsqrt_decay"
  hparams.learning_rate_warmup_steps = 10000
  hparams.initializer_gain = 1.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  # TODO(noam): use this to control sharing.  We now share always
  hparams.shared_embedding_and_softmax_weights = True
  # we only want one data shard.
  hparams.no_data_parallelism = True
  # bypass the symbol modality so that we can use model parallelism.
  hparams.target_modality = "symbol:identity"
  hparams.input_modalities = "inputs:symbol:identity"
  hparams.add_hparam("filter_size", 1280)
  hparams.add_hparam("mix_fraction", 0.5)
  # attention-related flags
  hparams.add_hparam("multihead_attention_num_heads", 4)
  hparams.add_hparam("multihead_attention_key_channels", 0)
  hparams.add_hparam("multihead_attention_value_channels", 0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam(
      "encoder_layers", ("n,att,m,d,a," "n,ffn,m,d,a,") * 6 + "n,d")
  hparams.add_hparam(
      "decoder_layers",
      ("n,att,m,d,a," "n,enc-att,m,d,a," "n,ffn,m,d,a,") * 6 + "n,d")
  # Number of model shards - each one has separate parameters.
  # Changing this number invalidates checkpoints.
  hparams.add_hparam("num_model_shards", 8)
  return hparams


@registry.register_hparams
def transformer_symshard_sh4():
  """4 shards instead of 8.  Similar model size to transformer_tpu()."""
  hparams = transformer_symshard_base()
  hparams.num_model_shards = 4
  return hparams


@registry.register_hparams
def transformer_symshard_lm_0():
  """For language modeling - suggested problem languagemodel_lm1b8k_packed."""
  hparams = transformer_symshard_base()
  hparams.label_smoothing = 0
  return hparams


@registry.register_hparams
def transformer_symshard_h4():
  """4 heads per shard."""
  hparams = transformer_symshard_base()
  hparams.encoder_layers = ("n,multihead-att,m,d,a," "n,ffn,m,d,a,") * 6 + "n,d"
  hparams.decoder_layers = (
      ("n,multihead-att,m,d,a," "n,enc-multihead-att,m,d,a," "n,ffn,m,d,a,") * 6
      + "n,d")
  return hparams
