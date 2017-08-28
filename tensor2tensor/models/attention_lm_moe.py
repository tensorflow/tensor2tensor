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
from tensor2tensor.utils import diet
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


class AttentionMoeType(object):
  NONE = "none"
  LOCAL = "local"
  GLOBAL = "global"

  @staticmethod
  def get_choices():
    return [
        AttentionMoeType.NONE,
        AttentionMoeType.LOCAL,
    ]


@registry.register_model
class AttentionLmMoe(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def model_fn_body_sharded(self, sharded_features):
    # Remove dropout if not training
    hparams = self._hparams
    dp = self._data_parallelism
    targets = sharded_features["targets"]
    targets = dp(tf.squeeze, targets, 2)

    def preprocess(x):
      return dp(common_layers.layer_preprocess, x, hparams)

    def postprocess(x, y):
      return dp(common_layers.layer_postprocess, x, y, hparams)

    (decoder_input, decoder_self_attention_bias, pad_remover) = dp(
        attention_lm_moe_prepare_decoder, targets, hparams)

    x = dp(tf.nn.dropout, decoder_input,
           1.0 - hparams.layer_prepostprocess_dropout)
    extra_loss = 0.0
    moe_hidden_sizes = [int(s) for s in hparams.moe_hidden_sizes.split(",")]
    if hparams.diet_experts:
      hsize, = moe_hidden_sizes

      def _diet_expert(x):
        return diet.diet_expert(x, hsize, diet.diet_adam_optimizer_params())

      expert_fn = _diet_expert
    else:
      expert_fn = expert_utils.ffn_expert_fn(
          hparams.hidden_size, moe_hidden_sizes, hparams.hidden_size)

    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope(
            "attention_{}".format(hparams.attention_moe_type)):
          x = preprocess(x)
          if hparams.attention_moe_type == AttentionMoeType.NONE:
            y = dp(
                common_attention.multihead_attention,
                x,
                None,
                decoder_self_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                name="decoder_self_attention")
          elif hparams.attention_moe_type == AttentionMoeType.LOCAL:
            y, loss = dp(
                common_attention.local_expert_attention,
                x,
                k=2,
                loss_coef=hparams.attention_load_balance,
                attention_num_experts=hparams.attention_num_experts,
                train=hparams.mode == tf.contrib.learn.ModeKeys.TRAIN,
                pad_remover=pad_remover,
                mask_right=True,
                attention_kq_size=hparams.attention_kq_size,
                attention_v_size=hparams.attention_v_size)
            # TODO(avaswani, epot, noam): Do we need to divide by num shards ?
            extra_loss += tf.add_n(loss) / dp.n
          else:
            raise ValueError("Only {} supported for now.".format(
                AttentionMoeType.get_choices()))
          x = postprocess(x, y)
        with tf.variable_scope("ffn"):
          if str(layer) in hparams.moe_layers.split(","):
            y, loss = expert_utils.distributed_moe(
                dp,
                self._ps_devices,
                preprocess(x),
                hparams.mode == tf.contrib.learn.ModeKeys.TRAIN,
                input_size=hparams.hidden_size,
                expert_fn=expert_fn,
                num_experts=hparams.moe_num_experts,
                k=hparams.moe_k,
                loss_coef=hparams.moe_loss_coef)
            extra_loss += loss
          else:
            y = dp(
                common_layers.conv_hidden_relu,
                preprocess(x),
                hparams.filter_size,
                hparams.hidden_size,
                dropout=hparams.relu_dropout)
          x = postprocess(x, y)
    x = preprocess(x)
    decoder_output = dp(tf.expand_dims, x, 2)
    return decoder_output, extra_loss


def attention_lm_moe_prepare_decoder(targets, hparams):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a Tensor, containing large negative values
    to implement masked attention and possibly baises for diagonal alignments
    pad_remover (expert_utils.PadRemover): an util object to remove padding
  """
  targets_pad_mask = common_attention.embedding_to_padding(targets)
  with tf.name_scope("pad_remover"):
    pad_remover = expert_utils.PadRemover(targets_pad_mask)

  if hparams.prepend_mode == "prepend_inputs_full_attention":
    decoder_self_attention_bias = (
        common_attention.attention_bias_prepended(targets_pad_mask))
  else:
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
  decoder_input = common_layers.shift_left_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias, pad_remover)


@registry.register_hparams
def attention_lm_moe_base():
  """Set of hyperparameters.

  suitable for 1 gpu.
  on lm1b_32k:
     ~229M params
     0.9 steps/sec on  [GeForce GTX TITAN X]

  Returns:
    a hparams object
  """
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
  hparams.num_hidden_layers = 4
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.0
  hparams.shared_embedding_and_softmax_weights = int(False)
  hparams.add_hparam("filter_size", 2048)  # Add new ones like this.
  hparams.moe_num_experts = 32
  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("moe_layers", "2")  # comma separated list of layer numbers
  # moe params. local attention moe.
  hparams.add_hparam("attention_moe_type", AttentionMoeType.NONE)
  hparams.add_hparam("attention_num_experts", 16)
  # Key, query and value dimensions for the attention
  hparams.add_hparam("attention_kq_size", 128)
  hparams.add_hparam("attention_v_size", 256)
  # Loss coef for load balancing
  hparams.add_hparam("attention_load_balance", 2e-2)
  hparams.add_hparam("diet_experts", int(False))
  return hparams


@registry.register_hparams
def attention_lm_moe_base_ae():
  """Base model with attention expert."""
  hparams = attention_lm_moe_base()
  hparams.attention_moe_type = AttentionMoeType.LOCAL
  hparams.max_length = hparams.batch_size
  hparams.eval_drop_long_sequences = int(True)
  hparams.batching_mantissa_bits = 2  # More buckets
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 10000
  return hparams


@registry.register_hparams
def attention_lm_moe_small():
  """Cheap model for single-gpu training.

  on lm1b_32k:
     ~312M params
     1.6 steps/sec on  [GeForce GTX TITAN X]
     After 50K steps on 8 GPUs (synchronous):
        eval_log_ppl_per_token = 3.31

  Returns:
    an hparams object.
  """
  hparams = attention_lm_moe_base()
  hparams.num_hidden_layers = 4
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.moe_num_experts = 128
  hparams.moe_layers = "2"
  return hparams


@registry.register_hparams
def attention_lm_moe_tiny():
  """Cheap model for debugging.

  Returns:
    an hparams object.
  """
  hparams = attention_lm_moe_small()
  hparams.moe_num_experts = 32
  return hparams


@registry.register_hparams
def attention_lm_attention_moe_tiny():
  """Cheap model for debugging.

  Returns:
    an hparams object.
  """
  hparams = attention_lm_moe_small()
  hparams.moe_layers = ""
  hparams.attention_num_experts = 128
  hparams.filter_size = 8192
  hparams.attention_moe_type = AttentionMoeType.LOCAL
  return hparams


@registry.register_hparams
def attention_lm_no_moe_small():
  """Without the mixture of experts (for comparison).

  on lm1b_32k:
     ~45M params
     2 steps/sec on  [GeForce GTX TITAN X]
     After 50K steps on 8 GPUs (synchronous):
        eval_log_ppl_per_token = 3.51

  Returns:
    an hparams object.
  """
  hparams = attention_lm_moe_small()
  hparams.moe_layers = ""
  return hparams


@registry.register_hparams
def attention_lm_moe_large():
  """Large model for distributed training.

  Over 1B parameters, so requires multi-gpu training due to memory
   requirements.

  on lm1b_32k:
     After 45K steps on 8 GPUs (synchronous):
        eval_log_ppl_per_token = 3.18
        eval_ppl_per_word = exp(1.107893 * eval_log_ppl_per_token) = 33.9

  Returns:
    an hparams object.
  """
  hparams = attention_lm_moe_base()
  hparams.num_hidden_layers = 5
  hparams.moe_layers = "3"
  hparams.hidden_size = 1024
  hparams.num_heads = 16
  hparams.filter_size = 4096
  hparams.moe_hidden_sizes = "4096"
  hparams.moe_num_experts = 128
  hparams.layer_prepostprocess_dropout = 0.2
  return hparams


@registry.register_hparams
def attention_lm_moe_large_diet():
  hparams = attention_lm_moe_large()
  hparams.diet_experts = int(True)
  return hparams


@registry.register_hparams
def attention_lm_moe_32b_diet():
  """Unnecessarily large model with 32B params - because we can."""
  hparams = attention_lm_moe_large_diet()
  hparams.moe_hidden_sizes = "16384"
  hparams.moe_num_experts = 1024
  return hparams


@registry.register_hparams
def attention_lm_moe_24b_diet():
  """Unnecessarily large model with 24B params - because we can."""
  hparams = attention_lm_moe_large_diet()
  hparams.moe_hidden_sizes = "12288"
  hparams.moe_num_experts = 1024
  hparams.batch_size = 4096
  return hparams


@registry.register_hparams
def attention_lm_moe_translation():
  """Version to use for seq2seq."""
  hparams = attention_lm_moe_base()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.learning_rate = 0.4
  hparams.prepend_mode = "prepend_inputs_masked_attention"
  hparams.max_length = 512
  hparams.label_smoothing = 0.1
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.num_hidden_layers = 6
  hparams.moe_layers = "0,1,2,3,4,5"
  hparams.shared_embedding_and_softmax_weights = int(True)
  return hparams
