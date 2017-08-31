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

import functools

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


ModeKeys = tf.contrib.learn.ModeKeys  # pylint: disable=invalid-name


class AttentionType(object):
  MULTIHEAD = "multihead"
  LOCAL_EXPERTS = "local_experts"
  GLOBAL_MOE = "global_experts"
  MEMORY_EFFICIENT = "memory_efficient"

  @staticmethod
  def get_choices():
    return [
        AttentionType.MULTIHEAD,
        AttentionType.LOCAL_EXPERTS,
        AttentionType.MEMORY_EFFICIENT,
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

    if hparams.attention_type == AttentionType.LOCAL_EXPERTS:
      # As preprocess and postprocess are called with batch of size one (all
      # batches concatenated), we just make sure that batch_norm is not use (
      # should not either way)
      assert hparams.norm_type != "batch"

      dp_remove_pad = functools.partial(
          dp, remove_pad, pad_remover=pad_remover, mode=hparams.mode)
      dp_restore_pad = functools.partial(
          dp, restore_pad, ref_x=x, pad_remover=pad_remover, mode=hparams.mode)
    elif (hparams.attention_type == AttentionType.MULTIHEAD or
          hparams.attention_type == AttentionType.MEMORY_EFFICIENT):
      # Using identity function: No effect
      dp_remove_pad = lambda x: (x, None)
      dp_restore_pad = lambda x: x
    else:
      raise ValueError("Only {} supported for now.".format(
          AttentionType.get_choices()))

    def print_shape(x, suffix):
      # To help debugging, print the input/output shapes at inference and eval
      # Inference for long sequences can take a long time, so that's help to
      # see the progession of the generation
      if hparams.mode == ModeKeys.TRAIN:
        return x
      return tf.Print(x, [tf.shape(x)], "shape_x_{}".format(suffix))

    x = dp(print_shape, x, "in")
    x, batch_coordinate = dp_remove_pad(x)
    x = dp(print_shape, x, "in_flat")

    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope(
            "attention_{}".format(hparams.attention_type)):
          if hparams.attention_type == AttentionType.MULTIHEAD:
            y = dp(
                common_attention.multihead_attention,
                preprocess(x),
                None,
                decoder_self_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                name="decoder_self_attention")
          elif hparams.attention_type == AttentionType.MEMORY_EFFICIENT:
            assert hparams.layer_preprocess_sequence == "n"
            y = dp(
                common_attention.multihead_self_attention_memory_efficient,
                x,
                decoder_self_attention_bias,
                hparams.num_heads,
                name="decoder_self_attention")
          elif hparams.attention_type == AttentionType.LOCAL_EXPERTS:
            y, loss = dp(
                common_attention.local_expert_attention,
                preprocess(x),
                k=hparams.attention_moe_k,
                loss_coef=hparams.attention_load_balance,
                attention_num_experts=hparams.attention_num_experts,
                train=hparams.mode == ModeKeys.TRAIN,
                batch_coordinate=batch_coordinate,
                mask_right=True,
                attention_kq_size=hparams.attention_kq_size,
                attention_v_size=hparams.attention_v_size)
            # TODO(avaswani, epot, noam): Do we need to divide by num shards ?
            extra_loss += tf.add_n(loss) / dp.n
          else:
            raise ValueError("Only {} supported for now.".format(
                AttentionType.get_choices()))
          x = postprocess(x, y)
        with tf.variable_scope("ffn"):
          if str(layer) in hparams.moe_layers.split(","):
            y, loss = expert_utils.distributed_moe(
                dp,
                self._ps_devices,
                preprocess(x),
                hparams.mode == ModeKeys.TRAIN,
                input_size=hparams.hidden_size,
                expert_fn=expert_fn,
                num_experts=hparams.moe_num_experts,
                k=hparams.moe_k,
                loss_coef=hparams.moe_loss_coef)
            extra_loss += loss
          elif hparams.memory_efficient_ffn:
            assert hparams.layer_preprocess_sequence == "n"
            y = dp(
                common_layers.conv_hidden_relu_memory_efficient,
                x,
                hparams.filter_size)
          else:
            y = dp(
                common_layers.conv_hidden_relu,
                preprocess(x),
                hparams.filter_size,
                hparams.hidden_size,
                dropout=hparams.relu_dropout)
          x = postprocess(x, y)
    x = preprocess(x)

    x = dp_restore_pad(x)

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
  # TODO(epot): The padding remover should take into account that the input is
  # shifted.
  decoder_input = common_layers.shift_left_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias, pad_remover)


def remove_pad(x, pad_remover, mode):
  """Remove padding by concatenating all dimension into one.

  Args:
    x (tf.Tensor): input of shape [batch_size, length, depth]
    pad_remover (obj): a PadRemover object
    mode (ModeKeys): infer, train or eval. If inference, the padding remover is
      not applied

  Returns:
    tf.Tensor of shape [1,length_nonpad,depth] where
      length_nonpad <= batch_size*length
  """
  # Compute the batch coordinate before flattening all batches
  batch_coordinate = tf.expand_dims(
      common_attention.coordinate_tensor(tf.shape(x)[:-1], axis=0), axis=-1)
  batch_coordinate = expert_utils.flatten_all_but_last(batch_coordinate)

  # Concatenate all tokens (without padding)
  x = expert_utils.flatten_all_but_last(x)

  # Remove padding for training and eval
  if mode != ModeKeys.INFER:
    # This is a hack to allows inference when the <go> token
    # is detected as padding and removed. This works for now because there is
    # no padding at inference.
    batch_coordinate = pad_remover.remove(batch_coordinate)
    x = pad_remover.remove(x)

  batch_coordinate = tf.expand_dims(batch_coordinate, axis=0)
  x = tf.expand_dims(x, axis=0)  # Now batch_size=1
  return x, batch_coordinate


def restore_pad(x, ref_x, pad_remover, mode):
  x = tf.squeeze(x, axis=0)
  if mode != ModeKeys.INFER:
    x = pad_remover.restore(x)
  x = expert_utils.reshape_like(x, ref_x)
  return x


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
  hparams.add_hparam("attention_type", AttentionType.MULTIHEAD)
  hparams.add_hparam("attention_moe_k", 2)
  hparams.add_hparam("attention_num_experts", 16)
  # Key, query and value dimensions for the attention
  hparams.add_hparam("attention_kq_size", 128)
  hparams.add_hparam("attention_v_size", 256)
  # Loss coef for load balancing
  hparams.add_hparam("attention_load_balance", 2e-2)
  hparams.add_hparam("diet_experts", int(False))
  hparams.add_hparam("memory_efficient_ffn", int(False))
  return hparams


@registry.register_hparams
def attention_lm_moe_base_ae():
  """Base model with attention expert."""
  hparams = attention_lm_moe_base()
  hparams.attention_type = AttentionType.LOCAL_EXPERTS
  hparams.max_length = hparams.batch_size
  hparams.eval_drop_long_sequences = int(True)
  hparams.min_length_bucket = 256  # Avoid cyclic problems for big batches
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
  hparams.attention_type = AttentionType.LOCAL_EXPERTS
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
def attention_lm_moe_memory_efficient():
  """Memory-efficient version."""
  hparams = attention_lm_moe_large()
  hparams.diet_experts = int(True)
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.memory_efficient_ffn = True
  hparams.attention_type = AttentionType.MEMORY_EFFICIENT
  hparams.num_heads = 8
  hparams.factored_logits = int(True)
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
