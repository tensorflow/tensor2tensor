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


ModeKeys = tf.estimator.ModeKeys  # pylint: disable=invalid-name


class AttentionType(object):
  """Enum of the attention layers types."""
  MULTIHEAD = "multihead"
  LOCAL_EXPERTS = "local_experts"
  GLOBAL_MOE = "global_experts"
  MEMORY_EFFICIENT = "memory_efficient"
  SPARSE_MULTIHEAD = "sparse_multihead"
  SPARSE_MULTIHEAD_TRUNCATED = "sparse_multihead_truncated"
  MULTIHEAD_REDUCED = "multihead_reduced"
  MULTIHEAD_FULL = "multihead_full"

  @staticmethod
  def get_choices():
    return [
        AttentionType.MULTIHEAD,
        AttentionType.LOCAL_EXPERTS,
        AttentionType.MEMORY_EFFICIENT,
        AttentionType.SPARSE_MULTIHEAD,
        AttentionType.SPARSE_MULTIHEAD_TRUNCATED,
        AttentionType.MULTIHEAD_REDUCED,
        AttentionType.MULTIHEAD_FULL,
    ]


LAYER_SYMBOLS = {
    "h": AttentionType.MULTIHEAD,  # multi-Head
    "e": AttentionType.LOCAL_EXPERTS,  # Experts
    "m": AttentionType.MEMORY_EFFICIENT,  # Memory
    "s": AttentionType.SPARSE_MULTIHEAD,  # Sparse (Locality sensitive hashing)
    "t": AttentionType.SPARSE_MULTIHEAD_TRUNCATED,  # Using TruncatedDispatcher
    "r": AttentionType.MULTIHEAD_REDUCED,  # Reduced
    "f": AttentionType.MULTIHEAD_FULL,  # Force using full attention
}


@registry.register_model
class AttentionLmMoe(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def model_fn_body_sharded(self, sharded_features):
    # Remove dropout if not training
    hparams = self._hparams
    dp = self._data_parallelism
    if hparams.use_inputs:
      decoder_input = dp(tf.squeeze, sharded_features["inputs"], 2)
      decoder_self_attention_bias = None
    else:
      targets = sharded_features["targets"]
      targets = dp(tf.squeeze, targets, 2)
      (decoder_input, decoder_self_attention_bias, pad_remover) = dp(
          attention_lm_moe_prepare_decoder, targets, hparams)

    def preprocess(x):
      return dp(common_layers.layer_preprocess, x, hparams)

    def postprocess(x, y):
      return dp(common_layers.layer_postprocess, x, y, hparams)

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

    if not hparams.use_inputs:
      # As preprocess and postprocess are called with batch of size one (all
      # batches concatenated), we just make sure that batch_norm is not use (
      # should not either way)
      assert hparams.norm_type != "batch"

      tf.logging.info("Applying Padding Remover for the attention experts")

      dp_remove_pad = functools.partial(
          dp, remove_pad, pad_remover=pad_remover, mode=hparams.mode)
      dp_restore_pad = functools.partial(
          dp, restore_pad, ref_x=x, pad_remover=pad_remover, mode=hparams.mode)
    else:
      # Using identity function: No effect
      dp_remove_pad = lambda x: x
      dp_restore_pad = lambda x: x

    if hparams.attention_exp_factor != 0:
      tf.logging.info("Expand/compress tokens before sending them to experts")
      dp_expand_bc = lambda x: dp(  # pylint: disable=g-long-lambda
          expand_batch_coordinates,
          x,
          hparams.attention_exp_factor)
      dp_expand_x = lambda x: dp(  # pylint: disable=g-long-lambda
          common_attention.deconv_elems_1d,
          x,
          hparams.attention_exp_factor,
          hparams.attention_exp_inputdim)
      dp_compress_x = lambda x, l: dp(  # pylint: disable=g-long-lambda
          common_attention.conv_elems_1d,
          x,
          hparams.attention_exp_factor,
          l)
    else:
      dp_expand_bc = lambda x: x
      dp_expand_x = lambda x: x
      dp_compress_x = lambda x, l: x

    def print_shape(x, suffix, debug=False):
      # To help debugging, print the input/output shapes at inference and eval
      # Inference for long sequences can take a long time, so that's help to
      # see the progession of the generation
      if not debug and hparams.mode == ModeKeys.TRAIN:
        return x
      return tf.Print(x, [tf.shape(x)], "shape_x_{}".format(suffix))

    with tf.name_scope("batch_coordinate_preprocess"):
      batch_coordinate = dp(get_batch_coordinate, x)
      batch_coordinate = dp_remove_pad(batch_coordinate)
      batch_coordinate = dp_expand_bc(batch_coordinate)
      batch_order = dp(get_batch_coordinate, x, axis=-1)
      batch_order = dp_remove_pad(batch_order)
      batch_order = dp_expand_bc(batch_order)

    x = dp(print_shape, x, "in")

    assert hparams.batch_size >= hparams.max_length

    num_hidden_layers = (
        len(hparams.attention_layers) or hparams.num_hidden_layers)
    for layer in xrange(num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):

        # Use the layer type defined in attention_layers
        if hparams.attention_layers:
          attention_type = LAYER_SYMBOLS[hparams.attention_layers[layer]]
        else:
          attention_type = hparams.attention_type

        with tf.variable_scope(
            "attention_{}".format(attention_type)):
          if attention_type in [
              AttentionType.MULTIHEAD, AttentionType.MULTIHEAD_FULL]:
            attention_dot_type = (
                "local_mask_right" if hparams.attention_local else
                "dot_product")
            if attention_type == AttentionType.MULTIHEAD_FULL:
              attention_dot_type = "dot_product"
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
                attention_type=attention_dot_type,
                block_length=hparams.attention_block_length,
                name="decoder_self_attention")
          elif attention_type == AttentionType.SPARSE_MULTIHEAD:
            x_in = preprocess(x)
            x_in = dp_remove_pad(x_in)
            y, loss_experts = dp(
                common_attention.multihead_attention_sparse_dot_prod,
                x_in,
                None,
                None,  # Bias is computed inside
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,

                # Additional parameters
                bi=[common_attention.BatchInfo(
                    coordinates=batch_coordinate[i],
                    order=batch_order[i],  # No future mask
                ) for i in range(dp.n)],
                use_map_fn=hparams.lsh_use_map_fn,
                experts_params=dict(
                    nb_hyperplanes=hparams.lsh_num_hyperplanes,
                ),
            )
            y = dp_restore_pad(y)

            # TODO(avaswani, epot, noam): Do we need to divide by num shards ?
            extra_loss += tf.add_n(loss_experts) / dp.n
          elif attention_type == AttentionType.SPARSE_MULTIHEAD_TRUNCATED:
            x_in = preprocess(x)
            y, loss_experts = dp(
                common_attention.multihead_attention_sparse_truncated,
                x_in,
                None,
                None,  # Bias is computed inside
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,

                # Additional parameters
                bi=[common_attention.BatchInfo(
                    coordinates=batch_coordinate[i],
                    order=batch_order[i],  # No future mask
                ) for i in range(dp.n)],
                mask_right=True,
                experts_params=dict(
                    nb_hyperplanes=hparams.lsh_num_hyperplanes,
                ),
            )

            # TODO(avaswani, epot, noam): Do we need to divide by num shards ?
            extra_loss += tf.add_n(loss_experts) / dp.n
          elif attention_type == AttentionType.MEMORY_EFFICIENT:
            assert hparams.layer_preprocess_sequence == "n"
            y = dp(
                common_attention.multihead_self_attention_memory_efficient,
                x,
                decoder_self_attention_bias,
                hparams.num_heads,
                name="decoder_self_attention")
          elif attention_type == AttentionType.MULTIHEAD_REDUCED:
            y = dp(
                common_attention.multihead_self_attention_reduced,
                preprocess(x),
                factor=hparams.attention_red_factor,
                reduction_type=hparams.attention_reduction_type,
                nonlinearity=hparams.attention_nonlinearity,
                multihead_params=dict(
                    total_key_depth=
                    hparams.attention_key_channels or hparams.hidden_size,
                    total_value_depth=
                    hparams.attention_value_channels or hparams.hidden_size,
                    num_heads=hparams.num_heads,
                    dropout_rate=hparams.attention_dropout,
                ))
          elif attention_type == AttentionType.LOCAL_EXPERTS:
            x_in = preprocess(x)
            x_in = dp_remove_pad(x_in)
            x_in = dp_expand_x(x_in)
            y, loss = dp(
                common_attention.local_expert_attention,
                x_in,
                k=hparams.attention_moe_k,
                loss_coef=hparams.attention_load_balance,
                attention_num_experts=hparams.attention_num_experts,
                train=hparams.mode == ModeKeys.TRAIN,
                batch_coordinate=batch_coordinate,
                mask_right=not hparams.use_inputs,
                split_batch=bool(hparams.attention_split_batch),
                attention_num_head=hparams.attention_num_head,
                attention_kq_size=hparams.attention_kq_size,
                attention_v_size=hparams.attention_v_size)
            y = dp_compress_x(y, x[0].get_shape().as_list()[-1])
            y = dp_restore_pad(y)
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
            additional_conv_params = dict()
            if hparams.use_sepconv:
              additional_conv_params = dict(
                  padding="LEFT",
                  # Parameters copied from the transformer model
                  kernel_size=(3, 1),
                  second_kernel_size=(31, 1),
              )
            y = dp(
                common_layers.conv_hidden_relu,
                preprocess(x),
                hparams.filter_size,
                hparams.hidden_size,
                dropout=hparams.relu_dropout,
                **additional_conv_params
            )
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
    # Because of the shift_right, the <eos> token will be considered as
    # padding. In practice, it doesn't really matter, due to the triangular
    # mask, this token should never be attended.
    pad_remover = expert_utils.PadRemover(targets_pad_mask)

  if hparams.prepend_mode == "prepend_inputs_full_attention":
    decoder_self_attention_bias = (
        common_attention.attention_bias_prepended(targets_pad_mask))
  else:
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias, pad_remover)


@expert_utils.add_name_scope()
def get_batch_coordinate(x, axis=0):
  """Return a flat int32 tensor of shape [1, batch_size*length, 1]."""
  # Compute the batch coordinate before flattening all batches
  batch_coordinate = tf.expand_dims(
      common_attention.coordinate_tensor(tf.shape(x)[:-1], axis=axis), axis=-1)
  return batch_coordinate


@expert_utils.add_name_scope()
def expand_batch_coordinates(bc, length_factor):
  """Duplicate elements of bc by length_factor.

  Args:
    bc (tf.Tensor): int32 tensor of shape [1, length, 1]
    length_factor (int):

  Returns:
    tf.Tensor: of shape [1, length*length_factor, 1] where every elements has
      been duplicated length_factor times.
  """
  assert bc.get_shape().as_list() == [1, None, 1]
  # bc has shape [1, length, 1]
  bc *= tf.constant([[1] * length_factor])
  # bc has shape [1, length, length_factor]
  bc = tf.reshape(bc, [1, -1, 1])
  # bc has shape [1, length*length_factor]
  return bc


@expert_utils.add_name_scope()
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
  # Concatenate all tokens (without padding)
  x = expert_utils.flatten_all_but_last(x)

  # Remove padding for training and eval
  if mode != ModeKeys.PREDICT:
    # This is a hack to allows inference when the <go> token
    # is detected as padding and removed. This works for now because there is
    # no padding at inference.
    x = pad_remover.remove(x)

  x = tf.expand_dims(x, axis=0)  # Now batch_size=1
  return x


@expert_utils.add_name_scope()
def restore_pad(x, ref_x, pad_remover, mode):
  x = tf.squeeze(x, axis=0)
  if mode != ModeKeys.PREDICT:
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
  hparams.shared_embedding_and_softmax_weights = False
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
  # If attention_layers is set, the num_hidden_layers parameter will be ignored
  # and each caracter of the string will correspond to one attention
  # layer type
  hparams.add_hparam("attention_layers", "")
  hparams.add_hparam("attention_type", AttentionType.MULTIHEAD)
  hparams.add_hparam("attention_local", False)
  hparams.add_hparam("attention_moe_k", 2)
  hparams.add_hparam("attention_num_head", 1)
  hparams.add_hparam("attention_num_experts", 16)
  hparams.add_hparam("attention_split_batch", False)
  hparams.add_hparam("attention_red_factor", 3)
  hparams.add_hparam("attention_block_length", 128)
  hparams.add_hparam("attention_reduction_type", "conv")
  # Non linearity for the attention reduction. Either "none", or "silu" (
  # Sigmoid Linear-Unit described in https://arxiv.org/abs/1710.05941)
  hparams.add_hparam("attention_nonlinearity", "none")
  # If attention_exp_factor is set, each input to local_expert_attention (of
  # dimensionality hidden size) is projected into attention_exp_factor smaller
  # inputs, each of dimensionality attention_exp_inputdim. (otherwise
  # attention_exp_inputdim is ignored)
  hparams.add_hparam("attention_exp_factor", 0)
  hparams.add_hparam("attention_exp_inputdim", 128)
  # Key, query and value dimensions for the attention
  hparams.add_hparam("attention_kq_size", 128)
  hparams.add_hparam("attention_v_size", 256)
  # Loss coef for load balancing
  hparams.add_hparam("attention_load_balance", 2e-2)
  # Locality-sensitive hashing params
  hparams.add_hparam("lsh_num_hyperplanes", 4)
  hparams.add_hparam("lsh_use_map_fn", False)

  hparams.add_hparam("use_sepconv", False)
  hparams.add_hparam("diet_experts", False)
  hparams.add_hparam("memory_efficient_ffn", False)
  # if True, we learn a non-autoregressive model from "inputs" to "targets".
  # if False, we learn an autoregressive model to generate "targets"
  hparams.add_hparam("use_inputs", False)
  return hparams


@registry.register_hparams
def attention_lm_moe_base_long_seq():
  """Hyper parameters specifics for long sequence generation."""
  hparams = attention_lm_moe_base()

  hparams.max_length = 0  # max_length == batch_size
  hparams.eval_drop_long_sequences = True
  hparams.min_length_bucket = 256  # Avoid cyclic problems for big batches
  hparams.use_sepconv = True

  return hparams


@registry.register_hparams
def attention_lm_moe_base_ae():
  """Base model with attention expert."""
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_type = AttentionType.LOCAL_EXPERTS

  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 10000
  # According to noam, ("n", "da") seems better for harder-to-learn models
  # hparams.layer_preprocess_sequence = "n"
  # hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def attention_lm_moe_base_local():
  """Base model with attention expert."""
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_local = True
  return hparams


@registry.register_hparams
def attention_lm_moe_base_hybrid():
  """Base model with attention expert."""
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_layers = "hehe"  # Alternate local/expert
  hparams.attention_local = True

  # hparams.layer_preprocess_sequence = "n"
  # hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def attention_lm_hybrid_v2():
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_layers = "hheh"  # Alternate local/expert
  hparams.attention_local = True
  hparams.attention_moe_k = 6

  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def attention_lm_16k():
  hparams = attention_lm_hybrid_v2()
  hparams.batch_size = 16384
  return hparams


@registry.register_hparams
def attention_lm_12k():
  hparams = attention_lm_hybrid_v2()
  hparams.batch_size = 12000
  return hparams


@registry.register_hparams
def attention_lm_11k():
  hparams = attention_lm_hybrid_v2()
  hparams.batch_size = 11500
  return hparams


@registry.register_hparams
def attention_lm_ae_extended():
  """Experiment with the exp_factor params."""
  hparams = attention_lm_moe_base_long_seq()
  hparams.attention_layers = "eeee"
  hparams.attention_local = True
  # hparams.factored_logits=1  # Necessary when the number of expert grow bigger
  hparams.attention_moe_k = 2
  hparams.attention_exp_factor = 4
  # hparams.attention_exp_inputdim = 128

  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  return hparams


@registry.register_hparams
def attention_lm_moe_base_memeff():
  """Base model with attention expert."""
  hparams = attention_lm_moe_base_long_seq()
  hparams.use_sepconv = False

  hparams.diet_experts = True
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.memory_efficient_ffn = True
  hparams.attention_type = AttentionType.MEMORY_EFFICIENT
  hparams.num_heads = 8
  hparams.factored_logits = True
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
  hparams.diet_experts = True
  return hparams


@registry.register_hparams
def attention_lm_moe_memory_efficient():
  """Memory-efficient version."""
  hparams = attention_lm_moe_large()
  hparams.diet_experts = True
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.memory_efficient_ffn = True
  hparams.attention_type = AttentionType.MEMORY_EFFICIENT
  hparams.num_heads = 8
  hparams.factored_logits = True
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
  hparams.shared_embedding_and_softmax_weights = True
  return hparams


@registry.register_hparams
def attention_lm_moe_unscramble_base():
  """Version to use with languagemodel_wiki_scramble1k50."""
  hparams = attention_lm_no_moe_small()
  hparams.use_inputs = True
  hparams.min_length_bucket = 1024
  hparams.max_length = 1024
  hparams.batch_size = 5000
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  return hparams
