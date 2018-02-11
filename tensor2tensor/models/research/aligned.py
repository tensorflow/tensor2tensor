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

"""Single stack of transformations with no masking.

Produces output aligned with inputs.

Configurable using hyperparameters to use some combination of convolutions,
attention, mixtures of experts, etc.

A good problem for this model is languagemodel_wiki_scramble1k50 .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import diet
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

ModeKeys = tf.estimator.ModeKeys  # pylint: disable=invalid-name


def _should_preprocess(layer_type):
  return layer_type not in ["timing", "pos_emb", "att_memory_efficient"]


def _should_postprocess(layer_type):
  return layer_type not in ["timing", "pos_emb"]


@registry.register_model
class Aligned(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  @property
  def use_body_sharded(self):
    return True

  def body_sharded(self, sharded_features):
    # Remove dropout if not training
    hparams = self._hparams
    dp = self._data_parallelism
    x = dp(tf.squeeze, sharded_features["inputs"], 2)

    def preprocess(x):
      return dp(common_layers.layer_preprocess, x, hparams)

    def postprocess(x, y):
      return dp(common_layers.layer_postprocess, x, y, hparams)

    x = dp(tf.nn.dropout, x, 1.0 - hparams.layer_prepostprocess_dropout)
    extra_loss = 0.0
    ffn_hidden_sizes = [int(s) for s in hparams.ffn_hidden_sizes.split(",")]
    moe_hidden_sizes = [int(s) for s in hparams.moe_hidden_sizes.split(",")]
    if hparams.mask_right:

      def _bias(x):
        return common_attention.attention_bias_lower_triangle(
            common_layers.shape_list(x)[1])

      bias = dp(_bias, x)
    else:
      bias = tf.zeros([1, 1, 1, 1])
    if hparams.diet_experts:
      hsize, = moe_hidden_sizes

      def _diet_expert(x):
        return diet.diet_expert(x, hsize, diet.diet_adam_optimizer_params())

      expert_fn = _diet_expert
    else:
      expert_fn = expert_utils.ffn_expert_fn(
          hparams.hidden_size, moe_hidden_sizes, hparams.hidden_size)

    batch_coordinate = dp(get_batch_coordinate, x)

    layers = hparams.layers.strip(",").split(",")
    for layer_num, layer_type in enumerate(layers):
      with tf.variable_scope("%s_%d" % (layer_type, layer_num)):
        if _should_preprocess(layer_type):
          x = preprocess(x)
        if layer_type == "timing":
          y = dp(common_attention.add_timing_signal_nd, x)
        elif layer_type == "pos_emb":
          y = dp(
              common_attention.add_positional_embedding_nd,
              x,
              hparams.max_length,
              name="pos_emb")
        elif layer_type == "att":
          y = dp(
              common_attention.multihead_attention,
              x,
              None,
              bias,  # bias
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout)
        elif layer_type == "att_grouped":
          multiplicative_overhead = (
              hparams.multiplicative_overhead if hparams.mode == ModeKeys.TRAIN
              else hparams.multiplicative_overhead_eval)
          y, loss = dp(
              common_attention.grouped_attention_multihead,
              x,
              x,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              num_groups=hparams.attention_num_groups,
              memory_target_density=hparams.memory_target_density,
              multiplicative_overhead=multiplicative_overhead,
              make_image_summary=hparams.attention_image_summary,
              mask_right=hparams.mask_right,
          )
          extra_loss += tf.add_n(loss) / dp.n
        elif layer_type == "att_memory_efficient":
          assert hparams.layer_preprocess_sequence == "n"
          y = dp(common_attention.multihead_self_attention_memory_efficient, x,
                 bias, hparams.num_heads)
        elif layer_type == "att_local":
          y = dp(
              common_attention.multihead_attention,
              x,
              None,
              None,  # bias
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=("local_mask_right"
                              if hparams.mask_right else "local_unmasked"),
              block_length=hparams.local_attention_window,
              block_width=hparams.local_attention_window)
        elif layer_type == "att_pseudolocal":
          # This is an inefficient implementation of local attention, for the
          # purpose of testing model quality.
          def _pseudolocal_bias(x):
            return common_attention.attention_bias_local(
                common_layers.shape_list(x)[1], hparams.local_attention_window,
                0 if hparams.mask_right else hparams.local_attention_window)

          pseudolocal_bias = dp(_pseudolocal_bias, x)
          y = dp(common_attention.multihead_attention, x, None,
                 pseudolocal_bias, hparams.attention_key_channels or
                 hparams.hidden_size, hparams.attention_value_channels or
                 hparams.hidden_size, hparams.hidden_size, hparams.num_heads,
                 hparams.attention_dropout)
        elif layer_type == "att_local_expert":
          y, loss = dp(
              common_attention.local_expert_attention,
              x,
              k=hparams.attention_moe_k,
              loss_coef=hparams.attention_load_balance,
              attention_num_experts=hparams.attention_num_experts,
              train=hparams.mode == ModeKeys.TRAIN,
              batch_coordinate=batch_coordinate,
              mask_right=hparams.mask_right,
              split_batch=bool(hparams.attention_split_batch),
              attention_kq_size=hparams.attention_kq_size,
              attention_v_size=hparams.attention_v_size)
          # TODO(avaswani, epot, noam): Do we need to divide by num shards ?
          extra_loss += tf.add_n(loss) / dp.n
        elif layer_type == "att_lsh":
          if hparams.lsh_truncated:
            attention_fn = common_attention.multihead_attention_sparse_truncated
          else:
            attention_fn = common_attention.multihead_attention_sparse_dot_prod
          y, loss = dp(
              attention_fn,
              x,
              None,
              None,  # Bias is computed inside
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,

              # Additional parameters
              bi=[
                  common_attention.BatchInfo(
                      coordinates=batch_coordinate[i],
                      order=None,  # No future mask
                  ) for i in range(dp.n)
              ],
              use_map_fn=False,
              experts_params=dict(nb_hyperplanes=4,))
          extra_loss += tf.add_n(loss) / dp.n
        elif layer_type == "moe":
          y, loss = expert_utils.distributed_moe(
              dp,
              self._ps_devices,
              x,
              hparams.mode == ModeKeys.TRAIN,
              input_size=hparams.hidden_size,
              expert_fn=expert_fn,
              num_experts=hparams.moe_num_experts,
              k=hparams.moe_k,
              loss_coef=hparams.moe_loss_coef)
          extra_loss += loss
        elif layer_type == "ffn":
          y = dp(
              expert_utils.ffn_expert_fn(hparams.hidden_size, ffn_hidden_sizes,
                                         hparams.hidden_size),
              dp(expert_utils.flatten_all_but_last, x))
          y = dp(expert_utils.reshape_like, y, x)
        elif layer_type == "conv":
          y = dp(
              common_layers.conv1d,
              x,
              hparams.hidden_size,
              hparams.kernel_height,
              activation=tf.nn.relu,
              padding="SAME",
          )
        else:
          assert False, "unknown sublayer %s" % layer_type
        if _should_postprocess(layer_type):
          x = postprocess(x, y)
        else:
          x = y
    x = preprocess(x)

    decoder_output = dp(tf.expand_dims, x, 2)
    return decoder_output, extra_loss


def get_batch_coordinate(x):
  """Return a flat int32 tensor of shape [1, batch_size*length, 1]."""
  # Compute the batch coordinate before flattening all batches
  batch_coordinate = tf.expand_dims(
      common_attention.coordinate_tensor(
          common_layers.shape_list(x)[:-1], axis=0),
      axis=-1)
  return batch_coordinate


@registry.register_hparams
def aligned_base():
  """Set of hyperparameters.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps (10min): log(ppl)_eval = 2.60
  12.0 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.00

  Returns:
    a hparams object
  """
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.batch_size = 5000
  hparams.max_length = 0
  hparams.min_length_bucket = 1024
  hparams.dropout = 0.0
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.label_smoothing = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 2000
  hparams.initializer_gain = 1.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.shared_embedding_and_softmax_weights = True
  hparams.add_hparam("ffn_hidden_sizes", "2048")  # Add new ones like this.
  hparams.moe_num_experts = 32
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.add_hparam("layers", "timing," + "conv,att,ffn," * 2)

  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  # moe params. local attention moe.
  hparams.add_hparam("attention_local", False)
  hparams.add_hparam("attention_moe_k", 2)
  hparams.add_hparam("attention_num_experts", 16)
  hparams.add_hparam("attention_split_batch", False)
  # Key, query and value dimensions for the attention
  hparams.add_hparam("attention_kq_size", 128)
  hparams.add_hparam("attention_v_size", 256)
  # Loss coef for load balancing
  hparams.add_hparam("attention_load_balance", 2e-2)
  hparams.add_hparam("diet_experts", False)
  hparams.add_hparam("memory_efficient_ffn", False)
  hparams.add_hparam("local_attention_window", 128)
  hparams.add_hparam("attention_num_groups", 8)
  hparams.add_hparam("memory_target_density", 2.0)
  hparams.add_hparam("multiplicative_overhead", 1.25)
  hparams.add_hparam("multiplicative_overhead_eval", 2.0)
  hparams.add_hparam("attention_image_summary", True)
  # LSH params
  hparams.add_hparam("lsh_truncated", True)
  # For testing right-masking.
  # This is not implemented in all layers.
  hparams.add_hparam("mask_right", False)
  return hparams


@registry.register_hparams
def aligned_memory_efficient():
  """Use multihead_self_attention_memory_efficient.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.59
  8.7 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.02

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "timing," + "conv,att_memory_efficient,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_local_expert():
  """Use local_expert_attention.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.72
  10.2 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.27

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "timing," + "conv,att_local_expert,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_grouped():
  """Use local_expert_attention.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.63
  10.2 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.04

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "timing," + "conv,att_grouped,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_local():
  """Use local attention code.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.57
  12.8 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.08

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "timing," + "conv,att_local,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_local_1k():
  """Use local attention code, attend to full sequence.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.57
  7.5 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.00

  Returns:
    a hparams object
  """
  hparams = aligned_local()
  hparams.local_attention_window = 1024
  return hparams


@registry.register_hparams
def aligned_pseudolocal():
  """Use a bias to simulate local attention.  attention radius 128.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.57
  12.0 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.06

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "timing," + "conv,att_pseudolocal,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_pseudolocal_256():
  """Use a bias to simulate local attention.  attentio radius 256.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.56
  12.0 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.05

  Returns:
    a hparams object
  """
  hparams = aligned_pseudolocal()
  hparams.local_attention_window = 256
  return hparams


@registry.register_hparams
def aligned_no_timing():
  """No timing signal.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.75
  12.3 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.39

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "conv,att,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_no_att():
  """No attention at all.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.89
  20.8 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.70

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "conv,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_pos_emb():
  """positional embedding insead of timing signal.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.67
  12.1 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.00

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "pos_emb," + "conv,att,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_moe():
  """mixture of experts instead of ffn.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.62
  6.7 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 1.94

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "timing," + "conv,att,moe," * 2
  return hparams


@registry.register_hparams
def aligned_lsh():
  """Use multihead_attention_sparse_dot_prod.

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.layers = "timing," + "conv,att_lsh,ffn," * 2
  return hparams


@registry.register_hparams
def aligned_8k():
  """version for languagemodel_wiki_scramble8k50.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.93
  1.5 steps/sec on P100

  Returns:
    a hparams object
  """
  hparams = aligned_base()
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def aligned_8k_grouped():
  """version for languagemodel_wiki_scramble8k50.

  languagemodel_wiki_scramble1k50, 1gpu, 7k steps: log(ppl)_eval = 2.92
  3.3 steps/sec on P100
  8gpu (8x batch), 7k steps: log(ppl)_eval = 2.15

  Returns:
    a hparams object
  """
  hparams = aligned_grouped()
  hparams.batch_size = 8192
  # hparams.attention_image_summary = False
  hparams.num_groups = 16
  hparams.multiplicative_overhead = 1.1
  return hparams
