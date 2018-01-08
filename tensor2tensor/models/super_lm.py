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

"""Supercomputer-based language model.

Uses model-parallelism.

Each shard (device) has a similar structure with different weights.
Occasional cross-replica-sum across shards.

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

ModeKeys = tf.estimator.ModeKeys  # pylint: disable=invalid-name


def _embedding(inputs, vocab_size, dense_size):
  embedding_var = tf.get_variable("embedding", [vocab_size, dense_size])
  emb_x = tf.gather(embedding_var, tf.to_int32(inputs))
  emb_x *= dense_size ** 0.5
  return emb_x


@registry.register_model
class SuperLM(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def body(self, features):
    # Remove dropout if not training
    hparams = self._hparams
    ps_devices = self._ps_devices
    assert hparams.num_model_shards % len(ps_devices) == 0
    shards_per_device = hparams.num_model_shards // len(ps_devices)
    model_devices = [ps_devices[i // shards_per_device]
                     for i in xrange(hparams.num_model_shards)]
    print("model_devices = %s" % model_devices)
    mp = expert_utils.Parallelism(model_devices, reuse=False)
    vocab_size = self._problem_hparams.vocabulary["targets"].vocab_size
    # squeeze out channels, heights
    targets = features["targets_raw"]
    targets = tf.squeeze(targets, 3)
    targets = tf.squeeze(targets, 2)
    shifted_targets = common_layers.shift_right_2d(targets)
    # Bypass the symbol modality and use a different embedding on each shard.
    decoder_input = mp(
        _embedding, shifted_targets, vocab_size, hparams.hidden_size)
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
    else:
      targets_position = None

    if hparams.pos == "timing":
      if targets_position is None:
        decoder_input = mp(common_attention.add_timing_signal_1d, decoder_input)
      else:
        decoder_input = mp(
            common_attention.add_timing_signal_1d_given_position,
            decoder_input, targets_position)

    decoder_input = mp(
        tf.nn.dropout, decoder_input,
        1.0 - hparams.layer_prepostprocess_dropout)
    decoder_output, extra_loss = _super_stack(
        decoder_input, decoder_self_attention_bias, hparams, mp)
    # Bypass the symbol modality and compute logits directly.
    # We compute a different set of logits on each shard, and sum them.
    logits = mp(tf.layers.dense, decoder_output, vocab_size, name="logits")
    logits = common_layers.all_reduce_ring(logits, mp)
    logits = mp(tf.multiply, logits, mp.n ** -0.5)
    # We now have identical logits on all shards.
    # Shard 0 gets returned to the estimator.
    logits_shard_0 = logits[0]
    logits_shard_0 = tf.expand_dims(logits_shard_0, 2)
    logits_shard_0 = tf.expand_dims(logits_shard_0, 3)
    # On each device, we compute the loss for a part of the batch.
    # This is faster than computing the whole loss on one shard.
    mp, logits = common_layers.reduce_by_device(mp, logits, lambda l: l[0])
    def _loss_for_shard(logits, targets, shard):
      if mp.n > 1:
        logits = common_layers.approximate_split(logits, mp.n, 0)[shard]
        targets = common_layers.approximate_split(targets, mp.n, 0)[shard]
      return common_layers.padded_cross_entropy(
          logits, targets, hparams.label_smoothing)
    num, denom = mp(_loss_for_shard, logits, targets, range(mp.n))
    # override training loss so that it is not computed externally.
    losses = {"training": tf.add_n(num) / tf.add_n(denom)}
    if extra_loss is not None:
      losses["extra"] = extra_loss
    return logits_shard_0, losses


def _super_stack(inputs,
                 attention_bias,
                 hparams,
                 mp,
                 padding="LEFT"):
  """A stack of super_lm layers.

  Args:
    inputs: a list of Tensors
    attention_bias: list of bias Tensor for self-attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    mp: a Parallelism object
    padding: a string

  Returns:
    y: a list of Tensors
    extra_loss: an optional scalar
  """
  layers = hparams.layers.strip(",").split(",")
  ffn_hidden_sizes = [int(s) for s in hparams.ffn_hidden_sizes.split(",")]
  moe_hidden_sizes = [int(s) for s in hparams.moe_hidden_sizes.split(",")]
  if hparams.diet_experts:
    hsize, = moe_hidden_sizes
    def _diet_expert(x):
      return diet.diet_expert(x, hsize, diet.diet_adam_optimizer_params())
    expert_fn = _diet_expert
  else:
    expert_fn = expert_utils.ffn_expert_fn(
        hparams.hidden_size, moe_hidden_sizes, hparams.hidden_size)
  # scaled_dot_product_attention_with_projections uses a 3d attention bias
  # (no heads), where multihead_attention uses 4d attention bias.
  mix_size = int(hparams.mix_fraction * hparams.hidden_size)
  attention_bias_3d = mp(tf.squeeze, attention_bias, 1)
  accumulator = inputs
  x = inputs
  extra_losses = []
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
            q, x, x, attention_bias_3d)
        x = mp(tf.layers.dense, x, hparams.hidden_size, use_bias=False,
               name="o_transform")
      elif layer_type == "multihead-att":
        # multi-head attention
        x = mp(
            common_attention.multihead_attention,
            x,
            None,
            attention_bias,  # bias
            hparams.attention_key_channels or hparams.hidden_size,
            hparams.attention_value_channels or hparams.hidden_size,
            hparams.hidden_size,
            hparams.num_heads,
            hparams.attention_dropout)
      elif layer_type == "ffn":
        y = mp(
            expert_utils.ffn_expert_fn(
                hparams.hidden_size, ffn_hidden_sizes, hparams.hidden_size),
            mp(expert_utils.flatten_all_but_last, x))
        x = mp(expert_utils.reshape_like, y, x)
      elif layer_type == "conv":
        # convolution
        x = mp(
            common_layers.conv1d,
            x,
            hparams.hidden_size,
            hparams.kernel_height,
            activation=tf.nn.relu,
            padding=padding,
        )
      elif layer_type == "moe":
        # mixture of experts - each model shard has its own local MoE.
        x, loss = mp(
            expert_utils.local_moe,
            x,
            train=hparams.mode == tf.estimator.ModeKeys.TRAIN,
            expert_fn=expert_fn,
            num_experts=hparams.moe_num_experts,
            k=hparams.moe_k,
            loss_coef=hparams.moe_loss_coef)
        extra_losses.extend(loss)
      else:
        assert False, "unknown sublayer %s" % layer_type
  if extra_losses:
    extra_loss = tf.add_n(extra_losses)
  else:
    extra_loss = None
  return x, extra_loss


@registry.register_hparams
def super_lm_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.moe_hidden_sizes = "512"
  hparams.batch_size = 16384
  hparams.max_length = 0
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.label_smoothing = 0.0
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.initializer_gain = 1.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.shared_embedding_and_softmax_weights = False
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  # we only want one data shard.
  hparams.no_data_parallelism = True
  # bypass the symbol modality so that we can use model parallelism.
  hparams.target_modality = "symbol:identity"
  hparams.add_hparam("ffn_hidden_sizes", "512")  # Add new ones like this.
  hparams.add_hparam("mix_fraction", 0.5)
  # attention-related flags
  hparams.add_hparam("num_heads", 4)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam(
      "layers", ("n,att,m,d,a," "n,ffn,m,d,a,") * 4 + "n,ffn,d")
  # Number of model shards - each one has separate parameters.
  # Changing this number invalidates checkpoints.
  hparams.add_hparam("num_model_shards", 8)
  hparams.add_hparam("diet_experts", False)
  return hparams


@registry.register_hparams
def super_lm_conv():
  """Add some convolutions."""
  hparams = super_lm_base()
  hparams.layers = (
      ("n,conv,m,d,a," "n,att,m,d,a," "n,ffn,m,d,a,") * 4 + "n,ffn,d")
  return hparams


@registry.register_hparams
def super_lm_big():
  """Big model."""
  hparams = super_lm_base()
  hparams.hidden_size = 1024
  hparams.ffn_hidden_sizes = "2048"
  return hparams


@registry.register_hparams
def super_lm_low_mix():
  """Less mixuing."""
  hparams = super_lm_base()
  hparams.mix_fraction = 0.125
  return hparams


@registry.register_hparams
def super_lm_high_mix():
  """More mixing."""
  hparams = super_lm_base()
  hparams.mix_fraction = 0.875
  return hparams


@registry.register_hparams
def super_lm_b8k():
  """Smaller batch."""
  hparams = super_lm_base()
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def super_lm_moe():
  """Add mixture of experts with ~1B params."""
  hparams = super_lm_base()
  hparams.layers = (
      ("n,att,m,d,a," "n,moe,m,d,a,") * 4 + "n,ffn,d")
  hparams.moe_num_experts = 32
  hparams.moe_hidden_sizes = "1024"
  return hparams


@registry.register_hparams
def super_lm_moe_h4():
  """Add mixture of experts."""
  hparams = super_lm_moe()
  hparams.layers = (
      ("n,multihead-att,m,d,a," "n,moe,m,d,a,") * 4 + "n,ffn,d")
  return hparams


@registry.register_hparams
def super_lm_moe_4b_diet():
  """Add mixture of experts with ~4B params and diet variables.

  Currently, hangs.  See this issue:
  https://github.com/tensorflow/tensorflow/issues/13351

  Returns:
    a hparams.
  """
  hparams = super_lm_moe()
  hparams.moe_num_experts = 128
  hparams.diet_experts = True
  return hparams
