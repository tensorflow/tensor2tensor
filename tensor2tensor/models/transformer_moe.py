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

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class TransformerMoe(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def model_fn_body_sharded(self, sharded_features):
    hparams = self._hparams
    dp = self._data_parallelism
    targets = sharded_features["targets"]
    inputs = sharded_features["inputs"]
    target_space = sharded_features["target_space_id"]

    inputs = dp(common_layers.flatten4d3d, inputs)
    targets = dp(common_layers.flatten4d3d, targets)

    def preprocess(x):
      return dp(common_layers.layer_preprocess, x, hparams)

    def postprocess(x, y):
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
    extra_loss = 0
    moe_hidden_sizes = [int(s) for s in hparams.moe_hidden_sizes.split(",")]
    expert_fn = expert_utils.ffn_expert_fn(
        hparams.hidden_size, moe_hidden_sizes, hparams.hidden_size)
    x = encoder_input
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("encoder_layer_%d" % layer):
        with tf.variable_scope("encoder_self_attention"):
          y = dp(
              common_attention.multihead_attention,
              preprocess(x),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout)
          x = postprocess(x, y)
        with tf.variable_scope("ffn"):
          if str(layer) in hparams.moe_layers_encoder.split(","):
            y, loss = expert_utils.distributed_moe(
                dp,
                self._ps_devices,
                preprocess(x),
                hparams.mode == tf.estimator.ModeKeys.TRAIN,
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
    encoder_output = preprocess(x)
    x = decoder_input
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("decoder_layer_%d" % layer):
        with tf.variable_scope("decoder_self_attention"):
          y = dp(
              common_attention.multihead_attention,
              preprocess(x),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout)
          x = postprocess(x, y)
        with tf.variable_scope("encoder_decoder_attention"):
          y = dp(
              common_attention.multihead_attention,
              preprocess(x),
              encoder_output,
              encoder_decoder_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout)
          x = postprocess(x, y)
        with tf.variable_scope("ffn"):
          if str(layer) in hparams.moe_layers_decoder.split(","):
            y, loss = expert_utils.distributed_moe(
                dp,
                self._ps_devices,
                preprocess(x),
                hparams.mode == tf.estimator.ModeKeys.TRAIN,
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

  hparams.add_hparam("filter_size", 2048)  # Add new ones like this.
  # attention-related flags
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", int(False))
  # FLAGS RELATED TO MIXTURE-OF-EXPERTS
  # comma-separated list of layer numbers.
  # At each of these layers, we replace the ffn with a mixture of experts.
  hparams.add_hparam("moe_layers_encoder", "2")
  hparams.add_hparam("moe_layers_decoder", "2")
  return hparams


@registry.register_hparams
def transformer_no_moe():
  """Without the mixture of experts (for comparison)."""
  hparams = transformer_moe_base()
  hparams.moe_layers_encoder = ""
  hparams.moe_layers_decoder = ""
  return hparams


@registry.register_hparams
def transformer_moe_1b():
  """1-billion parameter model - requires multi-gpu sync training."""
  hparams = transformer_moe_base()
  hparams.moe_n1 = 128
  hparams.moe_layers_encoder = "1,3"
  hparams.moe_layers_decoder = "1,3"
  return hparams
