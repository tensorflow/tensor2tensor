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

"""MultiModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.models import slicenet
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def conv_res_step(x, hparams, padding, mask):
  """One step of convolutions and mid-residual."""
  k = (hparams.kernel_height, hparams.kernel_width)
  k2 = (hparams.large_kernel_size, 1)
  dilations_and_kernels1 = [((1, 1), k), ((1, 1), k)]
  dilations_and_kernels2 = [((1, 1), k2), ((4, 4), k2)]
  with tf.variable_scope("conv_res_step"):
    y = common_layers.subseparable_conv_block(
        x,
        hparams.filter_size,
        dilations_and_kernels1,
        padding=padding,
        mask=mask,
        separabilities=0,
        name="residual1")
    y = tf.nn.dropout(y, 1.0 - hparams.dropout)
    return common_layers.subseparable_conv_block(
        y,
        hparams.hidden_size,
        dilations_and_kernels2,
        padding=padding,
        mask=mask,
        separabilities=0,
        name="residual2")


def residual_fn2(x, y, hparams):
  y = tf.nn.dropout(y, 1.0 - hparams.dropout)
  return common_layers.layer_norm(x + y)


def residual_fn3(x, y, z, hparams):
  y = tf.nn.dropout(y, 1.0 - hparams.dropout)
  z = tf.nn.dropout(z, 1.0 - hparams.dropout)
  return common_layers.layer_norm(x + y + z)


def conv_experts(xs, hparams, dp, ps, padding, mask, layer_id):
  """Convolutions + Mixture-of-Experts layer."""
  del layer_id  # Unused.
  train = hparams.mode == tf.estimator.ModeKeys.TRAIN,
  conv_out = dp(conv_res_step, xs, hparams, padding, mask)
  loss = 0.0
  moe_hidden_sizes = [hparams.filter_size]
  expert_fn = expert_utils.ffn_expert_fn(hparams.hidden_size, moe_hidden_sizes,
                                         hparams.hidden_size)
  moe_out, loss = expert_utils.distributed_moe(
      dp,
      ps,
      xs,
      train,
      input_size=hparams.hidden_size,
      expert_fn=expert_fn,
      num_experts=hparams.moe_num_experts,
      k=hparams.moe_k,
      loss_coef=1.0)
  return dp(residual_fn3, xs, moe_out, conv_out, hparams), loss


def prepare_decoder(targets, target_space_emb):
  """Prepare decoder."""
  decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
  target_space_emb = tf.reshape(target_space_emb, [1, 1, -1])
  target_space_emb = tf.tile(target_space_emb, [tf.shape(targets)[0], 1, 1])
  decoder_input = common_layers.shift_right_3d(
      targets, pad_value=target_space_emb)
  decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_self_attention_bias)


@registry.register_model
class MultiModel(t2t_model.T2TModel):

  def model_fn_body_sharded(self, sharded_features):
    train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
    dp = self._data_parallelism
    hparams = self._hparams

    def project_to_hidden(inputs):
      return common_layers.conv_block(
          inputs,
          hparams.hidden_size, [((1, 1), (3, 3))],
          first_relu=False,
          padding="SAME",
          force2d=True)

    def flatten(inputs):
      return tf.expand_dims(common_layers.flatten4d3d(inputs), axis=2)

    # Project to hidden size if necessary
    if (sharded_features["inputs"][0].get_shape().as_list()[-1] !=
        hparams.hidden_size):
      inputs = dp(project_to_hidden, sharded_features["inputs"])

    inputs = dp(flatten, inputs)
    inputs_pad = dp(slicenet.embedding_to_padding, inputs)
    inputs_mask = dp(lambda x: 1.0 - x, inputs_pad)
    inputs_encoded = dp(common_layers.add_timing_signal, inputs)
    expert_loss = 0.0
    for i in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("enc_layer_%d" % i):
        inputs_encoded, moe_loss = conv_experts(inputs_encoded, hparams, dp,
                                                self._ps_devices, "SAME",
                                                inputs_mask, i)
        expert_loss += tf.reduce_mean(moe_loss) * hparams.moe_loss_coef

    # If we're just predicing a class, there is no use for a decoder, return.
    if isinstance(hparams.problems[self._problem_idx].target_modality,
                  modalities.ClassLabelModality):
      return inputs_encoded, tf.reduce_mean(expert_loss)

    # Decoder.
    inputs3d = dp(tf.squeeze, inputs, 2)
    inputs_encoded3d = dp(tf.squeeze, inputs_encoded, 2)
    encoder_padding = dp(common_attention.embedding_to_padding, inputs3d)
    encoder_attention_bias = dp(common_attention.attention_bias_ignore_padding,
                                encoder_padding)
    targets = dp(common_layers.flatten4d3d, sharded_features["targets"])
    target_space_emb = dp(slicenet.embed_target_space,
                          sharded_features["target_space_id"],
                          hparams.hidden_size)

    (decoder_input, decoder_self_attention_bias) = dp(prepare_decoder, targets,
                                                      target_space_emb)

    moe_hidden_sizes = [int(s) for s in hparams.moe_hidden_sizes.split(",")]
    expert_fn = expert_utils.ffn_expert_fn(
        hparams.hidden_size, moe_hidden_sizes, hparams.hidden_size)
    x = dp(tf.nn.dropout, decoder_input, 1.0 - hparams.dropout)
    for layer in xrange(hparams.num_hidden_layers):
      with tf.variable_scope("dec_layer_%d" % layer):
        with tf.variable_scope("attention"):
          y = dp(
              common_attention.multihead_attention,
              x,
              None,
              decoder_self_attention_bias,
              hparams.hidden_size,
              hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              name="decoder_self_attention")
          z = dp(
              common_attention.multihead_attention,
              y,
              inputs_encoded3d,
              encoder_attention_bias,
              hparams.hidden_size,
              hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              name="encdec_attention")
          x = dp(residual_fn3, x, y, z, hparams)
        with tf.variable_scope("ffn"):
          if str(layer) in hparams.moe_layers.split(","):
            y, moe_loss = expert_utils.distributed_moe(
                dp,
                self._ps_devices,
                x,
                train,
                input_size=hparams.hidden_size,
                expert_fn=expert_fn,
                num_experts=hparams.moe_num_experts,
                k=hparams.moe_k,
                loss_coef=hparams.moe_loss_coef)
            expert_loss += tf.reduce_mean(moe_loss)
          else:
            y = dp(
                common_layers.conv_hidden_relu,
                x,
                hparams.filter_size,
                hparams.hidden_size,
                dropout=hparams.dropout)
          x = dp(residual_fn2, x, y, hparams)

    x = dp(tf.expand_dims, x, 2)
    return x, tf.reduce_mean(expert_loss)


@registry.register_hparams
def multimodel_base():
  """Base parameters for MultiModel."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 512
  hparams.batch_size = 2048
  hparams.num_hidden_layers = 4
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.dropout = 0.1
  hparams.add_hparam("filter_size", 2048)  # Add new ones like this.
  hparams.add_hparam("large_kernel_size", 15)
  hparams.add_hparam("attention_dropout", 0.1)
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("moe_layers", "2")
  hparams.moe_num_experts = 30
  return hparams


@registry.register_hparams
def multimodel_tiny():
  """Tiny parameters for MultiModel."""
  hparams = multimodel_base()
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.batch_size = 512
  hparams.num_hidden_layers = 2
  hparams.moe_n1 = 10
  hparams.moe_layers = "0"
  return hparams
