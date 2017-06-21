# Copyright 2017 Google Inc.
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

from tensor2tensor.models import common_layers
from tensor2tensor.models import modalities
from tensor2tensor.models import slicenet
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def experts(xs, moe_n1, moe_n2, hidden_size, filter_size, dp, ps, train):
  """Mixture-of-Experts layer."""
  # Set up the hyperparameters for the gating networks.
  primary_gating_hp = eu.NoisyTopKGatingParams()
  primary_gating_hp.num_experts = moe_n1
  if moe_n2:
    # Hierarchical MoE containing moe_n1 groups of moe_n2 experts.
    assert moe_n2 > 1
    secondary_gating_hp = eu.NoisyTopKGatingParams()
    secondary_gating_hp.num_experts = moe_n2
  else:
    # Flat mixture of moe_n1 experts.
    secondary_gating_hp = None
  # Set up the hyperparameters for the expert networks.
  # Each expert contains a hidden RELU layer of size filter_size
  expert_hp = eu.FeedForwardExpertParams()
  expert_hp.hidden_layer_sizes = [filter_size]
  # Create the mixture of experts.
  moe = eu.DistributedMixtureOfExperts(primary_gating_hp, secondary_gating_hp,
                                       expert_hp, hidden_size, hidden_size, ps,
                                       "moe")
  # MoE expects input tensors to be 2d.  Flatten out spatial dimensions.
  xs_2d = dp(tf.reshape, xs, [[-1, hidden_size]] * dp.n)
  # Call the MoE
  moe_out_2d, importance, load, _, _ = moe.Eval(
      dp.devices, xs_2d, train, summaries=False, identifiers=None)
  # Reshape the output to the original shape.
  moe_out = dp(tf.reshape, moe_out_2d, dp(tf.shape, xs))
  # These losses encourage equal load on the different experts.
  loss = eu.CVSquared(importance) + eu.CVSquared(load)

  # Apply residual and normalize.
  def add_and_normalize(x, y):
    return common_layers.layer_norm(x + y, hidden_size, name="moe_norm")

  return dp(add_and_normalize, xs, moe_out), loss


@registry.register_model
class MultiModel(t2t_model.T2TModel):

  def model_fn_body_sharded(self, sharded_features, train):
    dp = self._data_parallelism
    hparams = self._hparams
    targets = sharded_features["targets"]

    def flatten(inputs):
      return tf.expand_dims(common_layers.flatten4d3d(inputs), axis=2)

    inputs = dp(flatten, sharded_features["inputs"])

    # Encode inputs.
    def encode_half(inputs, inputs_mask, hparams):
      # Add timing and encode.
      inputs = common_layers.add_timing_signal(inputs)
      return slicenet.multi_conv_res(inputs, "SAME", "encoder1",
                                     hparams.num_hidden_layers // 2,
                                     hparams, train, mask=inputs_mask)

    target_space_emb = dp(slicenet.embed_target_space,
                          sharded_features["target_space_id"],
                          hparams.hidden_size)
    inputs_pad = dp(slicenet.embedding_to_padding, inputs)
    inputs_mask = dp(lambda x: 1.0 - x, inputs_pad)
    inputs_encoded = dp(encode_half, inputs, inputs_mask, hparams)
    with tf.variable_scope("experts_enc"):
      inputs_encoded, expert_loss = experts(
          inputs_encoded, hparams.moe_n1, hparams.moe_n2, hparams.hidden_size,
          hparams.hidden_size, dp, self._ps_devices, train)
      expert_loss *= hparams.moe_loss_coef
    inputs_encoded = dp(
        slicenet.multi_conv_res, inputs_encoded, "SAME",
        "encoder2", hparams.num_hidden_layers, hparams, train,
        mask=inputs_mask)

    # If we're just predicing a class, there is no use for a decoder, return.
    if isinstance(hparams.problems[self._problem_idx].target_modality,
                  modalities.ClassLabelModality):
      return inputs_encoded, tf.reduce_mean(expert_loss)

    # Do the middle part.
    decoder_start, similarity_loss = dp(
        slicenet.slicenet_middle, inputs_encoded, targets,
        target_space_emb, inputs_mask, hparams, train)

    # Decode.
    decoder_half = dp(
        slicenet.multi_conv_res,
        decoder_start,
        "LEFT",
        "decoder1",
        hparams.num_hidden_layers // 2,
        hparams,
        train,
        mask=inputs_mask,
        source=inputs_encoded)
    with tf.variable_scope("experts_dec"):
      decoder_half, expert_dec_loss = experts(
          decoder_half, hparams.moe_n1, hparams.moe_n2, hparams.hidden_size,
          hparams.hidden_size, dp, self._ps_devices, train)
      expert_loss += expert_dec_loss * hparams.moe_loss_coef
    decoder_final = dp(
        slicenet.multi_conv_res,
        decoder_half,
        "LEFT",
        "decoder2",
        hparams.num_hidden_layers // 2,
        hparams,
        train,
        mask=inputs_mask,
        source=inputs_encoded)

    total_loss = tf.reduce_mean(expert_loss) + tf.reduce_mean(similarity_loss)
    return decoder_final, total_loss


@registry.register_hparams("multimodel_1p8")
def multimodel_params1_p8():
  """Version for eight problem runs."""
  hparams = slicenet.slicenet_params1()
  hparams.problem_choice = "distributed"
  hparams.attention_type = "simple"  # TODO(lukaszkaiser): add transformer.
  hparams.hidden_size = 1536
  hparams.moe_n1 = 120
  hparams.shared_embedding_and_softmax_weights = int(False)
  hparams.dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.learning_rate_decay_scheme = "exp500k"
  return hparams
