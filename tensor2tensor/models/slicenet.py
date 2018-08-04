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
"""SliceNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def attention(targets_shifted, inputs_encoded, norm_fn, hparams, bias=None):
  """Complete attention layer with preprocessing."""
  separabilities = [hparams.separability, hparams.separability]
  if hparams.separability < 0:
    separabilities = [hparams.separability - 1, hparams.separability]
  targets_timed = common_layers.subseparable_conv_block(
      common_layers.add_timing_signal(targets_shifted),
      hparams.hidden_size, [((1, 1), (5, 1)), ((4, 1), (5, 1))],
      normalizer_fn=norm_fn,
      padding="LEFT",
      separabilities=separabilities,
      name="targets_time")
  if hparams.attention_type == "transformer":
    targets_timed = tf.squeeze(targets_timed, 2)
    target_shape = tf.shape(targets_timed)
    targets_segment = tf.zeros([target_shape[0], target_shape[1]])
    target_attention_bias = common_attention.attention_bias(
        targets_segment, targets_segment, lower_triangular=True)
    inputs_attention_bias = tf.zeros([
        tf.shape(inputs_encoded)[0], hparams.num_heads,
        tf.shape(targets_segment)[1],
        tf.shape(inputs_encoded)[1]
    ])

    qv = common_attention.multihead_attention(
        targets_timed,
        None,
        target_attention_bias,
        hparams.hidden_size,
        hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        name="self_attention")
    qv = common_attention.multihead_attention(
        qv,
        inputs_encoded,
        inputs_attention_bias,
        hparams.hidden_size,
        hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
        hparams.attention_dropout,
        name="encdec_attention")
    return tf.expand_dims(qv, 2)
  elif hparams.attention_type == "simple":
    targets_with_attention = common_layers.simple_attention(
        targets_timed, inputs_encoded, bias=bias)
    return norm_fn(targets_shifted + targets_with_attention, name="attn_norm")


def multi_conv_res(x, padding, name, layers, hparams, mask=None, source=None):
  """A stack of separable convolution blocks with residual connections."""
  with tf.variable_scope(name):
    padding_bias = None
    if mask is not None:
      padding_bias = (1.0 - mask) * -1e9  # Bias to not attend to padding.
      if padding == "LEFT":  # Do not mask anything when left-padding.
        mask = None
    if (hparams.kernel_scheme in _KERNEL_SCHEMES and
        hparams.dilation_scheme in _DILATION_SCHEMES):
      kernels = _KERNEL_SCHEMES[hparams.kernel_scheme]
      dilations = _DILATION_SCHEMES[hparams.dilation_scheme]
      dilations_and_kernels = list(zip(dilations, kernels))
      dilations_and_kernels1 = dilations_and_kernels[:2]
      dilations_and_kernels2 = dilations_and_kernels[2:]
    else:
      k = (hparams.kernel_height, hparams.kernel_width)
      k2 = (hparams.large_kernel_size, 1)
      dilations_and_kernels1 = [((1, 1), k), ((1, 1), k)]
      dilations_and_kernels2 = [((1, 1), k2), ((4, 4), k2)]
    separabilities1 = [hparams.separability, hparams.separability]
    separabilities2 = [hparams.separability] * len(dilations_and_kernels2)
    if hparams.separability < 0:
      separabilities1 = [hparams.separability - 1, hparams.separability]
      separabilities2 = [
          hparams.separability - i
          for i in reversed(range(len(dilations_and_kernels2)))
      ]

    def norm_fn(x, name):
      with tf.variable_scope(name, default_name="norm"):
        return common_layers.apply_norm(
            x, hparams.norm_type, hparams.hidden_size, hparams.norm_epsilon)

    for layer in range(layers):
      with tf.variable_scope("layer_%d" % layer):
        y = common_layers.subseparable_conv_block(
            x,
            hparams.hidden_size,
            dilations_and_kernels1,
            normalizer_fn=norm_fn,
            padding=padding,
            mask=mask,
            separabilities=separabilities1,
            name="residual1")
        x += common_layers.subseparable_conv_block(
            x + y,
            hparams.hidden_size,
            dilations_and_kernels2,
            normalizer_fn=norm_fn,
            padding=padding,
            mask=mask,
            separabilities=separabilities2,
            name="residual2") + y
        if source is not None and hparams.attention_type != "none":
          x += attention(x, source, norm_fn, hparams, bias=padding_bias)
        if mask is not None:
          x *= mask
    return tf.nn.dropout(x, 1.0 - hparams.dropout)


def rank_loss(sentence_emb, image_emb, margin=0.2):
  """Experimental rank loss, thanks to kkurach@ for the code."""
  with tf.name_scope("rank_loss"):
    # Normalize first as this is assumed in cosine similarity later.
    sentence_emb = tf.nn.l2_normalize(sentence_emb, 1)
    image_emb = tf.nn.l2_normalize(image_emb, 1)
    # Both sentence_emb and image_emb have size [batch, depth].
    scores = tf.matmul(image_emb, tf.transpose(sentence_emb))  # [batch, batch]
    diagonal = tf.diag_part(scores)  # [batch]
    cost_s = tf.maximum(0.0, margin - diagonal + scores)  # [batch, batch]
    cost_im = tf.maximum(
        0.0, margin - tf.reshape(diagonal, [-1, 1]) + scores)  # [batch, batch]
    # Clear diagonals.
    batch_size = tf.shape(sentence_emb)[0]
    empty_diagonal_mat = tf.ones_like(cost_s) - tf.eye(batch_size)
    cost_s *= empty_diagonal_mat
    cost_im *= empty_diagonal_mat
    return tf.reduce_mean(cost_s) + tf.reduce_mean(cost_im)


def similarity_cost(inputs_encoded, targets_encoded):
  """Loss telling to be more similar to your own targets than to others."""
  # This is a first very simple version: handle variable-length by padding
  # to same length and putting everything into batch. In need of a better way.
  x, y = common_layers.pad_to_same_length(inputs_encoded, targets_encoded)
  depth = tf.shape(inputs_encoded)[3]
  x, y = tf.reshape(x, [-1, depth]), tf.reshape(y, [-1, depth])
  return rank_loss(x, y)


def slicenet_middle(inputs_encoded, targets, target_space_emb, mask, hparams):
  """Middle part of slicenet, connecting encoder and decoder."""

  def norm_fn(x, name):
    with tf.variable_scope(name, default_name="norm"):
      return common_layers.apply_norm(x, hparams.norm_type, hparams.hidden_size,
                                      hparams.norm_epsilon)

  # Flatten targets and embed target_space_id.
  targets_flat = tf.expand_dims(common_layers.flatten4d3d(targets), axis=2)
  target_space_emb = tf.tile(target_space_emb,
                             [tf.shape(targets_flat)[0], 1, 1, 1])

  # Use attention from each target to look at input and retrieve.
  targets_shifted = common_layers.shift_right(
      targets_flat, pad_value=target_space_emb)
  if hparams.attention_type == "none":
    targets_with_attention = tf.zeros_like(targets_shifted)
  else:
    inputs_padding_bias = (1.0 - mask) * -1e9  # Bias to not attend to padding.
    targets_with_attention = attention(
        targets_shifted,
        inputs_encoded,
        norm_fn,
        hparams,
        bias=inputs_padding_bias)

  # Positional targets: merge attention and raw.
  kernel = (hparams.kernel_height, hparams.kernel_width)
  targets_merged = common_layers.subseparable_conv_block(
      tf.concat([targets_with_attention, targets_shifted], axis=3),
      hparams.hidden_size, [((1, 1), kernel)],
      normalizer_fn=norm_fn,
      padding="LEFT",
      separability=4,
      name="targets_merge")

  return targets_merged, 0.0


def embed_target_space(target_space_id, hidden_size):
  target_space_emb = common_layers.embedding(
      target_space_id, 32, hidden_size, name="target_space_embedding")
  return tf.reshape(target_space_emb, [1, 1, 1, -1])


def embedding_to_padding(emb):
  """Input embeddings -> is_padding."""
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1, keep_dims=True)
  return tf.to_float(tf.equal(emb_sum, 0.0))


def slicenet_internal(inputs, targets, target_space, hparams, run_decoder=True):
  """The slicenet model, main step used for training."""
  with tf.variable_scope("slicenet"):
    # Project to hidden size if necessary
    if inputs.get_shape().as_list()[-1] != hparams.hidden_size:
      inputs = common_layers.conv_block(
          inputs,
          hparams.hidden_size, [((1, 1), (3, 3))],
          first_relu=False,
          padding="SAME",
          force2d=True)

    # Flatten inputs and encode.
    inputs = tf.expand_dims(common_layers.flatten4d3d(inputs), axis=2)
    inputs_mask = 1.0 - embedding_to_padding(inputs)
    inputs = common_layers.add_timing_signal(inputs)  # Add position info.
    target_space_emb = embed_target_space(target_space, hparams.hidden_size)
    extra_layers = int(hparams.num_hidden_layers * 1.5)
    inputs_encoded = multi_conv_res(
        inputs, "SAME", "encoder", extra_layers, hparams, mask=inputs_mask)
    if not run_decoder:
      return inputs_encoded
    # Do the middle part.
    decoder_start, similarity_loss = slicenet_middle(
        inputs_encoded, targets, target_space_emb, inputs_mask, hparams)
    # Decode.
    decoder_final = multi_conv_res(
        decoder_start,
        "LEFT",
        "decoder",
        hparams.num_hidden_layers,
        hparams,
        mask=inputs_mask,
        source=inputs_encoded)
    return decoder_final, tf.reduce_mean(similarity_loss)


@registry.register_model
class SliceNet(t2t_model.T2TModel):

  def body(self, features):
    target_modality_name = (
        self._problem_hparams.target_modality.name)
    # If we're just predicting a class, there is no use for a decoder.
    run_decoder = "class_label_modality" not in target_modality_name
    return slicenet_internal(
        features["inputs"],
        features["targets"],
        features["target_space_id"],
        self._hparams,
        run_decoder=run_decoder)


_KERNEL_SCHEMES = {
    "3.3.3.3": [(3, 1), (3, 1), (3, 1), (3, 1)],
    "3.7.7.7": [(3, 1), (7, 1), (7, 1), (7, 1)],
    "3.7.15.15": [(3, 1), (7, 1), (15, 1), (15, 1)],
    "3.7.15.31": [(3, 1), (7, 1), (15, 1), (31, 1)],
    "3.7.15.31.63": [(3, 1), (7, 1), (15, 1), (31, 1), (63, 1)],
}
_DILATION_SCHEMES = {
    "1.1.1.1.1": [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
    "1.1.1.1": [(1, 1), (1, 1), (1, 1), (1, 1)],
    "1.1.1.2": [(1, 1), (1, 1), (1, 1), (2, 1)],
    "1.1.2.4": [(1, 1), (1, 1), (2, 1), (4, 1)],
    "1.2.4.8": [(1, 1), (2, 1), (4, 1), (8, 1)],
}


@registry.register_hparams("slicenet_1")
def slicenet_params1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 1024
  hparams.hidden_size = 768
  hparams.dropout = 0.5
  hparams.symbol_dropout = 0.2
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 4
  hparams.kernel_height = 3
  hparams.kernel_width = 1
  hparams.norm_type = "layer"
  hparams.learning_rate_decay_scheme = "exp"
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 3.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  hparams.add_hparam("large_kernel_size", 15)  # New ones are added like this.
  hparams.add_hparam("separability", -2)
  # A dilation scheme, one of _DILATION_SCHEMES.
  hparams.add_hparam("dilation_scheme", "1.1.1.1")
  # A kernel scheme, one of _KERNEL_SCHEMES; overrides large_kernel_size.
  hparams.add_hparam("kernel_scheme", "3.7.15.31")
  hparams.add_hparam("audio_compression", 8)
  # attention-related flags
  hparams.add_hparam("attention_type", "simple")
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("sim_loss_mult", 0.0)  # Try 10.0 for experiments.
  hparams.add_hparam("attention_dropout", 0.2)
  hparams.shared_embedding_and_softmax_weights = True
  return hparams


@registry.register_hparams("slicenet_1noam")
def slicenet_params1_noam():
  """Version with Noam's decay scheme."""
  hparams = slicenet_params1()
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 1.0
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer = "uniform_unit_scaling"
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  return hparams


@registry.register_hparams("slicenet_1tiny")
def slicenet_params1_tiny():
  """Version for fast local runs."""
  hparams = slicenet_params1()
  hparams.attention_type = "simple"
  hparams.separability = 0
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 2
  hparams.batch_size = 512
  hparams.learning_rate_warmup_steps = 200
  return hparams


@registry.register_ranged_hparams("slicenet1")
def slicenet_range1(ranged_hparams):
  """Small range of hyperparameters."""
  rhp = ranged_hparams
  rhp.set_float("clip_grad_norm", 1.0, 10.0, scale=rhp.LOG_SCALE)
  rhp.set_float("learning_rate", 0.02, 1.0, scale=rhp.LOG_SCALE)
  rhp.set_float("optimizer_adam_beta2", 0.995, 0.998)
  rhp.set_float("weight_decay", 1.0, 5.0)
