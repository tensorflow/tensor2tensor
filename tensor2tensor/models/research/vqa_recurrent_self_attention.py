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
"""Recurrent self attention models for VQA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import vqa_layers
from tensor2tensor.models.research import universal_transformer
from tensor2tensor.models.research import universal_transformer_util
from tensor2tensor.models.research import vqa_attention
from tensor2tensor.utils import registry
# from tensor2tensor.utils import restore_hook

import tensorflow as tf

from tensorflow.contrib.layers.python.layers import utils


@registry.register_model
class VqaRecurrentSelfAttention(vqa_attention.VqaAttentionBaseline):
  """Recurrent Self attention both on image and question."""

  # @staticmethod
  # def train_hooks():
  #   restore_resnet_hook = restore_hook.RestoreHook(
  #       # TODO(zichaoy): hard code the path given static function.
  #       checkpoint_path="/home/zichaoy/resnet_v1_152.ckpt",
  #       new_model_scope="vqa_recurrent_self_attention/body/",
  #       old_model_scope="resnet_v1_152/",
  #   )
  #   return [restore_resnet_hook]

  def body(self, features):
    hp = self.hparams
    # pylint: disable=eval-used
    if hp.image_input_type == "image":
      image_feat = vqa_layers.image_embedding(
          features["inputs"],
          model_fn=eval(hp.image_model_fn),
          trainable=hp.train_resnet,
          is_training=hp.mode == tf.estimator.ModeKeys.TRAIN)
    else:
      image_feat = features["inputs"]

    image_feat = common_layers.flatten4d3d(image_feat)
    image_feat = common_layers.dense(image_feat, hp.hidden_size)
    utils.collect_named_outputs("norms", "image_feat_after_proj",
                                tf.norm(image_feat, axis=-1))

    question = common_layers.flatten4d3d(features["question"])
    utils.collect_named_outputs("norms", "question_embedding",
                                tf.norm(question, axis=-1))
    (encoder_input, encoder_self_attention_bias,
     encoder_decoder_attention_bias) = prepare_image_question_encoder(
         image_feat, question, hp)

    encoder_input = tf.nn.dropout(
        encoder_input, keep_prob=1.-hp.layer_prepostprocess_dropout)

    encoder_output, _ = recurrent_transformer_decoder(
        encoder_input, None, encoder_self_attention_bias, None,
        hp, name="encoder")
    utils.collect_named_outputs(
        "norms", "encoder_output", tf.norm(encoder_output, axis=-1))

    # scale query by sqrt(hidden_size)
    query = tf.get_variable("query", [hp.hidden_size]) * hp.hidden_size **0.5
    query = tf.expand_dims(tf.expand_dims(query, axis=0), axis=0)
    batch_size = common_layers.shape_list(encoder_input)[0]
    query = tf.tile(query, [batch_size, 1, 1])
    query = tf.nn.dropout(
        query, keep_prob=1.-hp.layer_prepostprocess_dropout)

    decoder_output, _ = recurrent_transformer_decoder(
        query, encoder_output, None, encoder_decoder_attention_bias,
        hp, name="decoder")
    utils.collect_named_outputs("norms", "decoder_output",
                                tf.norm(decoder_output, axis=-1))

    norm_tensors = utils.convert_collection_to_dict("norms")
    vqa_layers.summarize_tensors(norm_tensors, tag="norms/")

    # Expand dimension 1 and 2
    return tf.expand_dims(decoder_output, axis=1)


def prepare_image_question_encoder(image_feat, question, hparams):
  """Prepare encoder.

  Args:
    image_feat: a Tensor.
    question: a Tensor.
    hparams: run hyperparameters

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
  """

  encoder_input = tf.concat([image_feat, question], axis=1)
  encoder_padding = common_attention.embedding_to_padding(encoder_input)
  ignore_padding = common_attention.attention_bias_ignore_padding(
      encoder_padding)
  encoder_self_attention_bias = ignore_padding
  encoder_decoder_attention_bias = ignore_padding
  # Usual case - not a packed dataset.
  if hparams.pos == "timing":
    question = common_attention.add_timing_signal_1d(question)
  elif hparams.pos == "emb":
    question = common_attention.add_positional_embedding(
        question, hparams.max_length, "inputs_positional_embedding",
        None)
  encoder_input = tf.concat([image_feat, question], axis=1)

  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def recurrent_transformer_decoder(
    decoder_input,
    encoder_output,
    decoder_self_attention_bias,
    encoder_decoder_attention_bias,
    hparams,
    name="decoder",
    nonpadding=None,
    save_weights_to=None,
    make_image_summary=True):
  """Recurrent decoder function."""
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    ffn_unit = functools.partial(
        # use encoder ffn, since decoder ffn use left padding
        universal_transformer_util.transformer_encoder_ffn_unit,
        hparams=hparams,
        nonpadding_mask=nonpadding)

    attention_unit = functools.partial(
        universal_transformer_util.transformer_decoder_attention_unit,
        hparams=hparams,
        encoder_output=encoder_output,
        decoder_self_attention_bias=decoder_self_attention_bias,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        attention_dropout_broadcast_dims=attention_dropout_broadcast_dims,
        save_weights_to=save_weights_to,
        make_image_summary=make_image_summary)

    x, extra_output = universal_transformer_util.universal_transformer_layer(
        x, hparams, ffn_unit, attention_unit)

    return common_layers.layer_preprocess(x, hparams), extra_output


@registry.register_hparams
def vqa_recurrent_self_attention_base():
  """VQA attention baseline hparams."""
  hparams = universal_transformer.universal_transformer_base()
  hparams.batch_size = 1024
  hparams.use_fixed_batch_size = True
  hparams.weight_decay = 0.
  hparams.clip_grad_norm = 0.
  # use default initializer
  # hparams.initializer = "xavier"
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_normalized_decay")
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate_constant = 7e-4
  hparams.learning_rate_decay_rate = 0.5
  hparams.learning_rate_decay_steps = 50000
  # hparams.dropout = 0.5
  hparams.summarize_grads = True
  hparams.summarize_vars = True

  # not used hparams
  hparams.label_smoothing = 0.1
  hparams.multiply_embedding_mode = "sqrt_depth"

  # add new hparams
  # use raw image as input
  hparams.add_hparam("image_input_type", "feature")
  hparams.add_hparam("image_model_fn", "resnet_v1_152")
  hparams.add_hparam("resize_side", 512)
  hparams.add_hparam("height", 448)
  hparams.add_hparam("width", 448)
  hparams.add_hparam("distort", True)
  hparams.add_hparam("train_resnet", False)

  # question hidden size
  # hparams.hidden_size = 512
  # hparams.filter_size = 1024
  # hparams.num_hidden_layers = 4

  # self attention parts
  # hparams.norm_type = "layer"
  # hparams.layer_preprocess_sequence = "n"
  # hparams.layer_postprocess_sequence = "da"
  # hparams.layer_prepostprocess_dropout = 0.1
  # hparams.attention_dropout = 0.1
  # hparams.relu_dropout = 0.1
  # hparams.add_hparam("pos", "timing")
  # hparams.add_hparam("num_encoder_layers", 0)
  # hparams.add_hparam("num_decoder_layers", 0)
  # hparams.add_hparam("num_heads", 8)
  # hparams.add_hparam("attention_key_channels", 0)
  # hparams.add_hparam("attention_value_channels", 0)
  # hparams.add_hparam("self_attention_type", "dot_product")

  # iterative part
  hparams.transformer_ffn_type = "fc"

  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_small():
  hparams = vqa_recurrent_self_attention_base()
  hparams.learning_rate_constant = 1e-3
  hparams.hidden_size = 512
  hparams.filter_size = 2048
  hparams.num_heads = 8
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_big():
  hparams = vqa_recurrent_self_attention_base()
  hparams.learning_rate_constant = 5e-4
  hparams.hidden_size = 2048
  hparams.filter_size = 8192
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_big_l4():
  hparams = vqa_recurrent_self_attention_big()
  hparams.num_rec_steps = 4
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_highway():
  hparams = vqa_recurrent_self_attention_base()
  hparams.recurrence_type = "highway"
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_gru():
  hparams = vqa_recurrent_self_attention_base()
  hparams.recurrence_type = "gru"
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_l8():
  hparams = vqa_recurrent_self_attention_base()
  hparams.num_rec_steps = 8
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_mix_before_ut():
  hparams = vqa_recurrent_self_attention_base()
  hparams.mix_with_transformer = "before_ut"
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_l4():
  hparams = vqa_recurrent_self_attention_base()
  hparams.num_rec_steps = 4
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_ls2():
  hparams = vqa_recurrent_self_attention_base()
  hparams.label_smoothing = 0.2
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_drop1():
  hparams = vqa_recurrent_self_attention_base()
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def vqa_recurrent_self_attention_drop3():
  hparams = vqa_recurrent_self_attention_base()
  hparams.relu_dropout = 0.3
  hparams.attention_dropout = 0.3
  return hparams
