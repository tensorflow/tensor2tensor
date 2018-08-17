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
"""Attention models for VQA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import vqa_layers
from tensor2tensor.utils import registry
# from tensor2tensor.utils import restore_hook
from tensor2tensor.utils import t2t_model

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_152
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152


@registry.register_model
class VqaAttentionBaseline(t2t_model.T2TModel):
  """Attention baseline model for VQA."""

  # @staticmethod
  # def train_hooks():
  #   restore_resnet_hook = restore_hook.RestoreHook(
  #       # TODO(zichaoy): hard code the path given static function.
  #       checkpoint_path="/home/zichaoy/resnet_v1_152.ckpt",
  #       new_model_scope="vqa_attention_baseline/body/",
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

    if hp.image_feat_size:
      image_feat = common_layers.dense(image_feat, hp.image_feat_size)

    # apply layer normalization and dropout on image_feature
    utils.collect_named_outputs("norms", "image_feat_before_l2",
                                tf.norm(image_feat, axis=-1))
    image_feat = common_layers.l2_norm(image_feat)
    utils.collect_named_outputs("norms", "image_feat_after_l2",
                                tf.norm(image_feat, axis=-1))

    image_feat = tf.nn.dropout(image_feat, keep_prob=1.-hp.dropout)

    query = question_encoder(features["question"], hp)
    utils.collect_named_outputs("norms", "query",
                                tf.norm(query, axis=-1))

    image_ave = attn(image_feat, query, hp)
    utils.collect_named_outputs("norms", "image_ave",
                                tf.norm(image_ave, axis=-1))

    image_question = tf.concat([image_ave, query], axis=1)
    utils.collect_named_outputs("norms", "image_question",
                                tf.norm(image_question, axis=-1))

    image_question = tf.nn.dropout(image_question, 1. - hp.dropout)

    output = mlp(image_question, hp)
    utils.collect_named_outputs("norms", "output",
                                tf.norm(output, axis=-1))

    norm_tensors = utils.convert_collection_to_dict("norms")
    vqa_layers.summarize_tensors(norm_tensors, tag="norms/")

    # Expand dimension 1 and 2
    return tf.expand_dims(tf.expand_dims(output, axis=1), axis=2)

  def infer(self,
            features=None,
            decode_length=1,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Predict."""
    del decode_length, beam_size, top_beams, alpha, use_tpu
    assert features is not None
    logits, _ = self(features)
    assert len(logits.get_shape()) == 5
    logits = tf.squeeze(logits, [1, 2, 3])
    log_probs = common_layers.log_prob_from_logits(logits)
    predictions, scores = common_layers.argmax_with_score(log_probs)
    return {
        "outputs": predictions,
        "scores": scores,
    }


@registry.register_model
class VqaSimpleImageSelfAttention(VqaAttentionBaseline):
  """Attention baseline model for VQA."""

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
    # image feature self attention
    # image_feat = tf.nn.dropout(
    #     image_feat, keep_prob=1.-hp.layer_prepostprocess_dropout)

    # image_feat = image_feat - tf.reduce_mean(
    #     image_feat, axis=-1, keepdims=True)
    # image_feat = tf.nn.l2_normalize(image_feat, -1)
    # utils.collect_named_outputs("norms", "image_feat_after_l2",
    #                             tf.norm(image_feat, axis=-1))

    image_feat = tf.nn.dropout(image_feat, keep_prob=1.-hp.dropout)

    image_feat = image_encoder(image_feat, hp)
    utils.collect_named_outputs("norms", "image_feat_encoded",
                                tf.norm(image_feat, axis=-1))
    image_feat = common_layers.l2_norm(image_feat)
    utils.collect_named_outputs("norms", "image_feat_encoded_l2",
                                tf.norm(image_feat, axis=-1))

    query = question_encoder(features["question"], hp)
    utils.collect_named_outputs("norms", "query",
                                tf.norm(query, axis=-1))

    image_ave = attn(image_feat, query, hp)
    utils.collect_named_outputs("norms", "image_ave",
                                tf.norm(image_ave, axis=-1))

    image_question = tf.concat([image_ave, query], axis=1)
    utils.collect_named_outputs("norms", "image_question",
                                tf.norm(image_question, axis=-1))

    image_question = tf.nn.dropout(image_question, 1. - hp.dropout)

    output = mlp(image_question, hp)
    utils.collect_named_outputs("norms", "output",
                                tf.norm(output, axis=-1))

    norm_tensors = utils.convert_collection_to_dict("norms")
    vqa_layers.summarize_tensors(norm_tensors, tag="norms/")

    # Expand dimension 1 and 2
    return tf.expand_dims(tf.expand_dims(output, axis=1), axis=2)


def image_encoder(image_feat,
                  hparams,
                  name="image_encoder",
                  save_weights_to=None,
                  make_image_summary=True):
  """A stack of self attention layers."""

  x = image_feat
  with tf.variable_scope(name):
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = vqa_layers.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              None,
              hparams.attention_key_channels or hparams.image_hidden_size,
              hparams.attention_value_channels or hparams.image_hidden_size,
              hparams.image_hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              save_weights_to=save_weights_to,
              max_relative_position=None,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=None,
              max_length=None,
              vars_3d=False,
              scale_otproduct=hparams.scale_dotproduct)
          utils.collect_named_outputs("norms", "image_feat_self_attention",
                                      tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs(
              "norms", "image_feat_self_attention_zero_add",
              tf.norm(x, axis=-1))
        with tf.variable_scope("ffn"):
          y = common_layers.dense_relu_dense(
              common_layers.layer_preprocess(x, hparams),
              hparams.image_filter_size,
              hparams.image_hidden_size,
              dropout=hparams.relu_dropout,
              dropout_broadcast_dims=None)
          utils.collect_named_outputs("norms", "image_feat_ffn",
                                      tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs("norms", "image_feat_ffn_zero_add",
                                      tf.norm(x, axis=-1))
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def _get_rnn_cell(hparams):
  if hparams.rnn_type == "lstm":
    rnn_cell = tf.contrib.rnn.BasicLSTMCell
  elif hparams.rnn_type == "lstm_layernorm":
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell
  return tf.contrib.rnn.DropoutWrapper(
      rnn_cell(hparams.hidden_size),
      output_keep_prob=1.0-hparams.dropout)


def question_encoder(question, hparams, name="encoder"):
  """Question encoder, run LSTM encoder and get the last output as encoding."""
  with tf.variable_scope(name, "encoder", values=[question]):
    question = common_layers.flatten4d3d(question)
    padding = common_attention.embedding_to_padding(question)
    length = common_attention.padding_to_length(padding)

    max_question_length = hparams.max_question_length
    question = question[:, :max_question_length, :]
    actual_question_length = common_layers.shape_list(question)[1]
    length = tf.minimum(length, max_question_length)
    padding = [[0, 0],
               [0, max_question_length-actual_question_length],
               [0, 0]]
    question = tf.pad(question, padding)
    question_shape = question.get_shape().as_list()
    question_shape[1] = max_question_length
    question.set_shape(question_shape)

    # apply tanh dropout on question embedding
    question = tf.tanh(question)
    question = tf.nn.dropout(question, keep_prob=1.-hparams.dropout)

    question = [question[:, i, :] for i in range(max_question_length)]

    # rnn_layers = [_get_rnn_cell(hparams)
    #               for _ in range(hparams.num_rnn_layers)]
    # rnn_multi_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
    rnn_cell = _get_rnn_cell(hparams)
    # outputs, _ = tf.nn.dynamic_rnn(
    #     rnn_cell, question, length, dtype=tf.float32)
    _, state = tf.nn.static_rnn(rnn_cell, question, sequence_length=length,
                                dtype=tf.float32)
    # outputs = [tf.expand_dims(output, axis=1) for output in outputs]
    # outputs = tf.concat(outputs, axis=1)

    # utils.collect_named_outputs("vqa_attention_debug", "question_output",
    #                             outputs)
    # utils.collect_named_outputs("vqa_attention_debug", "question_state",
    #                             state.h)

    # batch_size = common_layers.shape_list(outputs)[0]
    # row_indices = tf.range(batch_size)
    # # length - 1 as index
    # indices = tf.transpose([row_indices, tf.maximum(length-1, 0)])
    # last_output = tf.gather_nd(outputs, indices)

    # utils.collect_named_outputs("vqa_attention_debug",
    #                             "question_final_output", last_output)

  return state.h


def attn(image_feat, query, hparams, name="attn"):
  """Attention on image feature with question as query."""
  with tf.variable_scope(name, "attn", values=[image_feat, query]):
    attn_dim = hparams.attn_dim
    num_glimps = hparams.num_glimps
    num_channels = common_layers.shape_list(image_feat)[-1]
    if len(common_layers.shape_list(image_feat)) == 4:
      image_feat = common_layers.flatten4d3d(image_feat)
    query = tf.expand_dims(query, 1)
    image_proj = common_attention.compute_attention_component(
        image_feat, attn_dim, name="image_proj")
    query_proj = common_attention.compute_attention_component(
        query, attn_dim, name="query_proj")
    h = tf.nn.relu(image_proj + query_proj)
    h_proj = common_attention.compute_attention_component(
        h, num_glimps, name="h_proj")
    p = tf.nn.softmax(h_proj, axis=1)
    image_ave = tf.matmul(image_feat, p, transpose_a=True)
    image_ave = tf.reshape(image_ave, [-1, num_channels*num_glimps])

    return image_ave


def mlp(feature, hparams, name="mlp"):
  """Multi layer perceptron with dropout and relu activation."""
  with tf.variable_scope(name, "mlp", values=[feature]):
    num_mlp_layers = hparams.num_mlp_layers
    mlp_dim = hparams.mlp_dim
    for _ in range(num_mlp_layers):
      feature = common_layers.dense(feature, mlp_dim, activation=tf.nn.relu)
      feature = tf.nn.dropout(feature, keep_prob=1.-hparams.dropout)
    return feature


@registry.register_hparams
def vqa_attention_base():
  """VQA attention baseline hparams."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 128
  hparams.use_fixed_batch_size = True,
  hparams.optimizer = "Adam"
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.optimizer_adam_epsilon = 1e-8
  hparams.weight_decay = 0.
  hparams.clip_grad_norm = 0.
  hparams.initializer = "xavier"
  hparams.learning_rate = 0.5
  hparams.learning_rate_schedule = "legacy"
  hparams.learning_rate_warmup_steps = 0
  hparams.learning_rate_decay_scheme = "exp"
  hparams.learning_rate_decay_rate = 0.5
  hparams.learning_rate_decay_steps = 50000
  hparams.dropout = 0.5
  hparams.summarize_grads = True
  hparams.summarize_vars = True

  # not used hparams
  hparams.label_smoothing = 0.
  hparams.multiply_embedding_mode = ""

  # add new hparams
  # preprocess
  hparams.add_hparam("resize_side", 512)
  hparams.add_hparam("height", 448)
  hparams.add_hparam("width", 448)
  hparams.add_hparam("distort", True)

  hparams.add_hparam("train_resnet", False)
  hparams.add_hparam("rnn_type", "lstm")
  hparams.add_hparam("num_rnn_layers", 1)
  hparams.add_hparam("max_question_length", 15)
  # lstm hidden size
  hparams.hidden_size = 512

  hparams.add_hparam("attn_dim", 512)
  hparams.add_hparam("num_glimps", 2)

  hparams.add_hparam("num_mlp_layers", 1)
  hparams.add_hparam("mlp_dim", 1024)

  hparams.add_hparam("image_input_type", "image")
  hparams.add_hparam("image_model_fn", "resnet_v1_152")
  hparams.add_hparam("image_feat_size", 0)

  # self attention parts
  hparams.norm_type = "layer"
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.image_hidden_size = 2048
  hparams.add_hparam("num_encoder_layers", 1)
  # Attention-related flags.
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("image_filter_size", 1024)
  hparams.add_hparam("self_attention_type", "dot_product")
  hparams.add_hparam("scale_dotproduct", True)

  return hparams


@registry.register_hparams
def vqa_attention_feature_base():
  hparams = vqa_attention_base()
  hparams.image_input_type = "feature"
  return hparams


@registry.register_hparams
def vqa_attention_feature_lstmlayernorm():
  hparams = vqa_attention_feature_base()
  hparams.rnn_type = "lstm_layernorm"
  return hparams


@registry.register_hparams
def vqa_attention_feature_initializer():
  hparams = vqa_attention_feature_base()
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch512():
  hparams = vqa_attention_feature_base()
  hparams.batch_size = 512
  return hparams


@registry.register_hparams
def vqa_attention_feature_hidden1024():
  hparams = vqa_attention_feature_base()
  hparams.hidden_size = 1024
  return hparams


@registry.register_hparams
def vqa_attention_feature_imagefeat512():
  hparams = vqa_attention_feature_base()
  hparams.image_feat_size = 512
  return hparams


@registry.register_hparams
def vqa_attention_feature_imagefeat1024():
  hparams = vqa_attention_feature_base()
  hparams.image_feat_size = 1024
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024_lstmlayernorm():
  hparams = vqa_attention_feature_lstmlayernorm()
  hparams.batch_size = 1024
  return hparams


@registry.register_hparams
def vqa_attention_numglimps1():
  hparams = vqa_attention_base()
  hparams.num_glimps = 1
  return hparams


@registry.register_hparams
def vqa_attention_feature_numglimps1():
  hparams = vqa_attention_feature_base()
  hparams.num_glimps = 1
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024_numglimps1():
  hparams = vqa_attention_feature_numglimps1()
  hparams.batch_size = 1024
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024():
  hparams = vqa_attention_feature_base()
  hparams.batch_size = 1024
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024_dnz():
  hparams = vqa_attention_feature_batch1024()
  hparams.layer_preprocess_sequence = ""
  hparams.layer_postprocess_sequence = "dnz"
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024_dnz_l2():
  hparams = vqa_attention_feature_batch1024_dnz()
  hparams.norm_type = "l2"
  return hparams


@registry.register_hparams
def vqa_attention_feature_dnz():
  hparams = vqa_attention_feature_base()
  hparams.layer_preprocess_sequence = ""
  hparams.layer_postprocess_sequence = "dnz"
  return hparams


@registry.register_hparams
def vqa_attention_feature_dna():
  hparams = vqa_attention_feature_base()
  hparams.layer_preprocess_sequence = ""
  hparams.layer_postprocess_sequence = "dna"
  return hparams


@registry.register_hparams
def vqa_attention_feature_dnz_noscaledp():
  hparams = vqa_attention_feature_dnz()
  hparams.scale_dotproduct = False
  return hparams


@registry.register_hparams
def vqa_attention_feature_dnz_l2():
  hparams = vqa_attention_feature_dnz()
  hparams.norm_type = "l2"
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024_dnz_noscaledp():
  hparams = vqa_attention_feature_batch1024_dnz()
  hparams.scale_dotproduct = False
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024_drop01():
  hparams = vqa_attention_feature_batch1024()
  hparams.layer_prepostprocess_dropout = 0.1
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024_drop01_dna():
  hparams = vqa_attention_feature_batch1024_drop01()
  hparams.layer_preprocess_sequence = ""
  hparams.layer_postprocess_sequence = "dna"
  return hparams


@registry.register_hparams
def vqa_attention_drop01_dna():
  hparams = vqa_attention_feature_batch1024_drop01_dna()
  hparams.batch_size = 128
  hparams.image_input_type = "image"
  return hparams


@registry.register_hparams
def vqa_attention_feature_batch1024_drop01_dna_concat():
  hparams = vqa_attention_feature_batch1024_drop01()
  hparams.layer_preprocess_sequence = ""
  hparams.layer_postprocess_sequence = "dna"
  hparams.num_glimps = 1
  return hparams


@registry.register_hparams
def vqa_attention_feature_nonormalization():
  hparams = vqa_attention_feature_base()
  hparams.layer_preprocess_sequence = ""
  return hparams


@registry.register_ranged_hparams
def vqa_attention_base_range(rhp):
  """Small range of hyperparameters."""
  # After starting from base, set intervals for some parameters.
  rhp.set_float("learning_rate", 0.1, 1.0, scale=rhp.LOG_SCALE)
  rhp.set_float("clip_grad_norm", 0.1, 10, scale=rhp.LOG_SCALE)
  rhp.set_discrete("batch_size", [128, 256, 512, 1024])
  rhp.set_float("weight_decay", 0.0, 1e-4)
  rhp.set_categorical("rnn_type", ["lstm", "lstm_layernorm"])
