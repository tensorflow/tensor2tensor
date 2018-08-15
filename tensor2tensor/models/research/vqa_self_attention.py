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
"""Self attention models for VQA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import vqa_layers
from tensor2tensor.models.research import vqa_attention
from tensor2tensor.utils import registry
# from tensor2tensor.utils import restore_hook

import tensorflow as tf

from tensorflow.contrib.layers.python.layers import utils


@registry.register_model
class VqaSelfAttention(vqa_attention.VqaAttentionBaseline):
  """Self attention both on image and question."""

  # @staticmethod
  # def train_hooks():
  #   restore_resnet_hook = restore_hook.RestoreHook(
  #       # TODO(zichaoy): hard code the path given static function.
  #       checkpoint_path="/home/zichaoy/resnet_v1_152.ckpt",
  #       new_model_scope="vqa_self_attention/body/",
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
    image_hidden_size = hp.image_hidden_size or hp.hidden_size
    if hp.image_feat_preprocess_proj:
      image_feat = common_layers.dense(image_feat, image_hidden_size)
      utils.collect_named_outputs("norms", "image_feat_after_proj",
                                  tf.norm(image_feat, axis=-1))
    else:
      assert image_hidden_size == 2048

    image_feat = tf.nn.dropout(
        image_feat, keep_prob=1.-hp.layer_prepostprocess_dropout)

    if hp.image_feat_encode:
      image_feat = image_encoder(image_feat, hp)
      utils.collect_named_outputs("norms", "image_feat_encoded",
                                  tf.norm(image_feat, axis=-1))
    else:
      image_feat = common_layers.layer_norm(image_feat)
      utils.collect_named_outputs("norms", "image_feat_after_layer",
                                  tf.norm(image_feat, axis=-1))

    question = common_layers.flatten4d3d(features["question"])
    utils.collect_named_outputs("norms", "question_embedding",
                                tf.norm(question, axis=-1))
    question, question_self_attention_bias = prepare_question_encoder(
        question, hp)
    question = tf.nn.dropout(
        question, keep_prob=1.-hp.layer_prepostprocess_dropout)
    query = question_encoder(question, question_self_attention_bias, hp)
    utils.collect_named_outputs(
        "norms", "query_encode", tf.norm(query, axis=-1))
    query = (query + tf.expand_dims(
        tf.squeeze(question_self_attention_bias, [1, 2]), axis=2))
    query = tf.reduce_max(query, axis=1)
    utils.collect_named_outputs(
        "norms", "query_maxpool", tf.norm(query, axis=-1))

    # query = common_layers.l2_norm(query)
    # utils.collect_named_outputs("norms", "query_after_l2",
    #                             tf.norm(query, axis=-1))

    image_ave = attn(image_feat, query, hp)
    utils.collect_named_outputs("norms", "image_ave",
                                tf.norm(image_ave, axis=-1))

    if hp.multimodal_combine == "concat":
      image_question = tf.concat([image_ave, query], axis=1)
    elif hp.multimodal_combine == "sum":
      image_question = image_ave + query
    elif hp.multimodal_combine == "product":
      image_question = image_ave * query

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


@registry.register_model
class VqaCombinedSelfAttention(VqaSelfAttention):
  """Combined Self attention both on image and question."""

  # @staticmethod
  # def train_hooks():
  #   restore_resnet_hook = restore_hook.RestoreHook(
  #       # TODO(zichaoy): hard code the path given static function.
  #       checkpoint_path="/home/zichaoy/resnet_v1_152.ckpt",
  #       new_model_scope="vqa_combined_self_attention/body/",
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
    image_hidden_size = hp.hidden_size
    image_feat = common_layers.dense(image_feat, image_hidden_size)
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
    encoder_output = image_question_encoder(
        encoder_input, encoder_self_attention_bias, hp)
    utils.collect_named_outputs(
        "norms", "encoder_output", tf.norm(encoder_output, axis=-1))

    # scale query by sqrt(hidden_size)
    query = tf.get_variable("query", [hp.hidden_size]) * hp.hidden_size **0.5
    query = tf.expand_dims(tf.expand_dims(query, axis=0), axis=0)
    batch_size = common_layers.shape_list(encoder_input)[0]
    query = tf.tile(query, [batch_size, 1, 1])
    query = tf.nn.dropout(
        query, keep_prob=1.-hp.layer_prepostprocess_dropout)

    decoder_output = decoder(
        query, encoder_output, None, encoder_decoder_attention_bias, hp)
    utils.collect_named_outputs("norms", "decoder_output",
                                tf.norm(decoder_output, axis=-1))

    norm_tensors = utils.convert_collection_to_dict("norms")
    vqa_layers.summarize_tensors(norm_tensors, tag="norms/")

    # Expand dimension 1 and 2
    return tf.expand_dims(decoder_output, axis=1)


@registry.register_model
class VqaIterativeCombinedSelfAttention(VqaSelfAttention):
  """Combined Self attention both on image and question."""

  # @staticmethod
  # def train_hooks():
  #   restore_resnet_hook = restore_hook.RestoreHook(
  #       # TODO(zichaoy): hard code the path given static function.
  #       checkpoint_path="/home/zichaoy/resnet_v1_152.ckpt",
  #       new_model_scope="vqa_combined_self_attention/body/",
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
    image_hidden_size = hp.hidden_size
    image_feat = common_layers.dense(image_feat, image_hidden_size)
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

    # scale query by sqrt(hidden_size)
    query = tf.get_variable("query", [hp.hidden_size]) * hp.hidden_size **0.5
    query = tf.expand_dims(tf.expand_dims(query, axis=0), axis=0)
    batch_size = common_layers.shape_list(encoder_input)[0]
    query = tf.tile(query, [batch_size, 1, 1])
    query = tf.nn.dropout(
        query, keep_prob=1.-hp.layer_prepostprocess_dropout)

    decoder_output = iterative_encoder_decoder(
        encoder_input,
        encoder_self_attention_bias,
        encoder_decoder_attention_bias,
        query,
        hp)

    utils.collect_named_outputs("norms", "decoder_output",
                                tf.norm(decoder_output, axis=-1))

    norm_tensors = utils.convert_collection_to_dict("norms")
    vqa_layers.summarize_tensors(norm_tensors, tag="norms/")

    # Expand dimension 1 and 2
    return tf.expand_dims(decoder_output, axis=1)


def image_encoder(image_feat,
                  hparams,
                  name="image_encoder",
                  save_weights_to=None,
                  make_image_summary=True):
  """A stack of self attention layers."""

  x = image_feat
  image_hidden_size = hparams.image_hidden_size or hparams.hidden_size
  image_filter_size = hparams.image_filter_size or hparams.filter_size
  with tf.variable_scope(name):
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = vqa_layers.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              None,
              hparams.attention_key_channels or image_hidden_size,
              hparams.attention_value_channels or image_hidden_size,
              image_hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.image_self_attention_type,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              scale_dotproduct=hparams.scale_dotproduct,
          )
          utils.collect_named_outputs(
              "norms", "image_feat_self_attention_%d"%(layer),
              tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs(
              "norms", "image_feat_self_attention_postprocess_%d"%(layer),
              tf.norm(x, axis=-1))
        with tf.variable_scope("ffn"):
          y = common_layers.dense_relu_dense(
              common_layers.layer_preprocess(x, hparams),
              image_filter_size,
              image_hidden_size,
              dropout=hparams.relu_dropout,
          )
          utils.collect_named_outputs(
              "norms", "image_feat_ffn_%d"%(layer), tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs(
              "norms", "image_feat_ffn_postprocess_%d"%(layer),
              tf.norm(x, axis=-1))
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def prepare_question_encoder(inputs, hparams):
  """Prepare question encoder.

  Args:
    inputs: a Tensor.
    hparams: run hyperparameters

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
  """
  encoder_input = inputs
  # Usual case - not a packed dataset.
  encoder_padding = common_attention.embedding_to_padding(encoder_input)
  ignore_padding = common_attention.attention_bias_ignore_padding(
      encoder_padding)
  encoder_self_attention_bias = ignore_padding
  if hparams.pos == "timing":
    encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  elif hparams.pos == "emb":
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, hparams.max_length, "inputs_positional_embedding",
        None)
  return (encoder_input, encoder_self_attention_bias)


def question_encoder(question,
                     question_self_attention_bias,
                     hparams,
                     name="question_encoder",
                     save_weights_to=None,
                     make_image_summary=True):
  """A stack of self attention layers."""
  x = question
  with tf.variable_scope(name):
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = vqa_layers.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              question_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.question_self_attention_type,
              block_length=hparams.block_length,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              scale_dotproduct=hparams.scale_dotproduct,
          )
          utils.collect_named_outputs(
              "norms", "query_self_attention_%d"%(layer),
              tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs(
              "norms", "query_self_attention_postprocess_%d"%(layer),
              tf.norm(x, axis=-1))
        with tf.variable_scope("ffn"):
          y = common_layers.dense_relu_dense(
              common_layers.layer_preprocess(x, hparams),
              hparams.filter_size,
              hparams.hidden_size,
              dropout=hparams.relu_dropout,
              )
          utils.collect_named_outputs(
              "norms", "query_ffn_%d"%(layer), tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs(
              "norms", "query_ffn_postprocess_%d"%(layer),
              tf.norm(x, axis=-1))
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def attn(image_feat,
         query,
         hparams,
         name="attn",
         save_weights_to=None,
         make_image_summary=True):
  """Attention on image feature with question as query."""
  with tf.variable_scope(name, "attn", values=[image_feat, query]):
    total_key_depth = hparams.attention_key_channels or hparams.hidden_size
    total_value_depth = hparams.attention_value_channels or hparams.hidden_size
    num_heads = hparams.num_heads
    query = tf.expand_dims(query, 1)
    q, k, v = common_attention.compute_qkv(
        query,
        image_feat,
        total_key_depth,
        total_value_depth,
    )
    q = common_attention.split_heads(q, num_heads)
    k = common_attention.split_heads(k, num_heads)
    v = common_attention.split_heads(v, num_heads)

    if hparams.scale_dotproduct:
      key_depth_per_head = total_key_depth // num_heads
      q *= key_depth_per_head**-0.5

    # image_feat is input as v
    x = common_attention.dot_product_attention(
        q, k, v, None,
        dropout_rate=hparams.attention_dropout,
        image_shapes=None,
        save_weights_to=save_weights_to,
        make_image_summary=make_image_summary)
    x = common_attention.combine_heads(x)

    return tf.squeeze(x, axis=1)


def mlp(feature, hparams, name="mlp"):
  """Multi layer perceptron with dropout and relu activation."""
  with tf.variable_scope(name, "mlp", values=[feature]):
    num_mlp_layers = hparams.num_mlp_layers
    mlp_size = hparams.mlp_size
    for _ in range(num_mlp_layers):
      feature = common_layers.dense(feature, mlp_size, activation=None)
      utils.collect_named_outputs("norms", "mlp_feature",
                                  tf.norm(feature, axis=-1))
      feature = common_layers.layer_norm(feature)
      feature = tf.nn.relu(feature)
      feature = tf.nn.dropout(feature, keep_prob=1.-hparams.dropout)
    return feature


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


def image_question_encoder(encoder_inputs,
                           encoder_self_attention_bias,
                           hparams,
                           query=None,
                           name="image_question_encoder",
                           save_weights_to=None,
                           make_image_summary=True):
  """A stack of self attention layers."""
  x = encoder_inputs
  with tf.variable_scope(name):
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          y = vqa_layers.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              block_length=hparams.block_length,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              scale_dotproduct=hparams.scale_dotproduct,
          )
          utils.collect_named_outputs(
              "norms", "encoder_self_attention_%d"%(layer),
              tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs(
              "norms", "encoder_self_attention_postprocess_%d"%(layer),
              tf.norm(x, axis=-1))
        if query is not None:
          with tf.variable_scope("encdec_attention"):
            y = common_attention.multihead_attention(
                common_layers.layer_preprocess(x, hparams),
                query,
                None,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                attention_type=hparams.self_attention_type,
                block_length=hparams.block_length,
                save_weights_to=save_weights_to,
                make_image_summary=make_image_summary,
                scale_dotproduct=hparams.scale_dotproduct,
            )
            utils.collect_named_outputs(
                "norms",
                "encoder_decoder_attention_%d"%(layer),
                tf.norm(y, axis=-1))
            x = common_layers.layer_postprocess(x, y, hparams)
            utils.collect_named_outputs(
                "norms",
                "encoder_decoder_attention_post_%d"%(layer),
                tf.norm(x, axis=-1))
        with tf.variable_scope("ffn"):
          y = common_layers.dense_relu_dense(
              common_layers.layer_preprocess(x, hparams),
              hparams.filter_size,
              hparams.hidden_size,
              dropout=hparams.relu_dropout,
              )
          utils.collect_named_outputs(
              "norms", "encoder_ffn_%d"%(layer), tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs(
              "norms", "encoder_ffn_postprocess_%d"%(layer),
              tf.norm(x, axis=-1))
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def decoder(decoder_input,
            encoder_output,
            decoder_self_attention_bias,
            encoder_decoder_attention_bias,
            hparams,
            name="decoder",
            save_weights_to=None,
            make_image_summary=True,):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.

  Returns:
    y: a Tensors
  """
  x = decoder_input
  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              )
          utils.collect_named_outputs("norms",
                                      "decoder_self_attention_%d"%(layer),
                                      tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs("norms",
                                      "decoder_self_attention_post_%d"%(layer),
                                      tf.norm(x, axis=-1))
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y = common_attention.multihead_attention(
                common_layers.layer_preprocess(x, hparams),
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                save_weights_to=save_weights_to,
                make_image_summary=make_image_summary,
                )
            utils.collect_named_outputs(
                "norms",
                "decoder_encoder_attention_%d"%(layer),
                tf.norm(y, axis=-1))
            x = common_layers.layer_postprocess(x, y, hparams)
            utils.collect_named_outputs(
                "norms",
                "decoder_encoder_attention_post_%d"%(layer),
                tf.norm(x, axis=-1))
        with tf.variable_scope("ffn"):
          y = common_layers.dense_relu_dense(
              common_layers.layer_preprocess(x, hparams),
              hparams.filter_size,
              hparams.hidden_size,
              dropout=hparams.relu_dropout,
          )
          utils.collect_named_outputs("norms", "decoder_ffn_%d"%(layer),
                                      tf.norm(y, axis=-1))
          x = common_layers.layer_postprocess(x, y, hparams)
          utils.collect_named_outputs("norms", "decoder_ffn_post_%d"%(layer),
                                      tf.norm(x, axis=-1))
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def iterative_encoder_decoder(encoder_input,
                              encoder_self_attention_bias,
                              encoder_decoder_attention_bias,
                              query,
                              hparams):
  """Iterative encoder decoder."""
  for _ in xrange(hparams.num_rec_steps):
    with tf.variable_scope("step", reuse=tf.AUTO_REUSE):
      encoder_output = image_question_encoder(
          encoder_input,
          encoder_self_attention_bias,
          hparams,
          query)

      decoder_output = decoder(
          query,
          encoder_output,
          None,
          encoder_decoder_attention_bias,
          hparams)

      encoder_input = encoder_output
      query = decoder_output

      return decoder_output


@registry.register_hparams
def vqa_self_attention_base():
  """VQA attention baseline hparams."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 128
  hparams.use_fixed_batch_size = True,
  hparams.optimizer = "Adam"
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.997
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.weight_decay = 0.
  hparams.clip_grad_norm = 0.
  hparams.initializer = "xavier"
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_normalized_decay")
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate_constant = 1e-3
  hparams.learning_rate_decay_rate = 0.5
  hparams.learning_rate_decay_steps = 50000
  hparams.dropout = 0.5
  hparams.summarize_grads = True
  hparams.summarize_vars = True

  # not used hparams
  hparams.label_smoothing = 0.
  hparams.multiply_embedding_mode = "sqrt_depth"

  # add new hparams
  # use raw image as input
  hparams.add_hparam("image_input_type", "image")
  hparams.add_hparam("image_model_fn", "resnet_v1_152")
  hparams.add_hparam("resize_side", 512)
  hparams.add_hparam("height", 448)
  hparams.add_hparam("width", 448)
  hparams.add_hparam("distort", True)
  hparams.add_hparam("train_resnet", False)

  # image parts
  hparams.add_hparam("image_feat_preprocess_proj", True)
  hparams.add_hparam("image_feat_preprocess_layernorm", True)
  hparams.add_hparam("image_feat_encode", True)
  hparams.add_hparam("image_hidden_size", 0)  # default to hidden_size
  hparams.add_hparam("image_filter_size", 0)  # defaults to filter_size

  # question hidden size
  hparams.hidden_size = 512
  hparams.filter_size = 1024
  hparams.num_hidden_layers = 4

  hparams.add_hparam("multimodal_combine", "concat")
  hparams.add_hparam("num_mlp_layers", 1)
  hparams.add_hparam("mlp_size", 1024)

  # self attention parts
  hparams.norm_type = "layer"
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.add_hparam("pos", "timing")
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 0)
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("self_attention_type", "dot_product")
  hparams.add_hparam("image_self_attention_type", "dot_product")
  hparams.add_hparam("question_self_attention_type", "dot_product")
  hparams.add_hparam("block_length", 1)
  hparams.add_hparam("scale_dotproduct", True)

  # iterative part
  hparams.add_hparam("num_rec_steps", 3)

  return hparams


@registry.register_hparams
def vqa_self_attention_feature():
  hparams = vqa_self_attention_base()
  hparams.image_input_type = "feature"
  return hparams


@registry.register_hparams
def vqa_self_attention_feature_batch1024():
  hparams = vqa_self_attention_feature()
  hparams.batch_size = 1024
  return hparams


@registry.register_hparams
def vqa_self_attention_feature_batch1024_big():
  """Big model."""
  hparams = vqa_self_attention_feature_batch1024()
  hparams.learning_rate_constant = 7e-4
  hparams.batch_size = 256
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  hparams.attention_dropout = 0.3
  hparams.relu_dropout = 0.3
  return hparams


@registry.register_hparams
def vqa_self_attention_feature_batch1024_exp():
  hparams = vqa_self_attention_feature_batch1024()
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*exp_decay")
  hparams.learning_rate_decay_steps = 4000
  return hparams


@registry.register_hparams
def vqa_self_attention_feature_batch1024_hidden6():
  hparams = vqa_self_attention_feature_batch1024()
  hparams.num_hidden_layers = 6
  return hparams


@registry.register_hparams
def vqa_self_attention_feature_batch1024_hidden6_big():
  hparams = vqa_self_attention_feature_batch1024_hidden6()
  hparams.batch_size = 256
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def vqa_self_attention_feature_batch1024_drop03():
  hparams = vqa_self_attention_feature_batch1024()
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def vqa_self_attention_feature_lr5():
  hparams = vqa_self_attention_feature()
  hparams.learning_rate_constant = 5e-4
  return hparams
