# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

"""The Seq2Edits model.

Seq2Edits is an adaptation of the Transformer that predicts span level edits
and pairs them with tags. The Seq2Edits model is described in

  Stahlberg, Felix, and Kumar, Shankar. "Seq2Edits: Sequence Transduction Using
  Span-level Edit Operations." Proceedings of the 2020 Conference on Empirical
  Methods in Natural Language Processing (EMNLP). 2020.
  https://www.aclweb.org/anthology/2020.emnlp-main.418/

T2T problem definitions for Seq2Edits are in data_generators/seq2edits.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import transformer_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf


def maybe_flatten4d3d(x):
  """Flatten if tensor has 4 dimensions.

  Pass through otherwise.

  This is useful since additional dimensions are sometimes removed on the TPU,
  see e.g.
    https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models/transformer.py?l=159&rcl=279807999

  Args:
    x: a tensor

  Returns:
    A 3D tensor if x is 4D, unmodified x otherwise.
  """
  xshape = common_layers.shape_list(x)
  return common_layers.flatten4d3d(x) if len(xshape) == 4 else x


def maybe_flatten3d2d(x):
  """Flatten if tensor has 3 dimensions, similar to maybe_flatten4d3d()."""
  xshape = common_layers.shape_list(x)
  if len(xshape) != 3:
    return x
  return tf.reshape(x, [xshape[0], xshape[1] * xshape[2]])


def maybe_flatten4d2d(x):
  return maybe_flatten3d2d(maybe_flatten4d3d(x))


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  """See transformer.features_to_nonpadding."""
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None


def gather_2d(params, indices):
  """2D version of tf.gather.

  This is a batched version of tf.gather(), i.e. it applies tf.gather() to
  each batch separately.
  Example:
    params = [[10, 11, 12, 13, 14],
              [20, 21, 22, 23, 24]]
    indices = [[0, 0, 1, 1, 1, 2],
               [1, 3, 0, 0, 2, 2]]
    result = [[10, 10, 11, 11, 11, 12],
              [21, 23, 20, 20, 22, 22]]
  This method is copied from
    https://github.com/fstahlberg/tensor2tensor-usr/blob/master/usr/utils.py
  which is published under Apache 2.

  Args:
    params: A [batch_size, n, ...] tensor with data
    indices: A [batch_size, num_indices] int32 tensor with indices into params.
      Entries must be smaller than n

  Returns:
    The result of tf.gather() on each entry of the batch.
  """
  batch_size = tf.shape(params)[0]
  num_indices = tf.shape(indices)[1]
  batch_indices = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_indices])
  # batch_indices is [[0,0,0,0,...],[1,1,1,1,...],...]
  gather_nd_indices = tf.stack([batch_indices, indices], axis=2)
  return tf.gather_nd(params, gather_nd_indices)


@registry.register_model
class TransformerSeq2edits(t2t_model.T2TModel):
  """The Seq2Edits model. See file docstring."""

  def __init__(self, *args, **kwargs):
    super(TransformerSeq2edits, self).__init__(*args, **kwargs)
    self.attention_weights = {}  # For visualizing attention heads.
    self._encoder_function = transformer_layers.transformer_encoder
    self._decoder_function = transformer.transformer_decoder
    self._prepare_encoder_fn = transformer_layers.transformer_prepare_encoder
    self._prepare_decoder_fn = transformer.transformer_prepare_decoder
    self.loss_num = {}
    self.logits = {}
    self.loss_den = None

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encodes transformer inputs, see transformer.transformer_encode()."""
    return transformer.transformer_encode(
        self._encoder_function,
        inputs,
        target_space,
        hparams,
        attention_weights=self.attention_weights,
        features=features,
        losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             **kwargs):
    """Decodes Transformer outputs, see transformer.transformer_decode()."""
    return transformer.transformer_decode(
        self._decoder_function,
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        attention_weights=self.attention_weights,
        cache=cache,
        decode_loop_step=decode_loop_step,
        nonpadding=nonpadding,
        losses=losses,
        **kwargs)

  def body(self, features):
    """Seq2Edits main model_fn.

    Args:
      features: Feature dictionary. Should contain the following fields:
          "inputs": [batch_size, input_length, 1, hidden_dim] float tensor with
            input token embeddings.
          "targets": [batch_size, target_length, 1, hidden_dim] float tensor
            with target token embeddings.
          "targets_error_tag": [batch_size, target_length, 1, hidden_dim] float
            tensor with target error tag embeddings.
          "targets_start_token": [batch_size, target_length] int tensor with
            start token positions.
          "targets_end_token": [batch_size, target_length] int tensor with end
            token positions.
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. Dictionary containing the following fields:
        "targets": [batch_size, target_length, hidden_dim] float tensor with
          decoder outputs
        "targets_error_tag": [batch_size, target_length, hidden_dim] float
          tensor with decoder outputs
        "targets_start_token": [batch_size, input_length, target_length] float
          tensor with start token position logits
        "targets_end_token": [batch_size, input_length, target_length] float
          tensor with end token position logits
    """
    hparams = self._hparams

    losses = []

    if self.has_input:
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          features["inputs"],
          target_space,
          hparams,
          features=features,
          losses=losses)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
        targets, hparams, features=features)

    nonpadding = features_to_nonpadding(features, "targets")

    # Add edit ops layer to condition on start_token, end_token, and error_tag
    decoder_input = transformer_edit_ops_layer(
        decoder_input,
        hparams,
        encoder_output,
        features,
        nonpadding=nonpadding,
        losses=losses)

    if hparams.middle_prediction:
      num_decoder_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
      hparams.num_decoder_layers = int(
          num_decoder_layers / hparams.middle_prediction_layer_factor)

    decode_kwargs = {}
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=nonpadding,
        losses=losses,
        **decode_kwargs)

    loss_mask = common_layers.weights_nonzero(
        maybe_flatten4d2d(features["targets_raw"]))
    self.loss_den = tf.reduce_sum(loss_mask)
    decoder_output = self._prediction_cascade(
        hparams=hparams,
        features=features,
        losses=losses,
        loss_mask=loss_mask,
        nonpadding=nonpadding,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        encoder_output=encoder_output,
        decoder_output=decoder_output)

    if hparams.middle_prediction:
      with tf.variable_scope("after_prediction"):
        decoder_output = self.decode(
            decoder_input + decoder_output,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=nonpadding,
            losses=losses,
            **decode_kwargs)

    ret = {"targets": tf.reshape(decoder_output, targets_shape)}
    ret.update(self.logits)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret

  def _prediction_cascade(self, hparams, features, losses, loss_mask,
                          nonpadding, encoder_decoder_attention_bias,
                          encoder_output, decoder_output):
    if hparams.use_error_tags:
      (decoder_output, error_tag_logits,
       error_tag_loss) = transformer_error_tag_prediction_layer(
           decoder_output, hparams, features, loss_mask=loss_mask)
      self.logits["targets_error_tag"] = error_tag_logits
      self.loss_num["targets_error_tag"] = error_tag_loss
      decoder_output = transformer_between_predictions_layer(
          decoder_output,
          hparams,
          name="post_error_tag",
          nonpadding=nonpadding,
          losses=losses)

    pos_feat_names = []
    if hparams.use_start_token:
      pos_feat_names.append("targets_start_token")
    pos_feat_names.append("targets_end_token")
    for pos_feat_name in pos_feat_names:
      (decoder_output, pos_logits,
       pos_loss) = transformer_pointer_prediction_layer(
           pos_feat_name,
           encoder_output,
           decoder_output,
           encoder_decoder_attention_bias,
           hparams,
           features,
           loss_mask=loss_mask)
      self.logits[pos_feat_name] = pos_logits
      self.loss_num[pos_feat_name] = pos_loss
      decoder_output = transformer_between_predictions_layer(
          decoder_output,
          hparams,
          name="post_%s" % pos_feat_name,
          nonpadding=nonpadding,
          losses=losses)
    return decoder_output

  def _loss_single(self, logits, feature_name, feature, weights=None):
    """Prevents modality loss computation for targets_*."""
    if feature_name in [
        "targets_start_token", "targets_end_token", "targets_error_tag"
    ]:
      loss_num = self.loss_num[feature_name]
      loss_num *= self._problem_hparams.loss_multiplier
      loss_den = self.loss_den
    else:
      loss_num, loss_den = super(TransformerSeq2edits,
                                 self)._loss_single(logits, feature_name,
                                                    feature, weights)
    tf.summary.scalar("loss/%s" % feature_name, loss_num / loss_den)
    return loss_num, loss_den

  def top(self, body_output, features):
    """Adds additional dimensions and then calls super class implementation."""
    exp_features = features
    for feat in body_output.keys():
      while len(body_output[feat].shape) < 4:
        logging.warning("Expanding body output %s...", feat)
        body_output[feat] = tf.expand_dims(body_output[feat], -2)
      if feat in exp_features:
        while len(exp_features[feat].shape) < 4:
          exp_features[feat] = tf.expand_dims(exp_features[feat], -1)
          logging.warning("Expanding feature %s...", feat)
    return super(TransformerSeq2edits, self).top(body_output, exp_features)


def _pointer_feedback(pointers, encoder_output, shift=True):
  """Feedback loop for pointer networks.

  Args:
    pointers: [batch_size, target_length] int tensor with pointers into the
      source sentence.
    encoder_output: [batch_size, input_length, hidden_size] tensor with encoder
      outputs.
    shift: Whether to shift the pointers to the right.

  Returns:
    A [batch_size, target_length, hidden_size] tensor with encoder outputs.
  """
  if shift:
    pointers = common_layers.shift_right_2d(pointers)
  return gather_2d(encoder_output, pointers)


def transformer_edit_ops_layer(decoder_input,
                               hparams,
                               encoder_output,
                               features,
                               cache=None,
                               decode_loop_step=None,
                               nonpadding=None,
                               losses=None,
                               layer_collection=None):
  """Layer that conditions on the error tag and start and end token pointers."""
  if isinstance(encoder_output, list):  # Select forward encoder
    encoder_output = encoder_output[0]
  with tf.variable_scope("edit_ops_layer"):
    with tf.variable_scope("ffn"):
      x = decoder_input
      # Shorthand for layer preprocessing
      # pylint: disable=g-long-lambda
      preproc = lambda z: common_layers.layer_preprocess(
          z, hparams, layer_collection=layer_collection)
      # pylint: enable=g-long-lambda

      feedback_start_token = (hparams.use_start_token or
                              not hparams.feedback_end_token)
      if feedback_start_token:
        start_token = _pointer_feedback(
            features["targets_start_token"],
            encoder_output,
            shift=hparams.feedback_end_token)
      if hparams.feedback_end_token:
        end_token = _pointer_feedback(features["targets_end_token"],
                                      encoder_output)
      layer_inputs = [preproc(x)]
      if hparams.use_error_tags:
        error_tags = common_layers.shift_right_3d(
            common_layers.flatten4d3d(features["targets_error_tag"]))
        layer_inputs.append(preproc(error_tags))
      if feedback_start_token:
        layer_inputs.append(start_token)
      if hparams.feedback_end_token:
        layer_inputs.append(end_token)
      y = transformer_layers.transformer_ffn_layer(
          tf.concat(layer_inputs, axis=2),
          hparams,
          conv_padding="LEFT",
          nonpadding_mask=nonpadding,
          losses=losses,
          cache=cache,
          decode_loop_step=decode_loop_step,
          layer_collection=layer_collection)
      x = common_layers.layer_postprocess(x, y, hparams)
      return x


def transformer_between_predictions_layer(x,
                                          hparams,
                                          name,
                                          cache=None,
                                          decode_loop_step=None,
                                          nonpadding=None,
                                          losses=None,
                                          layer_collection=None):
  """Stack between prediction layers."""
  with tf.variable_scope(name):
    for i in range(hparams.ffn_in_prediction_cascade):
      with tf.variable_scope("layer_%d" % i):
        y = transformer_layers.transformer_ffn_layer(
            common_layers.layer_preprocess(
                x, hparams, layer_collection=layer_collection),
            hparams,
            conv_padding="LEFT",
            nonpadding_mask=nonpadding,
            losses=losses,
            cache=cache,
            decode_loop_step=decode_loop_step,
            layer_collection=layer_collection)
        x = common_layers.layer_postprocess(x, y, hparams)
  return x


def get_error_tag_embedding_matrix():
  candidates = [
      var for var in tf.global_variables() if "targets_error_tag" in var.op.name
  ]
  if len(candidates) != 1:
    raise ValueError("Could not identify error tag embedding matrix! "
                     "Matching variable names: %s" % candidates)
  embed_mat = candidates[0]
  return embed_mat


def transformer_error_tag_prediction_layer(x,
                                           hparams,
                                           features,
                                           loss_mask,
                                           layer_collection=None):
  """Layer that predicts the error tag."""
  with tf.variable_scope("error_tag_prediction"):
    x = maybe_flatten4d3d(x)
    vocab_size = hparams.problem.feature_info["targets_error_tag"].vocab_size
    labels = features["targets_error_tag_raw"]
    with tf.variable_scope("projection"):
      bottleneck = common_layers.dense(
          x,
          hparams.error_tag_embed_size,
          layer_collection=layer_collection,
          name="bottleneck")
      logits = common_layers.dense(
          bottleneck,
          vocab_size,
          use_bias=False,
          layer_collection=layer_collection,
          name="logits")
      xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
      loss = tf.reduce_sum(xent * loss_mask)
    with tf.variable_scope("embedding"):
      embed_mat = get_error_tag_embedding_matrix()
      y = common_layers.layer_preprocess(
          common_layers.embedding(
              labels, vocab_size, hparams.hidden_size, embedding_var=embed_mat),
          hparams,
          layer_collection=layer_collection)
      x = common_layers.layer_postprocess(x, y, hparams)
    return x, logits, loss


def transformer_pointer_prediction_layer(feature_name,
                                         encoder_output,
                                         x,
                                         encoder_decoder_attention_bias,
                                         hparams,
                                         features,
                                         loss_mask,
                                         layer_collection=None):
  """Layer that predicts the start or end token position.

  Args:
    feature_name: 'targets_start_token' or 'targets_end_token'
    encoder_output: [batch_size, input_length, hidden_size] tensor with encoder
      outputs
    x: [batch_size, target_length, 1, hidden_size] tensor with decoder outputs
    encoder_decoder_attention_bias: [batch_size, input_length, target_length]
      attention mask
    hparams: Hyper parameters
    features: Feature dictionary
    loss_mask: [batch_size, target_length] mask for loss computation.
    layer_collection: Layer collection

  Returns:
    (x, logits, loss)
  """
  if isinstance(encoder_output, list):
    pointer_encoder_output = encoder_output[1]
    encoder_output = sum(encoder_output)
  else:
    pointer_encoder_output = encoder_output
  with tf.variable_scope("%s_prediction" % feature_name):
    x = maybe_flatten4d3d(x)
    encoder_decoder_attention_bias = common_layers.flatten4d3d(
        encoder_decoder_attention_bias)
    q = common_attention.compute_attention_component(x, hparams.hidden_size)
    k = common_attention.compute_attention_component(encoder_output,
                                                     hparams.hidden_size)
    # Scaled dot-product attention
    scalar = tf.rsqrt(tf.to_float(common_layers.shape_list(q)[2]))
    logits = tf.matmul(q * scalar, k, transpose_b=True)

    logits += encoder_decoder_attention_bias

    labels = features["%s_raw" % feature_name]
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    loss = tf.reduce_sum(xent * loss_mask)

    pointer_out = gather_2d(pointer_encoder_output, labels)
    y = common_layers.layer_preprocess(
        pointer_out, hparams, layer_collection=layer_collection)
    x = common_layers.layer_postprocess(x, y, hparams)
    return x, logits, loss
