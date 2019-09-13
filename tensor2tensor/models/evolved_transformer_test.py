# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Tests for the Evolved Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import evolved_transformer
from tensor2tensor.models import transformer

import tensorflow as tf

BATCH_SIZE = 3
INPUT_LENGTH = 5
TARGET_LENGTH = 7
VOCAB_SIZE = 10
DECODE_LENGTH = 3


def print_vars(all_vars=None):
  """Print info about a list of variables."""
  if not all_vars:
    all_vars = tf.trainable_variables()
  tf.logging.info("Format: <name>, <shape>, <(soft) device placement>")
  for var in all_vars:
    tf.logging.info("  %s, %s, %s" %
                    (var.name, str(var.get_shape()), var.op.device))


def get_var(name):
  """Get trainable variable by name."""
  variables = [var for var in tf.trainable_variables() if var.name == name]
  if len(variables) == 1:
    return variables[0]
  raise ValueError("`name` must match exactly one variable. '%s' matched %d" %
                   (name, len(variables)))


def get_vars(names):
  """Get trainable variables by name."""
  return [get_var(name) for name in names]


def assert_with_message(assert_method, a, b, message):
  try:
    assert_method(a, b)
  except AssertionError as e:
    tf.logging.error(message)
    raise e


def get_model(hparams, has_input=True, num_decoder_layers=1):
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.hidden_size = 4
  hparams.num_heads = 1
  hparams.num_encoder_layers = 1
  hparams.num_decoder_layers = num_decoder_layers

  p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE, VOCAB_SIZE,
                                                   hparams)
  if not has_input:
    del p_hparams.modality["inputs"]
  hparams.problem_hparams = p_hparams

  inputs = np.random.randint(VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
  targets = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32),
  }
  if has_input:
    features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")

  return (evolved_transformer.EvolvedTransformer(hparams,
                                                 tf.estimator.ModeKeys.TRAIN,
                                                 p_hparams), features)


class EvolvedTransformerTest(tf.test.TestCase):

  def testEvolvedTransformer(self):
    model, features = get_model(hparams=transformer.transformer_tiny())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testSlowVsFast(self):
    model, features = get_model(transformer.transformer_tiny())

    decode_length = DECODE_LENGTH

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(10):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      greedy_result = model._slow_greedy_infer(features,
                                               decode_length)["outputs"]
      greedy_result = tf.squeeze(greedy_result, axis=[2, 3])

      fast_result = model._greedy_infer(features, decode_length)["outputs"]

    with self.test_session():
      greedy_res = greedy_result.eval()
      fast_res = fast_result.eval()

    self.assertEqual(fast_res.shape, (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(greedy_res, fast_res)

  def testSlowVsFastNoInput(self):
    model, features = get_model(transformer.transformer_tiny(), has_input=False)

    decode_length = DECODE_LENGTH

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(10):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      slow_result = model._slow_greedy_infer(features, decode_length)["outputs"]
      slow_result = tf.squeeze(slow_result, axis=[2, 3])

      fast_result = model._greedy_infer(features, decode_length)["outputs"]

    with self.test_session():
      slow_res = slow_result.eval()
      fast_res = fast_result.eval()

    self.assertEqual(slow_res.shape, (BATCH_SIZE, decode_length))
    self.assertAllClose(slow_res, fast_res)

  def testBeamVsFast(self):
    model, features = get_model(transformer.transformer_tiny())

    decode_length = DECODE_LENGTH

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(10):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      beam_result = model._beam_decode_slow(
          features, decode_length, beam_size=4, top_beams=1,
          alpha=1.0)["outputs"]

      fast_result = model._beam_decode(
          features, decode_length, beam_size=4, top_beams=1,
          alpha=1.0)["outputs"]

    with self.test_session():
      beam_res = beam_result.eval()
      fast_res = fast_result.eval()

    self.assertAllClose(beam_res, fast_res)

  def _create_greedy_infer_model(self):
    """Creates model for greedy inference testing.

    Returns:
      model: A t2t model.
      features: An map of string to tensor.
    """
    model, features = get_model(transformer.transformer_tiny())

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(10):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    return model, features

  def testGreedySlowTPUVsNonTPU(self):
    decode_length = DECODE_LENGTH

    model, features = self._create_greedy_infer_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      slow_result_non_tpu = model._slow_greedy_infer(features,
                                                     decode_length)["outputs"]
      slow_result_non_tpu = tf.squeeze(slow_result_non_tpu, axis=[2, 3])

      slow_result_tpu = model._slow_greedy_infer_tpu(features,
                                                     decode_length)["outputs"]
      slow_result_tpu = tf.squeeze(slow_result_tpu, axis=[2, 3])

    with self.test_session():
      slow_non_tpu_res = slow_result_non_tpu.eval()
      slow_tpu_res = slow_result_tpu.eval()

    self.assertEqual(slow_tpu_res.shape,
                     (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(slow_tpu_res, slow_non_tpu_res)

  def testGreedyFastTPUVsNonTPU(self):
    decode_length = DECODE_LENGTH

    model, features = self._create_greedy_infer_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      fast_result_non_tpu = model._greedy_infer(
          features, decode_length, use_tpu=False)["outputs"]

      fast_result_tpu = model._greedy_infer(
          features, decode_length, use_tpu=True)["outputs"]

    with self.test_session():
      fast_non_tpu_res = fast_result_non_tpu.eval()
      fast_tpu_res = fast_result_tpu.eval()

    self.assertEqual(fast_tpu_res.shape,
                     (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(fast_tpu_res, fast_non_tpu_res)

  def testGreedyTPUSlowVsFast(self):
    decode_length = DECODE_LENGTH

    model, features = self._create_greedy_infer_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      slow_result = model._slow_greedy_infer_tpu(features,
                                                 decode_length)["outputs"]
      slow_result = tf.squeeze(slow_result, axis=[2, 3])

      fast_result = model._greedy_infer(
          features, decode_length, use_tpu=True)["outputs"]

    with self.test_session():
      slow_res = slow_result.eval()
      fast_res = fast_result.eval()

    self.assertEqual(fast_res.shape, (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(fast_res, slow_res)

  def testFrozenWeightsUnchangedByTraining(self):
    # Arrange.
    hparams = transformer.transformer_tiny()
    hparams.add_hparam("num_trainable_top_decoder_layers", 1)
    model, features = get_model(hparams, num_decoder_layers=3)
    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)
    frozen_names = [
        "evolved_transformer/symbol_modality_10_4/shared/weights_0:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_1:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_2:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_3:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_4:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_5:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_6:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_7:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_8:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_9:0",
        "evolved_transformer/body/target_space_embedding/kernel:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/dense/kernel:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/dense/bias:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/dense_1/kernel:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/dense_1/bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/dense/kernel:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/dense/bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/standard_conv_3x1/kernel:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/standard_conv_3x1/bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/separable_conv_9x1/depthwise_kernel:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/separable_conv_9x1/pointwise_kernel:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/separable_conv_9x1/bias:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/dense/kernel:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/dense/bias:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/dense_1/kernel:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/dense_1/bias:0",
        "evolved_transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_0/first_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_0/first_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_0/first_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_0/first_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv11x1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv11x1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv11x1/bias:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_1/bias:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_2/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_2/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_2/bias:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/dense/kernel:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/dense/bias:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/dense_1/kernel:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/dense_1/bias:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_1/first_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_1/first_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_1/first_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_1/first_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv11x1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv11x1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv11x1/bias:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_1/bias:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_2/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_2/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_2/bias:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/dense/kernel:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/dense/bias:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/dense_1/kernel:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/dense_1/bias:0",
    ]
    train_names = [
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_2/first_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_2/first_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_2/first_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_2/first_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv11x1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv11x1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv11x1/bias:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_1/bias:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_2/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_2/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_2/bias:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/dense/kernel:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/dense/bias:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/dense_1/kernel:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/dense_1/bias:0",
        "evolved_transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_1:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_2:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_3:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_4:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_5:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_6:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_7:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_8:0",
        "evolved_transformer/symbol_modality_10_4/softmax/weights_9:0",
    ]
    frozen_vars = get_vars(frozen_names)
    train_vars = get_vars(train_names)
    print_vars()

    # Act.
    with self.test_session() as session:
      tf.global_variables_initializer().run()
      frozen_values_before = session.run(frozen_vars)
      train_values_before = session.run(train_vars)
      for _ in range(10):  # Arbitrary number of training steps.
        apply_grad.run()
      frozen_values_after = session.run(frozen_vars)
      train_values_after = session.run(train_vars)

    # Assert.
    self.assertTrue(
        model._original_hparams.shared_embedding_and_softmax_weights)
    self.assertFalse(model.hparams.shared_embedding_and_softmax_weights)
    self.assertTrue(model.hparams.shared_embedding)
    for name, before, after in zip(frozen_names, frozen_values_before,
                                   frozen_values_after):
      assert_with_message(
          self.assertAllClose, before, after,
          "%s should be frozen, but changed after training." % name)
    for name, before, after in zip(train_names, train_values_before,
                                   train_values_after):
      assert_with_message(
          self.assertNotAllClose, before, after,
          "%s should be trainable, but did not change after training." % name)

  def testAllWeightsTrainableByDefault(self):
    # Arrange.
    model, features = get_model(
        transformer.transformer_tiny(), num_decoder_layers=3)
    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)
    var_names = [
        "evolved_transformer/symbol_modality_10_4/shared/weights_0:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_1:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_2:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_3:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_4:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_5:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_6:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_7:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_8:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_9:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_10:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_11:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_12:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_13:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_14:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_15:0",
        "evolved_transformer/body/target_space_embedding/kernel:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/dense/kernel:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/dense/bias:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/dense_1/kernel:0",
        "evolved_transformer/body/encoder/layer_0/gated_linear_unit/dense_1/bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/dense/kernel:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/dense/bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/standard_conv_3x1/kernel:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/standard_conv_3x1/bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/separable_conv_9x1/depthwise_kernel:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/separable_conv_9x1/pointwise_kernel:0",
        "evolved_transformer/body/encoder/layer_0/conv_branches/separable_conv_9x1/bias:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/encoder/layer_0/self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/dense/kernel:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/dense/bias:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/dense_1/kernel:0",
        "evolved_transformer/body/encoder/layer_0/dense_layers/dense_1/bias:0",
        "evolved_transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/encoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_0/16_head_self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_0/first_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_0/first_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_0/first_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_0/first_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv11x1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv11x1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv11x1/bias:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_1/bias:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_2/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_2/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_0/conv_branches/separable_conv_7x1_2/bias:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_0/self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_0/second_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/dense/kernel:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/dense/bias:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/dense_1/kernel:0",
        "evolved_transformer/body/decoder/layer_0/dense_layers/dense_1/bias:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_1/16_head_self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_1/first_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_1/first_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_1/first_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_1/first_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv11x1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv11x1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv11x1/bias:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_1/bias:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_2/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_2/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_1/conv_branches/separable_conv_7x1_2/bias:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_1/self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_1/second_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/dense/kernel:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/dense/bias:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/dense_1/kernel:0",
        "evolved_transformer/body/decoder/layer_1/dense_layers/dense_1/bias:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_2/16_head_self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_2/first_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_2/first_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_2/first_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_2/first_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv11x1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv11x1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv11x1/bias:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_1/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_1/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_1/bias:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_2/depthwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_2/pointwise_kernel:0",
        "evolved_transformer/body/decoder/layer_2/conv_branches/separable_conv_7x1_2/bias:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_2/self_attention/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/multihead_attention/q/kernel:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/multihead_attention/k/kernel:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/multihead_attention/v/kernel:0",
        "evolved_transformer/body/decoder/layer_2/second_attend_to_encoder/multihead_attention/output_transform/kernel:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/layer_prepostprocess/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/dense/kernel:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/dense/bias:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/layer_prepostprocess_1/layer_norm/layer_norm_bias:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/dense_1/kernel:0",
        "evolved_transformer/body/decoder/layer_2/dense_layers/dense_1/bias:0",
        "evolved_transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale:0",
        "evolved_transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_bias:0",
    ]
    variables = get_vars(var_names)
    print_vars()

    # Act.
    with self.test_session() as session:
      tf.global_variables_initializer().run()
      values_before = session.run(variables)
      for _ in range(10):  # Arbitrary number of training steps.
        apply_grad.run()
      values_after = session.run(variables)

    # Assert.
    self.assertTrue(
        model._original_hparams.shared_embedding_and_softmax_weights)
    self.assertTrue(model.hparams.shared_embedding_and_softmax_weights)
    self.assertFalse(model.hparams.shared_embedding)
    self.assertSameElements(var_names,
                            [var.name for var in tf.trainable_variables()])
    empty_vars = {
        "evolved_transformer/symbol_modality_10_4/shared/weights_10:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_11:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_12:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_13:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_14:0",
        "evolved_transformer/symbol_modality_10_4/shared/weights_15:0"
    }
    for name, before, after in zip(var_names, values_before, values_after):
      if name in empty_vars:
        self.assertEqual(before.size, after.size)
        self.assertEqual(before.size, 0)
      else:
        assert_with_message(
            self.assertNotAllClose, before, after,
            "%s should be trainable, but did not change after training." % name)


if __name__ == "__main__":
  tf.test.main()
