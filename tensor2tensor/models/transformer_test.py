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
"""Tests for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import transformer

import tensorflow as tf


BATCH_SIZE = 3
INPUT_LENGTH = 5
TARGET_LENGTH = 7
VOCAB_SIZE = 10


def tf_version_has_inplace_ops():
  # Available in TF 1.8+
  major, minor = [int(el) for el in tf.__version__.split(".")[:2]]
  return major > 1 or (major == 1 and minor >= 8)


def get_model(hparams=None, mode=tf.estimator.ModeKeys.TRAIN,
              has_input=True, model_cls=transformer.Transformer):
  if hparams is None:
    hparams = transformer.transformer_tiny()
  hparams.hidden_size = 8
  hparams.filter_size = 32
  hparams.num_heads = 1
  hparams.layer_prepostprocess_dropout = 0.0

  p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE, VOCAB_SIZE)
  if not has_input:
    p_hparams.input_modality = {}
  hparams.problem_hparams = p_hparams

  inputs = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
  targets = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
  if has_input:
    features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")

  return model_cls(hparams, mode, p_hparams), features


class TransformerTest(tf.test.TestCase):

  def testTransformer(self):
    model, features = get_model(transformer.transformer_small())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testTransformerRelative(self):
    model, features = get_model(transformer.transformer_relative_tiny())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testSlowVsFast(self):
    model, features = get_model(transformer.transformer_small())

    decode_length = 3

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      greedy_result = model._slow_greedy_infer(
          features, decode_length)["outputs"]
      greedy_result = tf.squeeze(greedy_result, axis=[2, 3])

      fast_result = model._greedy_infer(features, decode_length)["outputs"]

    with self.test_session():
      greedy_res = greedy_result.eval()
      fast_res = fast_result.eval()

    self.assertEqual(fast_res.shape, (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(greedy_res, fast_res)

  def testSlowVsFastNoInput(self):
    model, features = get_model(
        transformer.transformer_small(), has_input=False)

    decode_length = 3

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      slow_result = model._slow_greedy_infer(
          features, decode_length)["outputs"]
      slow_result = tf.squeeze(slow_result, axis=[2, 3])

      fast_result = model._greedy_infer(features, decode_length)["outputs"]

    with self.test_session():
      slow_res = slow_result.eval()
      fast_res = fast_result.eval()

    self.assertEqual(slow_res.shape, (BATCH_SIZE, decode_length))
    self.assertAllClose(slow_res, fast_res)

  def testBeamDecodeWithRelativeAttention(self):
    decode_length = 2
    model, features = get_model(transformer.transformer_relative_tiny())
    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    beam_result = model._beam_decode(
        features, decode_length, beam_size=4, top_beams=1,
        alpha=1.0)["outputs"]

    with self.test_session():
      tf.global_variables_initializer().run()
      beam_result.eval()

    # TODO(petershaw): This test is flaky because the decode may hit EOS before
    # getting to the expected length.
    # self.assertEqual(beam_res.shape,
    #                  (BATCH_SIZE, INPUT_LENGTH + decode_length))

  def testBeamVsFast(self):
    model, features = get_model(transformer.transformer_small())

    decode_length = 2

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      beam_result = model._beam_decode_slow(
          features,
          decode_length,
          beam_size=4,
          top_beams=1,
          alpha=1.0)["outputs"]

      fast_result = model._beam_decode(
          features,
          decode_length,
          beam_size=4,
          top_beams=1,
          alpha=1.0)["outputs"]

    with self.test_session():
      beam_res = beam_result.eval()
      fast_res = fast_result.eval()

    self.assertEqual(fast_res.shape,
                     (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(beam_res, fast_res)

  def testTransformerWithoutProblem(self):
    hparams = transformer.transformer_test()

    embedded_inputs = np.random.random_sample(
        (BATCH_SIZE, INPUT_LENGTH, 1, hparams.hidden_size))
    embedded_targets = np.random.random_sample(
        (BATCH_SIZE, TARGET_LENGTH, 1, hparams.hidden_size))

    transformed_features = {
        "inputs": tf.constant(embedded_inputs, dtype=tf.float32),
        "targets": tf.constant(embedded_targets, dtype=tf.float32)
    }

    model = transformer.Transformer(hparams)
    body_out, _ = model(transformed_features)

    self.assertAllEqual(
        body_out.get_shape().as_list(),
        [BATCH_SIZE, TARGET_LENGTH, 1, hparams.hidden_size])

  def testTransformerWithEncoderDecoderAttentionLoss(self):
    model, features = get_model(
        transformer.transformer_supervised_attention())
    expected_attention_weights = np.random.random_sample(
        size=(BATCH_SIZE, TARGET_LENGTH, INPUT_LENGTH))
    features["expected_attentions"] = tf.constant(
        expected_attention_weights, dtype=tf.float32)
    _, extra_loss = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(extra_loss["attention_loss"])
    self.assertEqual(res.shape, ())

  def _create_greedy_infer_model(self):
    """Creates model for greedy inference testing.

    Returns:
      model: A t2t model.
      features: An map of string to tensor.
    """
    model, features = get_model(transformer.transformer_small())

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    return model, features

  def testGreedySlowTPUVsNonTPU(self):
    if not tf_version_has_inplace_ops():
      return

    decode_length = 3

    model, features = self._create_greedy_infer_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      slow_result_non_tpu = model._slow_greedy_infer(
          features, decode_length)["outputs"]
      slow_result_non_tpu = tf.squeeze(slow_result_non_tpu, axis=[2, 3])

      slow_result_tpu = model._slow_greedy_infer_tpu(
          features, decode_length)["outputs"]
      slow_result_tpu = tf.squeeze(slow_result_tpu, axis=[2, 3])

    with self.test_session():
      slow_non_tpu_res = slow_result_non_tpu.eval()
      slow_tpu_res = slow_result_tpu.eval()

    self.assertEqual(slow_tpu_res.shape,
                     (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(slow_tpu_res, slow_non_tpu_res)

  def testGreedyFastTPUVsNonTPU(self):
    if not tf_version_has_inplace_ops():
      return

    decode_length = 3

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
    if not tf_version_has_inplace_ops():
      return

    decode_length = 3

    model, features = self._create_greedy_infer_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      slow_result = model._slow_greedy_infer_tpu(
          features, decode_length)["outputs"]
      slow_result = tf.squeeze(slow_result, axis=[2, 3])

      fast_result = model._greedy_infer(
          features, decode_length, use_tpu=True)["outputs"]

    with self.test_session():
      slow_res = slow_result.eval()
      fast_res = fast_result.eval()

    self.assertEqual(fast_res.shape,
                     (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(fast_res, slow_res)


class TransformerScorerTest(tf.test.TestCase):

  def testReturnsScores(self):
    model, features = get_model(
        mode=tf.estimator.ModeKeys.PREDICT,
        model_cls=transformer.TransformerScorer)
    infer_out = model.infer(features)
    self.assertTrue("outputs" in infer_out)
    self.assertTrue("scores" in infer_out)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      infer_out = session.run(infer_out)
      self.assertEqual((BATCH_SIZE,), infer_out["scores"].shape)
      self.assertEqual((BATCH_SIZE, TARGET_LENGTH), infer_out["outputs"].shape)

  def testVarNames(self):
    with tf.Graph().as_default():
      model, features = get_model(
          mode=tf.estimator.ModeKeys.PREDICT,
          model_cls=transformer.TransformerScorer)
      _ = model.infer(features)
      scorer_vars = [v.name for v in tf.global_variables()]

    with tf.Graph().as_default():
      model, features = get_model(
          mode=tf.estimator.ModeKeys.EVAL,
          model_cls=transformer.TransformerScorer)
      _ = model(features)
      scorer_eval_vars = [v.name for v in tf.global_variables()]

    with tf.Graph().as_default():
      model, features = get_model(
          mode=tf.estimator.ModeKeys.EVAL,
          model_cls=transformer.Transformer)
      _ = model(features)
      transformer_vars = [v.name for v in tf.global_variables()]

    self.assertEqual(sorted(scorer_vars), sorted(transformer_vars))
    self.assertEqual(sorted(scorer_eval_vars), sorted(transformer_vars))


if __name__ == "__main__":
  tf.test.main()
