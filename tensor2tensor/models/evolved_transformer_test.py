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


def get_model(hparams, has_input=True):
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.hidden_size = 4
  hparams.num_heads = 1
  hparams.num_encoder_layers = 1
  hparams.num_decoder_layers = 1

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

  return (evolved_transformer.EvolvedTransformer(
      hparams, tf.estimator.ModeKeys.TRAIN, p_hparams), features)


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
        transformer.transformer_tiny(), has_input=False)

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
      slow_result = model._slow_greedy_infer(
          features, decode_length)["outputs"]
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


if __name__ == "__main__":
  tf.test.main()
