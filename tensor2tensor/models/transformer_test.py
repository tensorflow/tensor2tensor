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

"""Tests for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import transformer

import tensorflow as tf


BATCH_SIZE = 3
INPUT_LENGTH = 5
TARGET_LENGTH = 7
VOCAB_SIZE = 10


class TransformerTest(tf.test.TestCase):

  def getModel(self, hparams, mode=tf.estimator.ModeKeys.TRAIN):
    hparams.hidden_size = 8
    hparams.filter_size = 32
    hparams.num_heads = 1
    hparams.layer_prepostprocess_dropout = 0.0

    p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE, VOCAB_SIZE)
    hparams.problems = [p_hparams]

    inputs = -1 + np.random.random_integers(
        VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
    targets = -1 + np.random.random_integers(
        VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
    features = {
        "inputs": tf.constant(inputs, dtype=tf.int32),
        "targets": tf.constant(targets, dtype=tf.int32),
        "target_space_id": tf.constant(1, dtype=tf.int32),
    }

    return transformer.Transformer(hparams, mode, p_hparams), features

  def testTransformer(self):
    model, features = self.getModel(transformer.transformer_small())
    shadred_logits, _ = model.model_fn(features)
    logits = tf.concat(shadred_logits, 0)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testTransformerRelative(self):
    model, features = self.getModel(transformer.transformer_relative_tiny())
    shadred_logits, _ = model.model_fn(features)
    logits = tf.concat(shadred_logits, 0)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testGreedyVsFast(self):
    model, features = self.getModel(transformer.transformer_small())

    decode_length = 2

    out_logits, _ = model.model_fn(features)
    out_logits = tf.squeeze(out_logits[0], axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    model, _ = self.getModel(transformer.transformer_small(),
                             mode=tf.estimator.ModeKeys.PREDICT)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      greedy_result, _, _ = model._slow_greedy_infer(features, decode_length)
      greedy_result = tf.squeeze(greedy_result, axis=[2, 3])

      fast_result, _, _ = model._greedy_infer(features, decode_length)

    with self.test_session():
      greedy_res = greedy_result.eval()
      fast_res = fast_result.eval()

    self.assertEqual(fast_res.shape, (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(greedy_res, fast_res)

  def testBeamVsFast(self):
    model, features = self.getModel(transformer.transformer_small())

    decode_length = 2

    out_logits, _ = model.model_fn(features)
    out_logits = tf.squeeze(out_logits[0], axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    model, _ = self.getModel(transformer.transformer_small(),
                             mode=tf.estimator.ModeKeys.PREDICT)

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

    self.assertEqual(fast_res.shape, (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(beam_res, fast_res)


if __name__ == "__main__":
  tf.test.main()
