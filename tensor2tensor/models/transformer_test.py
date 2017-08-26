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
VOCAB_SIZE = 9


class TransformerTest(tf.test.TestCase):

  def getModel(self):
    hparams = transformer.transformer_small()
    p_hparams = problem_hparams.test_problem_hparams(
        hparams, VOCAB_SIZE, VOCAB_SIZE)
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

    return transformer.Transformer(
        hparams, tf.contrib.learn.ModeKeys.INFER, p_hparams), features

  def testTransformer(self):
    model, features = self.getModel()
    shadred_logits, _ = model.model_fn(features)
    logits = tf.concat(shadred_logits, 0)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testBeamDecodeVsGreedy(self):
    model, features = self.getModel()

    decode_length = 20

    greedy_result, _, _ = model._greedy_infer(
        features, decode_length, last_position_only=True)
    greedy_result = tf.squeeze(greedy_result, axis=[2, 3])

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      beam_res = model._beam_decode(
          features,
          decode_length,
          beam_size=1,
          top_beams=1,
          last_position_only=True,
          alpha=1.0)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      greedy_res, beam_res = session.run([greedy_result, beam_res])

    self.assertEqual(beam_res.shape, (BATCH_SIZE, INPUT_LENGTH + decode_length))
    self.assertAllClose(greedy_res, beam_res)


if __name__ == "__main__":
  tf.test.main()
