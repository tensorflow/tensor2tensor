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

"""Tests for Rouge metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from tensor2tensor.utils import rouge

import tensorflow as tf


class TestRouge2Metric(tf.test.TestCase):
  """Tests the rouge-2 metric."""

  def testRouge2Identical(self):
    hypotheses = np.array([[1, 2, 3, 4, 5, 1, 6, 7, 0],
                           [1, 2, 3, 4, 5, 1, 6, 8, 7]])
    references = np.array([[1, 2, 3, 4, 5, 1, 6, 7, 0],
                           [1, 2, 3, 4, 5, 1, 6, 8, 7]])
    self.assertAllClose(rouge.rouge_n(hypotheses, references), 1.0, atol=1e-03)

  def testRouge2Disjoint(self):
    hypotheses = np.array([[1, 2, 3, 4, 5, 1, 6, 7, 0],
                           [1, 2, 3, 4, 5, 1, 6, 8, 7]])
    references = np.array([[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                           [9, 10, 11, 12, 13, 14, 15, 16, 17, 0]])
    self.assertEqual(rouge.rouge_n(hypotheses, references), 0.0)

  def testRouge2PartialOverlap(self):
    hypotheses = np.array([[1, 2, 3, 4, 5, 1, 6, 7, 0],
                           [1, 2, 3, 4, 5, 1, 6, 8, 7]])
    references = np.array([[1, 9, 2, 3, 4, 5, 1, 10, 6, 7],
                           [1, 9, 2, 3, 4, 5, 1, 10, 6, 7]])
    self.assertAllClose(rouge.rouge_n(hypotheses, references), 0.53, atol=1e-03)


class TestRougeLMetric(tf.test.TestCase):
  """Tests the rouge-l metric."""

  def testRougeLIdentical(self):
    hypotheses = np.array([[1, 2, 3, 4, 5, 1, 6, 7, 0],
                           [1, 2, 3, 4, 5, 1, 6, 8, 7]])
    references = np.array([[1, 2, 3, 4, 5, 1, 6, 7, 0],
                           [1, 2, 3, 4, 5, 1, 6, 8, 7]])
    self.assertAllClose(
        rouge.rouge_l_sentence_level(hypotheses, references), 1.0, atol=1e-03)

  def testRougeLDisjoint(self):
    hypotheses = np.array([[1, 2, 3, 4, 5, 1, 6, 7, 0],
                           [1, 2, 3, 4, 5, 1, 6, 8, 7]])
    references = np.array([[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                           [9, 10, 11, 12, 13, 14, 15, 16, 17, 0]])
    self.assertEqual(rouge.rouge_l_sentence_level(hypotheses, references), 0.0)

  def testRougeLPartialOverlap(self):
    hypotheses = np.array([[1, 2, 3, 4, 5, 1, 6, 7, 0],
                           [1, 2, 3, 4, 5, 1, 6, 8, 7]])
    references = np.array([[1, 9, 2, 3, 4, 5, 1, 10, 6, 7],
                           [1, 9, 2, 3, 4, 5, 1, 10, 6, 7]])
    self.assertAllClose(
        rouge.rouge_l_sentence_level(hypotheses, references), 0.837, atol=1e-03)


class TestRougeMetricsE2E(tf.test.TestCase):
  """Tests the rouge metrics end-to-end."""

  def testRouge2MetricE2E(self):
    vocab_size = 4
    batch_size = 12
    seq_length = 12
    predictions = tf.one_hot(
        np.random.randint(vocab_size, size=(batch_size, seq_length, 1, 1)),
        depth=4,
        dtype=tf.float32)
    targets = np.random.randint(4, size=(12, 12, 1, 1))
    with self.test_session() as session:
      scores, _ = rouge.rouge_2_fscore(predictions,
                                       tf.constant(targets, dtype=tf.int32))
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      session.run(a)

  def testRougeLMetricE2E(self):
    vocab_size = 4
    batch_size = 12
    seq_length = 12
    predictions = tf.one_hot(
        np.random.randint(vocab_size, size=(batch_size, seq_length, 1, 1)),
        depth=4,
        dtype=tf.float32)
    targets = np.random.randint(4, size=(12, 12, 1, 1))
    with self.test_session() as session:
      scores, _ = rouge.rouge_l_fscore(
          predictions,
          tf.constant(targets, dtype=tf.int32))
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      session.run(a)


if __name__ == "__main__":
  tf.test.main()
