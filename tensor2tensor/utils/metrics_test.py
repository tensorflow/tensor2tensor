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

"""Tests for tensor2tensor.utils.metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.utils import metrics

import tensorflow as tf


class MetricsTest(tf.test.TestCase):

  def testAccuracyMetric(self):
    predictions = np.random.randint(1, 5, size=(12, 12, 12, 1))
    targets = np.random.randint(1, 5, size=(12, 12, 12, 1))
    expected = np.mean((predictions == targets).astype(float))
    with self.test_session() as session:
      scores, _ = metrics.padded_accuracy(
          tf.one_hot(predictions, depth=5, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32))
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      actual = session.run(a)
    self.assertAlmostEqual(actual, expected)

  def testAccuracyTopKMetric(self):
    predictions = np.random.randint(1, 5, size=(12, 12, 12, 1))
    targets = np.random.randint(1, 5, size=(12, 12, 12, 1))
    expected = np.mean((predictions == targets).astype(float))
    with self.test_session() as session:
      predicted = tf.one_hot(predictions, depth=5, dtype=tf.float32)
      scores1, _ = metrics.padded_accuracy_topk(
          predicted, tf.constant(targets, dtype=tf.int32), k=1)
      scores2, _ = metrics.padded_accuracy_topk(
          predicted, tf.constant(targets, dtype=tf.int32), k=7)
      a1 = tf.reduce_mean(scores1)
      a2 = tf.reduce_mean(scores2)
      session.run(tf.global_variables_initializer())
      actual1, actual2 = session.run([a1, a2])
    self.assertAlmostEqual(actual1, expected)
    self.assertAlmostEqual(actual2, 1.0)

  def testSequenceAccuracyMetric(self):
    predictions = np.random.randint(4, size=(12, 12, 12, 1))
    targets = np.random.randint(4, size=(12, 12, 12, 1))
    expected = np.mean(
        np.prod((predictions == targets).astype(float), axis=(1, 2)))
    with self.test_session() as session:
      scores, _ = metrics.padded_sequence_accuracy(
          tf.one_hot(predictions, depth=4, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32))
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      actual = session.run(a)
    self.assertEqual(actual, expected)

  def testRMSEMetric(self):
    predictions = np.full((10, 1), 1)  # All 1's
    targets = np.full((10, 1), 3)  # All 3's
    expected = np.sqrt(np.mean((predictions - targets)**2))  # RMSE = 2.0
    with self.test_session() as session:
      rmse, _ = metrics.padded_rmse(
          tf.constant(predictions, dtype=tf.int32),
          tf.constant(targets, dtype=tf.int32))
      session.run(tf.global_variables_initializer())
      actual = session.run(rmse)
    self.assertEqual(actual, expected)

  def testSequenceEditDistanceMetric(self):
    predictions = np.array([[3, 4, 5, 1, 0, 0],
                            [2, 1, 3, 4, 0, 0],
                            [2, 1, 3, 4, 0, 0]])
    # Targets are just a bit different:
    #  - first sequence has a different prediction
    #  - second sequence has a different prediction and one extra step
    #  - third sequence is identical
    targets = np.array([[5, 4, 5, 1, 0, 0],
                        [2, 5, 3, 4, 1, 0],
                        [2, 1, 3, 4, 0, 0]])
    # Reshape to match expected input format by metric fns.
    predictions = np.reshape(predictions, [3, 6, 1, 1])
    targets = np.reshape(targets, [3, 6, 1, 1])
    with self.test_session() as session:
      scores, weight = metrics.sequence_edit_distance(
          tf.one_hot(predictions, depth=6, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32))
      session.run(tf.global_variables_initializer())
      actual_scores, actual_weight = session.run([scores, weight])
    self.assertAlmostEqual(actual_scores, 3.0 / 13)
    self.assertEqual(actual_weight, 13)

  def testWordErrorRateMetric(self):

    ref = np.asarray([
        # a b c
        [97, 34, 98, 34, 99],
        [97, 34, 98, 34, 99],
        [97, 34, 98, 34, 99],
        [97, 34, 98, 34, 99],
    ])

    hyp = np.asarray([
        [97, 34, 98, 34, 99],  # a b c
        [97, 34, 98, 0, 0],  # a b
        [97, 34, 98, 34, 100],  # a b d
        [0, 0, 0, 0, 0]  # empty
    ])

    labels = np.reshape(ref, ref.shape + (1, 1))
    predictions = np.zeros((len(ref), np.max([len(s) for s in hyp]), 1, 1, 256))

    for i, sample in enumerate(hyp):
      for j, idx in enumerate(sample):
        predictions[i, j, 0, 0, idx] = 1

    with self.test_session() as session:
      actual_wer, unused_actual_ref_len = session.run(
          metrics.word_error_rate(predictions, labels))

    expected_wer = 0.417
    places = 3
    self.assertAlmostEqual(round(actual_wer, places), expected_wer, places)

  def testNegativeLogPerplexity(self):
    predictions = np.random.randint(4, size=(12, 12, 12, 1))
    targets = np.random.randint(4, size=(12, 12, 12, 1))
    with self.test_session() as session:
      scores, _ = metrics.padded_neg_log_perplexity(
          tf.one_hot(predictions, depth=4, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32))
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      actual = session.run(a)
    self.assertEqual(actual.shape, ())

  def testNegativeLogPerplexityMasked(self):
    predictions = np.random.randint(4, size=(12, 12, 12, 1))
    targets = np.random.randint(4, size=(12, 12, 12, 1))
    features = {
        'targets_mask': tf.to_float(tf.ones([12, 12]))
    }
    with self.test_session() as session:
      scores, _ = metrics.padded_neg_log_perplexity_with_masking(
          tf.one_hot(predictions, depth=4, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32),
          features)
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      actual = session.run(a)
    self.assertEqual(actual.shape, ())

  def testNegativeLogPerplexityMaskedAssert(self):
    predictions = np.random.randint(4, size=(12, 12, 12, 1))
    targets = np.random.randint(4, size=(12, 12, 12, 1))
    features = {}

    with self.assertRaisesRegexp(
        ValueError,
        'masked_neg_log_perplexity requires targets_mask feature'):
      with self.test_session() as session:
        scores, _ = metrics.padded_neg_log_perplexity_with_masking(
            tf.one_hot(predictions, depth=4, dtype=tf.float32),
            tf.constant(targets, dtype=tf.int32),
            features)
        a = tf.reduce_mean(scores)
        session.run(tf.global_variables_initializer())
        _ = session.run(a)

  def testSigmoidAccuracyOneHot(self):
    logits = np.array([
        [-1., 1.],
        [1., -1.],
        [-1., 1.],
        [1., -1.]
    ])
    labels = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
    ])
    logits = np.expand_dims(np.expand_dims(logits, 1), 1)
    labels = np.expand_dims(np.expand_dims(labels, 1), 1)

    with self.test_session() as session:
      score, _ = metrics.sigmoid_accuracy_one_hot(logits, labels)
      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      s = session.run(score)
    self.assertEqual(s, 0.5)

  def testSigmoidPrecisionOneHot(self):
    logits = np.array([
        [-1., 1.],
        [1., -1.],
        [1., -1.],
        [1., -1.]
    ])
    labels = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]
    ])
    logits = np.expand_dims(np.expand_dims(logits, 1), 1)
    labels = np.expand_dims(np.expand_dims(labels, 1), 1)

    with self.test_session() as session:
      score, _ = metrics.sigmoid_precision_one_hot(logits, labels)
      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      s = session.run(score)
    self.assertEqual(s, 0.25)

  def testSigmoidRecallOneHot(self):
    logits = np.array([
        [-1., 1.],
        [1., -1.],
        [1., -1.],
        [1., -1.]
    ])
    labels = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]
    ])
    logits = np.expand_dims(np.expand_dims(logits, 1), 1)
    labels = np.expand_dims(np.expand_dims(labels, 1), 1)

    with self.test_session() as session:
      score, _ = metrics.sigmoid_recall_one_hot(logits, labels)
      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      s = session.run(score)
    self.assertEqual(s, 0.25)

  def testSigmoidCrossEntropyOneHot(self):
    logits = np.array([
        [-1., 1.],
        [1., -1.],
        [1., -1.],
        [1., -1.]
    ])
    labels = np.array([
        [0, 1],
        [1, 0],
        [0, 0],
        [0, 1]
    ])
    logits = np.expand_dims(np.expand_dims(logits, 1), 1)
    labels = np.expand_dims(np.expand_dims(labels, 1), 1)

    with self.test_session() as session:
      score, _ = metrics.sigmoid_cross_entropy_one_hot(logits, labels)
      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      s = session.run(score)
    self.assertAlmostEqual(s, 0.688, places=3)

  def testRocAuc(self):
    logits = np.array([
        [-1., 1.],
        [1., -1.],
        [1., -1.],
        [1., -1.]
    ])
    labels = np.array([
        [1],
        [0],
        [1],
        [0]
    ])
    logits = np.expand_dims(np.expand_dims(logits, 1), 1)
    labels = np.expand_dims(np.expand_dims(labels, 1), 1)

    with self.test_session() as session:
      score, _ = metrics.roc_auc(logits, labels)
      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      s = session.run(score)
    self.assertAlmostEqual(s, 0.750, places=3)

  def testMultilabelMatch3(self):
    predictions = np.random.randint(1, 5, size=(100, 1, 1, 1))
    targets = np.random.randint(1, 5, size=(100, 10, 1, 1))
    weights = np.random.randint(0, 2, size=(100, 1, 1, 1))
    targets *= weights

    predictions_repeat = np.repeat(predictions, 10, axis=1)
    expected = (predictions_repeat == targets).astype(float)
    expected = np.sum(expected, axis=(1, 2, 3))
    expected = np.minimum(expected / 3.0, 1.)
    expected = np.sum(expected * weights[:, 0, 0, 0]) / weights.shape[0]
    with self.test_session() as session:
      scores, weights_ = metrics.multilabel_accuracy_match3(
          tf.one_hot(predictions, depth=5, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32))
      a, a_op = tf.metrics.mean(scores, weights_)
      session.run(tf.local_variables_initializer())
      session.run(tf.global_variables_initializer())
      _ = session.run(a_op)
      actual = session.run(a)
    self.assertAlmostEqual(actual, expected, places=6)

  def testPearsonCorrelationCoefficient(self):
    predictions = np.random.rand(12, 1)
    targets = np.random.rand(12, 1)

    expected = np.corrcoef(np.squeeze(predictions), np.squeeze(targets))[0][1]
    with self.test_session() as session:
      pearson, _ = metrics.pearson_correlation_coefficient(
          tf.constant(predictions, dtype=tf.float32),
          tf.constant(targets, dtype=tf.float32))
      session.run(tf.global_variables_initializer())
      session.run(tf.local_variables_initializer())
      actual = session.run(pearson)
    self.assertAlmostEqual(actual, expected)


if __name__ == '__main__':
  tf.test.main()
