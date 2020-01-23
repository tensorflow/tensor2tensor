# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Tests for tensor2tensor.utils.sari_hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from tensor2tensor.utils import sari_hook
import tensorflow.compat.v1 as tf


class SariHookTest(tf.test.TestCase):

  def setUp(self):
    """Sets up inputs and references from the paper's test cases."""
    self.input_sentence = (
        "About 95 species are currently accepted .".split())
    self.references = [
        "About 95 species are currently known .".split(),
        "About 95 species are now accepted .".split(),
        "95 species are now accepted .".split(),
    ]

  def testSariSent1(self):
    """Test case 1 from SARI-paper.

    The score is slightly different from what is reported in the paper (0.2683)
    since the authors' code seems to contain a bug in the keep recall score
    computation.
    """
    output = "About 95 you now get in ." .split()
    score, _, _, _ = sari_hook.get_sari_score(self.input_sentence, output,
                                              self.references)
    self.assertAlmostEqual(0.2695360, score)

  def testSariSent2(self):
    """Test case 2 from SARI-paper."""
    output = "About 95 species are now agreed .".split()
    score, _, _, _ = sari_hook.get_sari_score(self.input_sentence, output,
                                              self.references)
    self.assertAlmostEqual(0.6170966, score)

  def testSariSent3(self):
    """Test case 3 from SARI-paper."""
    output = "About 95 species are currently agreed .".split()
    score, _, _, _ = sari_hook.get_sari_score(self.input_sentence, output,
                                              self.references)
    self.assertAlmostEqual(0.5088682, score)

  def testMatchingSentences(self):
    """If input=output=reference, the score should be 1."""
    input_sentence = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    output = input_sentence
    references = [input_sentence]
    score, _, _, _ = sari_hook.get_sari_score(input_sentence, output,
                                              references)
    self.assertEqual(1, score)

  def testMatchingOutputAndReference(self):
    """If output=reference, the score should be 1."""
    input_sentence = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    output = [3, 1, 4, 1, 80, 70]
    references = [output]
    score, _, _, _ = sari_hook.get_sari_score(input_sentence, output,
                                              references)
    self.assertEqual(1, score)

  def testMatchingSentencesWithRepetitions(self):
    """Token frequencies should not matter if we only consider unigrams."""
    input_sentence = [3, 1, 4]
    output = [3, 3, 1, 1, 1, 4]
    references = [[3, 3, 3, 1, 1, 4, 4]]
    score, _, _, _ = sari_hook.get_sari_score(input_sentence, output,
                                              references, max_gram_size=1)
    self.assertEqual(1, score)

  def testKeepScore(self):
    """Toy example where Input='1 2', Output='2', References=['1 2', 1']."""
    # Unigram counts.
    source_counts = collections.Counter({1: 1, 2: 1})
    prediction_counts = collections.Counter({2: 1})
    target_counts = collections.Counter({1: 1, 2: 0.5})
    score = sari_hook.get_keep_score(source_counts, prediction_counts,
                                     target_counts)
    self.assertAlmostEqual(6.0/15, score)

  def testDeletionScore(self):
    """Toy example where Input='1 2', Output='1 2', References=['1']."""
    # Unigram counts.
    source_counts = collections.Counter({1: 1, 2: 1})
    prediction_counts = collections.Counter({1: 1, 2: 1})
    target_counts = collections.Counter({1: 1})
    # Output doesn't drop any (incorrect) tokens from the input so precision
    # should be 1, but since '2' is not dropped, recall should be 0. Thus we
    # should have F1=0 and F0=precision=1.
    f1_score = sari_hook.get_deletion_score(source_counts, prediction_counts,
                                            target_counts, beta=1)
    self.assertEqual(0, f1_score)
    f0_score = sari_hook.get_deletion_score(source_counts, prediction_counts,
                                            target_counts, beta=0)
    self.assertEqual(1, f0_score)

  def testIdsWithZeros(self):
    """Zeros should be ignored."""
    input_sentence = [3, 1, 4, 0, 0, 0]
    output = [3, 1, 4]
    references = [[3, 1, 4, 0, 0, 0, 0, 0]]
    score, _, _, _ = sari_hook.get_sari_score(input_sentence, output,
                                              references)
    self.assertEqual(1, score)

  def testSariScoreE2E(self):
    """Tests the SARI metrics end-to-end."""
    predictions = np.random.randint(4, size=(12, 12, 1, 1, 12))
    targets = np.random.randint(4, size=(12, 12, 1, 1))
    inputs = np.random.randint(4, size=(12, 12, 1, 1))
    with self.test_session() as session:
      scores, _ = sari_hook.sari_score(
          predictions=tf.constant(predictions, dtype=tf.int32),
          labels=tf.constant(targets, dtype=tf.int32),
          features={
              "inputs": tf.constant(inputs, dtype=tf.int32),
          })
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      session.run(a)


if __name__ == "__main__":
  tf.test.main()
