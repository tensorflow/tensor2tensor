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

"""Tests for tensor2tensor.utils.bleu_hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.utils import bleu_hook

import tensorflow as tf


class BleuHookTest(tf.test.TestCase):

  def testComputeBleuEqual(self):
    translation_corpus = [[1, 2, 3]]
    reference_corpus = [[1, 2, 3]]
    bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
    actual_bleu = 1.0
    self.assertEqual(bleu, actual_bleu)

  def testComputeNotEqual(self):
    translation_corpus = [[1, 2, 3, 4]]
    reference_corpus = [[5, 6, 7, 8]]
    bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
    actual_bleu = 0.0
    self.assertEqual(bleu, actual_bleu)

  def testComputeMultipleBatch(self):
    translation_corpus = [[1, 2, 3, 4], [5, 6, 7, 0]]
    reference_corpus = [[1, 2, 3, 4], [5, 6, 7, 10]]
    bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
    actual_bleu = 0.7231
    self.assertAllClose(bleu, actual_bleu, atol=1e-03)

  def testComputeMultipleNgrams(self):
    reference_corpus = [[1, 2, 1, 13], [12, 6, 7, 4, 8, 9, 10]]
    translation_corpus = [[1, 2, 1, 3], [5, 6, 7, 4]]
    bleu = bleu_hook.compute_bleu(reference_corpus, translation_corpus)
    actual_bleu = 0.486
    self.assertAllClose(bleu, actual_bleu, atol=1e-03)

if __name__ == '__main__':
  tf.test.main()
