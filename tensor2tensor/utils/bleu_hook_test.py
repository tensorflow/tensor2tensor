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

# coding=utf-8
"""Tests for tensor2tensor.utils.bleu_hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import six

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import bleu_hook

import tensorflow.compat.v1 as tf


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
    # The smoothing prevents 0 for small corpora
    actual_bleu = 0.0798679
    self.assertAllClose(bleu, actual_bleu, atol=1e-03)

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
    actual_bleu = 0.3436
    self.assertAllClose(bleu, actual_bleu, atol=1e-03)

  def testBleuTokenize(self):
    self.assertEqual(bleu_hook.bleu_tokenize(u"hi, “there”"),
                     [u"hi", u",", u"“", u"there", u"”"])

  def _generate_test_data(self, name, hyps, refs):
    """Writes test data to temporary files.

    Args:
      name: str, used for making temp files unique across tests
      hyps: list of unicode strings serving as translation hypotheses
      refs: list of unicode strings serving as references

    Returns:
      hyp_file: path to temporary file containing the hypotheses
      refs_file: path to temporary file containing the references
    """
    assert len(hyps) == len(refs)
    hyp_file = os.path.join(tempfile.gettempdir(), "{}.hyps".format(name))
    refs_file = os.path.join(tempfile.gettempdir(), "{}.refs".format(name))
    for filename, items in zip([hyp_file, refs_file], [hyps, refs]):
      with (open(filename, "wb")
            if six.PY2 else open(filename, "w", encoding="utf-8")) as out:
        content = text_encoder.unicode_to_native(u"\n".join(items))
        out.write(content)
    return hyp_file, refs_file

  def testBleuWrapper(self):
    hyp_filename, ref_filename = self._generate_test_data(
        "standard", [u"a b a c", u"e f g d"], [u"a b a z", u"y f g d k l m"])
    bleu = bleu_hook.bleu_wrapper(ref_filename, hyp_filename)
    actual_bleu = 0.3436
    self.assertAllClose(bleu, actual_bleu, atol=1e-03)

  def testBleuWrapperWithUnicodeLineSeparator(self):
    hyp_filename, ref_filename = self._generate_test_data(
        "unicode-linesep", [u"a b a c", u"e f \u2028 d"],
        [u"a b a z", u"y f g d k l m"])
    bleu = bleu_hook.bleu_wrapper(ref_filename, hyp_filename)
    actual_bleu = 0.2638
    self.assertAllClose(bleu, actual_bleu, atol=1e-03)


if __name__ == "__main__":
  tf.test.main()
