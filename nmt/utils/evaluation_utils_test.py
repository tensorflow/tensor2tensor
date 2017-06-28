# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Tests for evaluation_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ..utils import evaluation_utils


class EvaluationUtilsTest(tf.test.TestCase):

  def testEvaluate(self):
    output = "nmt/testdata/deen_output"
    ref_bpe = "nmt/testdata/deen_ref_bpe"

    bpe_delimiter = "@@"

    expected_blue_score = 22.5855084573
    expected_rouge_score = 50.8429782599

    bleu_score = evaluation_utils.evaluate(
        ref_bpe, output, "bleu", bpe_delimiter)
    rouge_score = evaluation_utils.evaluate(
        ref_bpe, output, "rouge", bpe_delimiter)

    self.assertAlmostEqual(expected_blue_score, bleu_score)
    self.assertAlmostEqual(expected_rouge_score, rouge_score)

  def testAccuracy(self):
    pred_output = "nmt/testdata/pred_output"
    label_ref = "nmt/testdata/label_ref"

    expected_accuracy_score = 60.00

    accuracy_score = evaluation_utils.evaluate(
        label_ref, pred_output, "accuracy")
    self.assertAlmostEqual(expected_accuracy_score, accuracy_score)

if __name__ == "__main__":
  tf.test.main()
