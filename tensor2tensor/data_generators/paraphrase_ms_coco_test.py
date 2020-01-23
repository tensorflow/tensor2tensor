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

"""Tests for tensor2tensor.data_generators.paraphrase_ms_coco."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock

from tensor2tensor.data_generators import paraphrase_ms_coco

import tensorflow.compat.v1 as tf


class ParaphraseGenerationProblemTest(tf.test.TestCase):

  def testCombinationPairs(self):
    inputs = ["A", "B", "C"]
    expected_combination = [("A", "B"), ("A", "C"), ("B", "C")]
    actual_combination = paraphrase_ms_coco.create_combination(inputs)
    self.assertEqual(actual_combination, expected_combination)

  @mock.patch("tensor2tensor.data_generators"
              ".paraphrase_ms_coco.ParaphraseGenerationProblem.prepare_data",
              return_value=[("sentence1", "sentence2")])
  @mock.patch("tensor2tensor.data_generators"
              ".paraphrase_ms_coco.ParaphraseGenerationProblem.bidirectional")
  def testBidirectionalTrue(self, data, bidirectional):
    paraphrase_problem = paraphrase_ms_coco.ParaphraseGenerationProblem()
    paraphrase_problem.bidirectional = True

    expected_generated_data = [{"inputs": "sentence1", "targets": "sentence2"},
                               {"inputs": "sentence2", "targets": "sentence1"}]
    actual_generated_data = list(paraphrase_problem
                                 .generate_samples("data_dir",
                                                   "tmp_dir",
                                                   "dataset_split"))
    self.assertEqual(actual_generated_data, expected_generated_data)

  @mock.patch("tensor2tensor.data_generators"
              ".paraphrase_ms_coco.ParaphraseGenerationProblem.prepare_data",
              return_value=[("sentence1", "sentence2")])
  @mock.patch("tensor2tensor.data_generators"
              ".paraphrase_ms_coco.ParaphraseGenerationProblem.bidirectional")
  def testBidirectionalFalse(self, data, bidirectional):
    paraphrase_problem = paraphrase_ms_coco.ParaphraseGenerationProblem()
    paraphrase_problem.bidirectional = False

    expected_generated_data = [{"inputs": "sentence1", "targets": "sentence2"}]
    actual_generated_data = list(paraphrase_problem
                                 .generate_samples("data_dir",
                                                   "tmp_dir",
                                                   "dataset_split"))
    self.assertEqual(actual_generated_data, expected_generated_data)


if __name__ == "__main__":
  tf.test.main()
