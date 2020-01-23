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

"""Tests for tensor2tensor.data_generators.transduction_problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

from absl.testing import parameterized

import numpy as np

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import transduction_problems

import tensorflow.compat.v1 as tf


class TransductionProblem(parameterized.TestCase):

  def setUp(self):
    super(TransductionProblem, self).setUp()
    # Create a temporary directory
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    super(TransductionProblem, self).tearDown()
    # Remove the directory after the test
    shutil.rmtree(self.test_dir)

  @parameterized.named_parameters(
      ('CopySequence',
       transduction_problems.CopySequence(),
       lambda x: x),
      ('CopySequenceSmall',
       transduction_problems.CopySequenceSmall(),
       lambda x: x),
      ('FlipBiGramSequence',
       transduction_problems.FlipBiGramSequence(),
       lambda x: [x[i+1] if i%2 == 0 else x[i-1] for i in range(len(x))]),
      ('ReverseSequence',
       transduction_problems.ReverseSequence(),
       lambda x: x[::-1]),
      ('ReverseSequenceSmall',
       transduction_problems.ReverseSequenceSmall(),
       lambda x: x[::-1]),
  )
  def testTransduction(self, p, transformation):
    data_dir = ''
    dataset_split = problem.DatasetSplit.TEST
    for sample in p.generate_samples(data_dir, self.test_dir, dataset_split):
      input_tokens = sample['inputs'].split(' ')
      target_tokens = sample['targets'].split(' ')
      self.assertBetween(len(input_tokens),
                         p.min_sequence_length(dataset_split),
                         p.max_sequence_length(dataset_split))
      self.assertBetween(len(target_tokens),
                         p.min_sequence_length(dataset_split),
                         p.max_sequence_length(dataset_split))

      transformed_inputs = np.array(transformation(input_tokens))

      np.testing.assert_equal(transformed_inputs, target_tokens)


if __name__ == '__main__':
  tf.test.main()
