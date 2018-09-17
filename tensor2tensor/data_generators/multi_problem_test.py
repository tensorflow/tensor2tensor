# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Tests for MultiProblem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import multi_problem
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.layers import modalities

import tensorflow as tf


class TestMultiProblem(multi_problem.MultiProblem):
  """Test multi-problem."""

  def __init__(self):
    super(TestMultiProblem, self).__init__()
    self.task_list.append(problem_hparams.TestProblem(2, 3))
    self.task_list.append(problem_hparams.TestProblem(4, 6))


class MultiProblemTest(tf.test.TestCase):

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsModality(self):
    problem = TestMultiProblem()
    p_hparams = problem.get_hparams()
    self.assertIsInstance(p_hparams.input_modality["inputs"],
                          modalities.SymbolModality)
    self.assertEqual(p_hparams.input_modality["inputs"].top_dimensionality, 3)
    self.assertIsInstance(p_hparams.target_modality, modalities.SymbolModality)
    self.assertEqual(p_hparams.target_modality.top_dimensionality, 5)

if __name__ == "__main__":
  tf.test.main()
