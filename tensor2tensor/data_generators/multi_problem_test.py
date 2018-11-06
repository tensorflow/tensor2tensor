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
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities

import tensorflow as tf


# TODO(trandustin): This test problem is required in order for MultiProblem
# to access vocab size via encoders. In a future change, enable MultiProblem to
# access vocab size more explicitly from the Problem.
class TestProblem(problem.Problem):
  """Test problem."""

  def __init__(self, input_vocab_size, target_vocab_size):
    super(TestProblem, self).__init__(False, False)
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

  def hparams(self, defaults, model_hparams):
    hp = defaults
    hp.input_modality = {
        "inputs": modalities.SymbolModality(model_hparams,
                                            self.input_vocab_size)
    }
    hp.target_modality = modalities.SymbolModality(model_hparams,
                                                   self.target_vocab_size)

  def feature_encoders(self, data_dir):
    encoders = {
        "inputs": text_encoder.ByteTextEncoder(),
        "targets": text_encoder.ByteTextEncoder(),
    }
    return encoders


class TestMultiProblem(multi_problem.MultiProblem):
  """Test multi-problem."""

  def __init__(self):
    super(TestMultiProblem, self).__init__()
    self.task_list.append(TestProblem(2, 3))
    self.task_list.append(TestProblem(4, 6))


class MultiProblemTest(tf.test.TestCase):

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsModality(self):
    multiproblem = TestMultiProblem()
    p_hparams = multiproblem.get_hparams()
    self.assertIsInstance(p_hparams.input_modality["inputs"],
                          modalities.SymbolModality)
    self.assertEqual(p_hparams.input_modality["inputs"].top_dimensionality, 2)
    self.assertIsInstance(p_hparams.target_modality, modalities.SymbolModality)
    self.assertEqual(p_hparams.target_modality.top_dimensionality, 260)

if __name__ == "__main__":
  tf.test.main()
