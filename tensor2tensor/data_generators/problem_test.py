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

"""Test for common problem functionalities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized  # for assertLen
import numpy as np

from tensor2tensor.data_generators import algorithmic
from tensor2tensor.data_generators import problem as problem_module
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.layers import modalities

import tensorflow as tf


def assert_tensors_equal(sess, t1, t2, n):
  """Compute tensors `n` times and ensure that they are equal."""

  for _ in range(n):

    v1, v2 = sess.run([t1, t2])

    if v1.shape != v2.shape:
      return False

    if not np.all(v1 == v2):
      return False

  return True


class ProblemTest(parameterized.TestCase, tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    algorithmic.TinyAlgo.setup_for_test()

  def testNoShuffleDeterministic(self):
    problem = algorithmic.TinyAlgo()
    dataset = problem.dataset(mode=tf.estimator.ModeKeys.TRAIN,
                              data_dir=algorithmic.TinyAlgo.data_dir,
                              shuffle_files=False)

    tensor1 = dataset.make_one_shot_iterator().get_next()["targets"]
    tensor2 = dataset.make_one_shot_iterator().get_next()["targets"]

    with tf.Session() as sess:
      self.assertTrue(assert_tensors_equal(sess, tensor1, tensor2, 20))

  def testNoShufflePreprocess(self):

    problem = algorithmic.TinyAlgo()
    dataset1 = problem.dataset(mode=tf.estimator.ModeKeys.TRAIN,
                               data_dir=algorithmic.TinyAlgo.data_dir,
                               shuffle_files=False, preprocess=False)
    dataset2 = problem.dataset(mode=tf.estimator.ModeKeys.TRAIN,
                               data_dir=algorithmic.TinyAlgo.data_dir,
                               shuffle_files=False, preprocess=True)

    tensor1 = dataset1.make_one_shot_iterator().get_next()["targets"]
    tensor2 = dataset2.make_one_shot_iterator().get_next()["targets"]

    with tf.Session() as sess:
      self.assertTrue(assert_tensors_equal(sess, tensor1, tensor2, 20))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsModality(self):
    problem = problem_hparams.TestProblem(input_vocab_size=2,
                                          target_vocab_size=3)
    p_hparams = problem.get_hparams()
    self.assertIsInstance(p_hparams.modality["inputs"],
                          modalities.SymbolModality)
    self.assertIsInstance(p_hparams.modality["targets"],
                          modalities.SymbolModality)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsModalityObj(self):
    class ModalityObjProblem(problem_module.Problem):

      def hparams(self, defaults, model_hparams):
        hp = defaults
        hp.modality = {"inputs": modalities.SymbolModality,
                       "targets": modalities.SymbolModality}
        hp.vocab_size = {"inputs": 2,
                         "targets": 3}

    problem = ModalityObjProblem(False, False)
    p_hparams = problem.get_hparams()
    self.assertIsInstance(p_hparams.modality["inputs"],
                          modalities.SymbolModality)
    self.assertIsInstance(p_hparams.modality["targets"],
                          modalities.SymbolModality)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsInputOnlyModality(self):
    class InputOnlyProblem(problem_module.Problem):

      def hparams(self, defaults, model_hparams):
        hp = defaults
        hp.modality = {"inputs": modalities.SymbolModality}
        hp.vocab_size = {"inputs": 2}

    problem = InputOnlyProblem(False, False)
    p_hparams = problem.get_hparams()
    self.assertIsInstance(p_hparams.modality["inputs"],
                          modalities.SymbolModality)
    self.assertLen(p_hparams.modality, 1)

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testProblemHparamsTargetOnlyModality(self):
    class TargetOnlyProblem(problem_module.Problem):

      def hparams(self, defaults, model_hparams):
        hp = defaults
        hp.modality = {"targets": modalities.SymbolModality}
        hp.vocab_size = {"targets": 3}

    problem = TargetOnlyProblem(False, False)
    p_hparams = problem.get_hparams()
    self.assertIsInstance(p_hparams.modality["targets"],
                          modalities.SymbolModality)
    self.assertLen(p_hparams.modality, 1)

if __name__ == "__main__":
  tf.test.main()
