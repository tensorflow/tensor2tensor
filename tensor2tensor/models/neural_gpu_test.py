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
"""Tests for Neural GPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.layers import common_hparams
from tensor2tensor.models import neural_gpu

import tensorflow as tf


class NeuralGPUTest(tf.test.TestCase):

  def testNeuralGPU(self):
    hparams = common_hparams.basic_params1()
    batch_size = 3
    input_length = 5
    target_length = input_length
    input_vocab_size = 9
    target_vocab_size = 11
    p_hparams = problem_hparams.test_problem_hparams(input_vocab_size,
                                                     target_vocab_size)
    inputs = -1 + np.random.random_integers(
        input_vocab_size, size=(batch_size, input_length, 1, 1))
    targets = -1 + np.random.random_integers(
        target_vocab_size, size=(batch_size, target_length, 1, 1))
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(inputs, dtype=tf.int32),
          "targets": tf.constant(targets, dtype=tf.int32)
      }
      model = neural_gpu.NeuralGPU(hparams, tf.estimator.ModeKeys.TRAIN,
                                   p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (batch_size, target_length, 1, 1,
                                 target_vocab_size))


if __name__ == "__main__":
  tf.test.main()
