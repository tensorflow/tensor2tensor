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

"""Xception tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import xception
from tensor2tensor.utils import registry

import tensorflow as tf


class XceptionTest(tf.test.TestCase):

  def _testXception(self, img_size, output_size):
    vocab_size = 9
    batch_size = 3
    x = np.random.random_integers(
        0, high=255, size=(batch_size, img_size, img_size, 3))
    y = np.random.random_integers(
        1, high=vocab_size - 1, size=(batch_size, 1, 1, 1))
    hparams = xception.xception_tiny()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size)
    p_hparams.input_modality["inputs"] = (registry.Modalities.IMAGE, None)
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = xception.Xception(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, output_size + (1, vocab_size))

  def testXceptionSmall(self):
    self._testXception(img_size=9, output_size=(3, 5, 5))

  def testXceptionLarge(self):
    self._testXception(img_size=256, output_size=(3, 8, 8))


if __name__ == "__main__":
  tf.test.main()
