# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.layers import modalities
from tensor2tensor.models import xception

import tensorflow as tf


class XceptionTest(tf.test.TestCase):

  def _test_xception(self, img_size):
    vocab_size = 9
    batch_size = 3
    x = np.random.randint(
        256, size=(batch_size, img_size, img_size, 3))
    y = np.random.randint(
        1, high=vocab_size, size=(batch_size, 1, 1, 1))
    hparams = xception.xception_tiny()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size,
                                                     vocab_size,
                                                     hparams)
    p_hparams.modality["inputs"] = modalities.ModalityType.IMAGE
    p_hparams.modality["targets"] = modalities.ModalityType.CLASS_LABEL
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = xception.Xception(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (batch_size, 1, 1, 1, vocab_size))

  def testXceptionSmallImage(self):
    self._test_xception(img_size=9)

  def testXceptionLargeImage(self):
    self._test_xception(img_size=256)


if __name__ == "__main__":
  tf.test.main()
