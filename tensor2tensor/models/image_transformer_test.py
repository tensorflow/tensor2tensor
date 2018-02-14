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

"""Tests for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import image_transformer

import tensorflow as tf


class ImagetransformerTest(tf.test.TestCase):

  def _testImagetransformer(self, net):
    batch_size = 3
    size = 7
    vocab_size = 256
    hparams = image_transformer.imagetransformer_base()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size)
    inputs = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, 1, 1, 1))
    targets = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, size, size, 3))
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(inputs, dtype=tf.int32),
          "targets": tf.constant(targets, dtype=tf.int32),
          "target_space_id": tf.constant(1, dtype=tf.int32),
      }
      model = net(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (batch_size, size, size, 3, vocab_size))

  def testImagetransformer(self):
    self._testImagetransformer(image_transformer.Imagetransformer)


if __name__ == "__main__":
  tf.test.main()
