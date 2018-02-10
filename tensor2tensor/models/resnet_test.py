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

"""Resnet tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import resnet
from tensor2tensor.utils import registry

import tensorflow as tf


def resnet_tiny_cpu():
  hparams = resnet.resnet_base()
  hparams.layer_sizes = [2, 2, 2, 2]
  hparams.use_nchw = False
  return hparams


class ResnetTest(tf.test.TestCase):

  def _testResnet(self, img_size, output_size):
    vocab_size = 9
    batch_size = 2
    x = np.random.random_integers(
        0, high=255, size=(batch_size, img_size, img_size, 3))
    y = np.random.random_integers(
        1, high=vocab_size - 1, size=(batch_size, 1, 1, 1))
    hparams = resnet_tiny_cpu()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size)
    p_hparams.input_modality["inputs"] = (registry.Modalities.IMAGE, None)
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = resnet.Resnet(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (batch_size,) + output_size + (1, vocab_size))

  def testResnetLarge(self):
    self._testResnet(img_size=224, output_size=(1, 1))


if __name__ == "__main__":
  tf.test.main()
