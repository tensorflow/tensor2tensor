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

"""Tests for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.data_generators import celeba  # pylint: disable=unused-import
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import image_transformer_2d
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


class Img2imgTransformerTest(tf.test.TestCase):

  def _test_img2img_transformer(self, net):
    batch_size = 3
    hparams = image_transformer_2d.img2img_transformer2d_tiny()
    hparams.data_dir = ""
    p_hparams = registry.problem("image_celeba").get_hparams(hparams)
    inputs = np.random.randint(256, size=(batch_size, 4, 4, 3))
    targets = np.random.randint(256, size=(batch_size, 8, 8, 3))
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
    self.assertEqual(res.shape, (batch_size, 8, 8, 3, 256))

  def testImg2imgTransformer(self):
    self._test_img2img_transformer(image_transformer_2d.Img2imgTransformer)


class Imagetransformer2dTest(tf.test.TestCase):

  def _test_imagetransformer_2d(self, net):
    batch_size = 3
    size = 7
    vocab_size = 256
    hparams = image_transformer_2d.imagetransformer2d_tiny()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size,
                                                     vocab_size,
                                                     hparams)
    inputs = np.random.randint(
        vocab_size, size=(batch_size, 1, 1, 1))
    targets = np.random.randint(
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

  def testImagetransformer2d(self):
    self._test_imagetransformer_2d(image_transformer_2d.Imagetransformer2d)


if __name__ == "__main__":
  tf.test.main()
