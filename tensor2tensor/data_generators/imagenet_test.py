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

"""Tests for ImageNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2tensor.data_generators import imagenet
from tensor2tensor.utils import hparam

import tensorflow as tf


class ImagenetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("Default", None),
      ("Area", "AREA"),
      ("Dilated", "DILATED"))
  def testImagenetMultiResolutionPreprocessExample(self, resize_method):
    example = {"inputs": tf.random_uniform([64, 64, 3], minval=-1.)}
    mode = tf.estimator.ModeKeys.TRAIN
    hparams = hparam.HParams(resolutions=[8, 16, 32])
    if resize_method is not None:
      hparams.resize_method = resize_method

    problem = imagenet.ImageImagenetMultiResolutionGen()
    preprocessed_example = problem.preprocess_example(example, mode, hparams)
    self.assertLen(preprocessed_example, 1)
    self.assertEqual(preprocessed_example["inputs"].shape, (42, 32, 3))

  def testImagenetIsNormalized(self):
    problem = imagenet.ImageImagenet224()
    self.assertTrue(problem.normalize_image)
    problem = imagenet.ImageImagenet224NoNormalization()
    self.assertFalse(problem.normalize_image)


if __name__ == "__main__":
  tf.test.main()
