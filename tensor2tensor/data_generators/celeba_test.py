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

"""Tests for CelebA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2tensor.data_generators import celeba

import tensorflow as tf


class CelebaTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("Default", None),
      ("Area", "AREA"),
      ("Dilated", "DILATED"))
  def testCelebaMultiResolutionPreprocessExample(self, resize_method):
    example = {"inputs": tf.random_uniform([218, 178, 3], minval=-1.)}
    mode = tf.estimator.ModeKeys.TRAIN
    hparams = tf.contrib.training.HParams(resolutions=[8, 16, 32])
    if resize_method is not None:
      hparams.resize_method = resize_method

    problem = celeba.ImageCelebaMultiResolution()
    preprocessed_example = problem.preprocess_example(example, mode, hparams)
    self.assertLen(preprocessed_example, 2)
    self.assertEqual(preprocessed_example["inputs"].shape, (138, 138, 3))
    self.assertEqual(preprocessed_example["targets"].shape, (42, 32, 3))


if __name__ == "__main__":
  tf.test.main()
