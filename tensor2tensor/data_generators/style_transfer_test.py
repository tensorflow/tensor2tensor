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
"""Tests for tensor2tensor.data_generators.style_transfer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import style_transfer
import tensorflow as tf


class StyleTransferProblemShakespeareTest(tf.test.TestCase):

  def testSourceAndTargetPathsTrainModern2Shakespeare(self):
    tmp_dir = "tmp_dir"
    modern_to_shakespeare_data_gen = (
        style_transfer.StyleTransferModernToShakespeare())
    actual_source, actual_target = (
        modern_to_shakespeare_data_gen.source_target_paths(
            problem.DatasetSplit.TRAIN, tmp_dir))

    expected_source = "{}/train.modern".format(tmp_dir)
    expected_target = "{}/train.original".format(tmp_dir)

    self.assertEqual(actual_source, expected_source)
    self.assertEqual(actual_target, expected_target)

  def testSourceAndTargetPathsTrainShakespeare2Modern(self):
    tmp_dir = "tmp_dir"
    shakespeare_to_modern_data_gen = (
        style_transfer.StyleTransferShakespeareToModern())
    actual_source, actual_target = (
        shakespeare_to_modern_data_gen.source_target_paths(
            problem.DatasetSplit.TRAIN, tmp_dir))

    expected_source = "{}/train.original".format(tmp_dir)
    expected_target = "{}/train.modern".format(tmp_dir)

    self.assertEqual(actual_source, expected_source)
    self.assertEqual(actual_target, expected_target)

  def testSourceAndTargetPathsDevModern2Shakespeare(self):
    tmp_dir = "tmp_dir"
    modern_to_shakespeare_data_gen = (
        style_transfer.StyleTransferModernToShakespeare())
    actual_source, actual_target = (
        modern_to_shakespeare_data_gen.source_target_paths(
            problem.DatasetSplit.EVAL, tmp_dir))

    expected_source = "{}/dev.modern".format(tmp_dir)
    expected_target = "{}/dev.original".format(tmp_dir)

    self.assertEqual(actual_source, expected_source)
    self.assertEqual(actual_target, expected_target)

  def testSourceAndTargetPathsDevShakespeare2Modern(self):
    tmp_dir = "tmp_dir"
    shakespeare_to_modern_data_gen = (
        style_transfer.StyleTransferShakespeareToModern())
    actual_source, actual_target = (
        shakespeare_to_modern_data_gen.source_target_paths(
            problem.DatasetSplit.EVAL, tmp_dir))

    expected_source = "{}/dev.original".format(tmp_dir)
    expected_target = "{}/dev.modern".format(tmp_dir)

    self.assertEqual(actual_source, expected_source)
    self.assertEqual(actual_target, expected_target)


if __name__ == "__main__":
  tf.test.main()
