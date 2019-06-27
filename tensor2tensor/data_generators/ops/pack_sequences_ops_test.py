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

"""Tests for pack_sequences_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.ops import pack_sequences_ops
import tensorflow as tf


class PackSequencesOpsTest(tf.test.TestCase):

  def test_pack_sequences(self):
    inputs = [
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
    ]
    targets = [
        [10, 0, 0],
        [20, 30, 40],
        [50, 60, 0],
    ]
    max_length = 5
    (inputs_packed, inputs_segmentation, inputs_position,
     targets_packed, targets_segmentation, targets_position) = (
         pack_sequences_ops.pack_sequences2(inputs, targets, max_length))
    self.assertAllEqual(
        inputs_packed, [
            [1, 2, 3, 4, 5],
            [6, 0, 0, 0, 0],
        ])
    self.assertAllEqual(
        inputs_segmentation, [
            [1, 1, 1, 2, 2],
            [1, 0, 0, 0, 0],
        ])
    self.assertAllEqual(
        inputs_position, [
            [0, 1, 2, 0, 1],
            [0, 0, 0, 0, 0],
        ])
    self.assertAllEqual(
        targets_packed, [
            [10, 20, 30, 40, 0],
            [50, 60, 0, 0, 0],
        ])
    self.assertAllEqual(
        targets_segmentation, [
            [1, 2, 2, 2, 0],
            [1, 1, 0, 0, 0],
        ])
    self.assertAllEqual(
        targets_position, [
            [0, 0, 1, 2, 0],
            [0, 1, 0, 0, 0],
        ])


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
