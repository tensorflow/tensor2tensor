# coding=utf-8
# Copyright 2021 The Tensor2Tensor Authors.
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
import tensorflow.compat.v1 as tf


class PackSequencesOpsTest(tf.test.TestCase):

  def test_pack_sequences2(self):
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
         pack_sequences_ops.pack_sequences2(
             inputs, targets, max_length, max_length))
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

  def test_pack_sequences_k(self):
    inputs = tf.convert_to_tensor([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
    ], dtype=tf.int32)
    targets = tf.convert_to_tensor([
        [10, 0, 0],
        [20, 30, 40],
        [50, 60, 0],
    ], dtype=tf.int32)
    max_length = tf.convert_to_tensor(5, dtype=tf.int32)
    (packed, segmentation, position) = pack_sequences_ops.pack_sequences_k(
        [inputs, targets], [max_length, max_length])
    (inputs_packed, targets_packed) = packed
    (inputs_segmentation, targets_segmentation) = segmentation
    (inputs_position, targets_position) = position
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

  def test_pack_sequences_k_multi_input(self):
    input_tokens = tf.convert_to_tensor([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
    ], dtype=tf.int32)
    input_vectors = tf.convert_to_tensor([
        [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        [[3, 4, 5], [4, 5, 6], [0, 0, 0]],
        [[5, 6, 7], [0, 0, 0], [0, 0, 0]],
    ], dtype=tf.float32)
    targets = tf.convert_to_tensor([
        [10, 0, 0],
        [20, 30, 40],
        [50, 60, 0],
    ], dtype=tf.int32)
    (packed, segmentation, position) = pack_sequences_ops.pack_sequences_k(
        [input_tokens, input_vectors, targets],
        [5, 3, 5])
    (input_tokens_packed, input_vectors_packed, targets_packed) = packed
    (input_tokens_segmentation, input_vectors_segmentation,
     targets_segmentation) = segmentation
    (input_tokens_position, input_vectors_position, targets_position) = position
    self.assertAllEqual(
        input_tokens_packed, [
            [1, 2, 3, 0, 0],
            [4, 5, 6, 0, 0],
        ])
    self.assertAllEqual(
        input_vectors_packed, [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
        ])
    self.assertAllEqual(
        input_tokens_segmentation, [
            [1, 1, 1, 0, 0],
            [1, 1, 2, 0, 0],
        ])
    self.assertAllEqual(
        input_vectors_segmentation, [
            [1, 1, 1],
            [1, 1, 2],
        ])
    self.assertAllEqual(
        input_tokens_position, [
            [0, 1, 2, 0, 0],
            [0, 1, 0, 0, 0],
        ])
    self.assertAllEqual(
        input_vectors_position, [
            [0, 1, 2],
            [0, 1, 0],
        ])
    self.assertAllEqual(
        targets_packed, [
            [10, 0, 0, 0, 0],
            [20, 30, 40, 50, 60],
        ])
    self.assertAllEqual(
        targets_segmentation, [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 2, 2],
        ])
    self.assertAllEqual(
        targets_position, [
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 1],
        ])

  def test_pack_sequences_k_int64(self):
    inputs = tf.convert_to_tensor([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
    ], dtype=tf.int64)
    max_length = tf.convert_to_tensor(5, dtype=tf.int32)
    (packed, segmentation, position) = pack_sequences_ops.pack_sequences_k(
        [inputs], [max_length])
    (inputs_packed,) = packed
    (inputs_segmentation,) = segmentation
    (inputs_position,) = position
    self.assertAllEqual(
        inputs_packed, [
            [1, 2, 3, 4, 5],
            [6, 0, 0, 0, 0],
        ])
    self.assertEqual(inputs_packed.dtype, tf.int64)
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

  def test_pack_sequences_k_bfloat16(self):
    inputs = tf.convert_to_tensor([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
    ], dtype=tf.bfloat16)
    max_length = tf.convert_to_tensor(5, dtype=tf.int32)
    (packed, segmentation, position) = pack_sequences_ops.pack_sequences_k(
        [inputs], [max_length])
    (inputs_packed,) = packed
    (inputs_segmentation,) = segmentation
    (inputs_position,) = position
    self.assertAllEqual(
        inputs_packed, [
            [1, 2, 3, 4, 5],
            [6, 0, 0, 0, 0],
        ])
    self.assertEqual(inputs_packed.dtype, tf.bfloat16)
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


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
