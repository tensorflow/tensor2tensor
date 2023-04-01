# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

import numpy as np
from tensor2tensor.data_generators.ops import pack_sequences_ops
import tensorflow.compat.v1 as tf


def _pack_sequences_k(inputs, targets, input_max_length, target_max_length):
  """Wrapper for pack_sequences_k with same interface as pack_sequences_2."""
  inputs = tf.convert_to_tensor(inputs, tf.int32)
  targets = tf.convert_to_tensor(targets, tf.int32)
  input_max_length = tf.convert_to_tensor(input_max_length, dtype=tf.int32)
  target_max_length = tf.convert_to_tensor(target_max_length, dtype=tf.int32)
  (packed, segmentation, position) = pack_sequences_ops.pack_sequences_k(
      [inputs, targets], [input_max_length, target_max_length])
  (inputs_packed, targets_packed) = packed
  (inputs_segmentation, targets_segmentation) = segmentation
  (inputs_position, targets_position) = position
  return (inputs_packed, inputs_segmentation, inputs_position, targets_packed,
          targets_segmentation, targets_position)


class PackSequencesOpsTest(tf.test.TestCase):

  def do_test_pack_sequences_length3(self, pack_fn):
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
    inputs_max_length = 3
    targets_max_length = 3
    (inputs_packed, inputs_segmentation, inputs_position, targets_packed,
     targets_segmentation, targets_position) = (
         pack_fn(inputs, targets, inputs_max_length, targets_max_length))
    self.assertAllEqual(inputs_packed, [
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
    ])
    self.assertAllEqual(inputs_segmentation, [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
    ])
    self.assertAllEqual(inputs_position, [
        [0, 1, 2],
        [0, 1, 0],
        [0, 0, 0],
    ])
    self.assertAllEqual(targets_packed, [
        [10, 0, 0],
        [20, 30, 40],
        [50, 60, 0],
    ])
    self.assertAllEqual(targets_segmentation, [
        [1, 0, 0],
        [1, 1, 1],
        [1, 1, 0],
    ])
    self.assertAllEqual(targets_position, [
        [0, 0, 0],
        [0, 1, 2],
        [0, 1, 0],
    ])

  def do_test_pack_sequences_length4(self, pack_fn):
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
    inputs_max_length = 4
    targets_max_length = 4
    (inputs_packed, inputs_segmentation, inputs_position, targets_packed,
     targets_segmentation, targets_position) = (
         pack_fn(inputs, targets, inputs_max_length, targets_max_length))
    self.assertAllEqual(inputs_packed, [
        [1, 2, 3, 6],
        [4, 5, 0, 0],
    ])
    self.assertAllEqual(inputs_segmentation, [
        [1, 1, 1, 2],
        [1, 1, 0, 0],
    ])
    self.assertAllEqual(inputs_position, [
        [0, 1, 2, 0],
        [0, 1, 0, 0],
    ])
    self.assertAllEqual(targets_packed, [
        [10, 50, 60, 0],
        [20, 30, 40, 0],
    ])
    self.assertAllEqual(targets_segmentation, [
        [1, 2, 2, 0],
        [1, 1, 1, 0],
    ])
    self.assertAllEqual(targets_position, [
        [0, 0, 1, 0],
        [0, 1, 2, 0],
    ])

  def do_test_pack_sequences_length5(self, pack_fn):
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
    (inputs_packed, inputs_segmentation, inputs_position, targets_packed,
     targets_segmentation, targets_position) = (
         pack_fn(inputs, targets, max_length, max_length))
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

  def do_test_pack_sequences_length6(self, pack_fn):
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
    max_length = 6
    (inputs_packed, inputs_segmentation, inputs_position, targets_packed,
     targets_segmentation, targets_position) = (
         pack_fn(inputs, targets, max_length, max_length))
    self.assertAllEqual(inputs_packed, [
        [1, 2, 3, 4, 5, 6],
    ])
    self.assertAllEqual(inputs_segmentation, [
        [1, 1, 1, 2, 2, 3],
    ])
    self.assertAllEqual(inputs_position, [
        [0, 1, 2, 0, 1, 0],
    ])
    self.assertAllEqual(targets_packed, [
        [10, 20, 30, 40, 50, 60],
    ])
    self.assertAllEqual(targets_segmentation, [
        [1, 2, 2, 2, 3, 3],
    ])
    self.assertAllEqual(targets_position, [
        [0, 0, 1, 2, 0, 1],
    ])

  def do_test_pack_sequences_length7(self, pack_fn):
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
    max_length = 7
    (inputs_packed, inputs_segmentation, inputs_position, targets_packed,
     targets_segmentation, targets_position) = (
         pack_fn(inputs, targets, max_length, max_length))
    self.assertAllEqual(inputs_packed, [
        [1, 2, 3, 4, 5, 6, 0],
    ])
    self.assertAllEqual(inputs_segmentation, [
        [1, 1, 1, 2, 2, 3, 0],
    ])
    self.assertAllEqual(inputs_position, [
        [0, 1, 2, 0, 1, 0, 0],
    ])
    self.assertAllEqual(targets_packed, [
        [10, 20, 30, 40, 50, 60, 0],
    ])
    self.assertAllEqual(targets_segmentation, [
        [1, 2, 2, 2, 3, 3, 0],
    ])
    self.assertAllEqual(targets_position, [
        [0, 0, 1, 2, 0, 1, 0],
    ])

  def do_test_pack_sequences_length_different_lengths(self, pack_fn):
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
    input_max_length = 3
    target_max_length = 4
    (inputs_packed, inputs_segmentation, inputs_position, targets_packed,
     targets_segmentation, targets_position) = (
         pack_fn(inputs, targets, input_max_length, target_max_length))
    self.assertAllEqual(inputs_packed, [
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
    ])
    self.assertAllEqual(inputs_segmentation, [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
    ])
    self.assertAllEqual(inputs_position, [
        [0, 1, 2],
        [0, 1, 0],
        [0, 0, 0],
    ])
    self.assertAllEqual(targets_packed, [
        [10, 0, 0, 0],
        [20, 30, 40, 0],
        [50, 60, 0, 0],
    ])
    self.assertAllEqual(targets_segmentation, [
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ])
    self.assertAllEqual(targets_position, [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 1, 0, 0],
    ])

  def test_pack_sequences2(self):
    self.do_test_pack_sequences_length3(pack_sequences_ops.pack_sequences2)
    self.do_test_pack_sequences_length4(pack_sequences_ops.pack_sequences2)
    self.do_test_pack_sequences_length5(pack_sequences_ops.pack_sequences2)
    self.do_test_pack_sequences_length6(pack_sequences_ops.pack_sequences2)
    self.do_test_pack_sequences_length7(pack_sequences_ops.pack_sequences2)
    self.do_test_pack_sequences_length_different_lengths(
        pack_sequences_ops.pack_sequences2)

  def test_pack_sequences_k(self):
    self.do_test_pack_sequences_length3(_pack_sequences_k)
    self.do_test_pack_sequences_length4(_pack_sequences_k)
    self.do_test_pack_sequences_length5(_pack_sequences_k)
    self.do_test_pack_sequences_length6(_pack_sequences_k)
    self.do_test_pack_sequences_length7(_pack_sequences_k)
    self.do_test_pack_sequences_length_different_lengths(_pack_sequences_k)

  def test_random_inputs(self):
    for _ in range(10):
      batch_size = np.random.randint(900, 1100, size=[])
      input_seqlen = np.random.randint(1, 10, size=[])
      target_seqlen = np.random.randint(1, 10, size=[])
      inputs_list = []
      targets_list = []
      for _ in range(batch_size):
        input_num_pads = np.random.randint(0, input_seqlen, size=[])
        input_pads = np.full([input_num_pads], 0, dtype=np.int32)
        inputs = np.random.randint(1, 10, size=[input_seqlen - input_num_pads])
        inputs = np.concatenate([inputs, input_pads], axis=0)

        target_num_pads = np.random.randint(0, target_seqlen, size=[])
        target_pads = np.full([target_num_pads], 0, dtype=np.int32)
        targets = np.random.randint(
            1, 10, size=[target_seqlen - target_num_pads])
        targets = np.concatenate([targets, target_pads], axis=0)

        inputs_list.append(inputs)
        targets_list.append(targets)
      input_maxlen = np.random.randint(input_seqlen, input_seqlen + 10, size=[])
      target_maxlen = np.random.randint(
          target_seqlen, target_seqlen + 10, size=[])
      (inputs_packed2, inputs_segmentation2, inputs_positions2, targets_packed2,
       targets_segmentation2, targets_positions2) = (
           pack_sequences_ops.pack_sequences2(inputs_list, targets_list,
                                              input_maxlen, target_maxlen))
      (inputs_packed_k, inputs_segmentation_k, inputs_positions_k,
       targets_packed_k, targets_segmentation_k, targets_positions_k) = (
           _pack_sequences_k(inputs_list, targets_list, input_maxlen,
                             target_maxlen))

      self.assertAllEqual(inputs_packed2, inputs_packed_k)
      self.assertAllEqual(inputs_segmentation2, inputs_segmentation_k)
      self.assertAllEqual(inputs_positions2, inputs_positions_k)
      self.assertAllEqual(targets_packed2, targets_packed_k)
      self.assertAllEqual(targets_segmentation2, targets_segmentation_k)
      self.assertAllEqual(targets_positions2, targets_positions_k)

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
