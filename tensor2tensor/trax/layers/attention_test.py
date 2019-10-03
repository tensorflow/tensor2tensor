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

"""Tests for tensor2tensor.trax.layers.attention."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
from tensor2tensor.trax.layers import attention
from tensor2tensor.trax.layers import base
from tensorflow import test


class AttentionTest(test.TestCase):

  def test_shift_right(self):
    # Test shifts right on axis=1
    layer = attention.ShiftRight()
    input_np = onp.arange(2*3*3).reshape(2, 3, 3)
    output_np = layer(input_np)
    self.assertEqual(input_np.shape, output_np.shape)
    self.assertAllEqual(onp.array([[[0, 0, 0],
                                    [0, 1, 2],
                                    [3, 4, 5]],

                                   [[0, 0, 0],
                                    [9, 10, 11],
                                    [12, 13, 14]]]),
                        output_np)

  def test_shift_right_float(self):
    layer = attention.ShiftRight()
    input_np = onp.arange(2*3*3).reshape(2, 3, 3).astype(onp.float32)
    # Test on a float array.
    input_np = input_np.astype(onp.float32)
    input_np /= 2.0
    self.assertEqual(input_np.dtype, onp.float32)

    output_np = layer(input_np)
    self.assertEqual(input_np.shape, output_np.shape)
    self.assertEqual(output_np.dtype, onp.float32)

    self.assertAllEqual(onp.array([[[0., 0., 0.],
                                    [0., 0.5, 1.],
                                    [1.5, 2., 2.5]],

                                   [[0., 0., 0.],
                                    [4.5, 5., 5.5],
                                    [6., 6.5, 7.]]]),
                        output_np)

  def test_merged_hashed_causal_attention(self):
    qkv_shape = (3, 32, 8)
    input_shape = (qkv_shape, qkv_shape, qkv_shape)
    layer = attention.MemoryEfficientCausalAttention(
        loop_stride=16, dropout=0.1, mode='train')
    final_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual((3, 32, 8), final_shape)

  def test_time_bin_causal_attention_bin_length(self):
    qkv_shape = (3, 57, 8)
    input_shape = (qkv_shape, qkv_shape, qkv_shape)
    layer = attention.TimeBinCausalAttention(
        bin_length=16, dropout=0.1, mode='train')
    final_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual((3, 57, 8), final_shape)

  def test_time_bin_causal_attention_n_bins(self):
    qkv_shape = (3, 57, 8)
    input_shape = (qkv_shape, qkv_shape, qkv_shape)
    layer = attention.TimeBinCausalAttention(
        n_bins=4, dropout=0.1, mode='train')
    final_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual((3, 57, 8), final_shape)

  def test_time_bin_and_dot_product_causal_attention_are_consistent(self):
    dot_product_layer = attention.DotProductCausalAttention(
        dropout=0.0, mode='train')
    time_bin_layer = attention.TimeBinCausalAttention(
        bin_length=4, dropout=0.0, mode='train')

    # Exactly 2 bins.
    input_shape = (3, 8, 8)
    inputs = [onp.random.uniform(size=input_shape) for _ in range(3)]

    dot_product_output = dot_product_layer(inputs)
    time_bin_output = time_bin_layer(inputs)
    onp.testing.assert_array_almost_equal(dot_product_output, time_bin_output)


if __name__ == '__main__':
  test.main()
