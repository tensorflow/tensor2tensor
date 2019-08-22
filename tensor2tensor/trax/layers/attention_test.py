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
from tensorflow import test


class AttentionTest(test.TestCase):

  def test_shift_right(self):
    # Test shifts right on axis=1
    layer = attention.ShiftRight()
    input_np = onp.arange(2*3*3).reshape(2, 3, 3)
    output_np, _ = layer(input_np)
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

    output_np, _ = layer(input_np)
    self.assertEqual(input_np.shape, output_np.shape)
    self.assertEqual(output_np.dtype, onp.float32)

    self.assertAllEqual(onp.array([[[0., 0., 0.],
                                    [0., 0.5, 1.],
                                    [1.5, 2., 2.5]],

                                   [[0., 0., 0.],
                                    [4.5, 5., 5.5],
                                    [6., 6.5, 7.]]]),
                        output_np)


if __name__ == '__main__':
  test.main()
