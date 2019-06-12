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

"""Tests for combinator layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import combinators as cb
from tensor2tensor.trax.layers import core


_EMPTY_STACK = ()
_REST_OF_STACK = ((1, 5), (4,))


class CombinatorLayerTest(absltest.TestCase):

  def test_drop(self):
    layer = cb.Drop()
    input_shape = ((3, 2),)
    expected_shape = _EMPTY_STACK
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

    input_shape = ((3, 2),) + _REST_OF_STACK
    expected_shape = _REST_OF_STACK
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_dup(self):
    layer = cb.Dup()
    input_shape = ((3, 2),)
    expected_shape = ((3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

    input_shape = ((3, 2),) + _REST_OF_STACK
    expected_shape = ((3, 2), (3, 2)) + _REST_OF_STACK
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_swap(self):
    layer = cb.Swap()
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((4, 7), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

    input_shape = ((3, 2), (4, 7)) + _REST_OF_STACK
    expected_shape = ((4, 7), (3, 2)) + _REST_OF_STACK
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_no_op_list(self):
    layer = cb.Serial([])
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

    input_shape = ((3, 2), (4, 7)) + _REST_OF_STACK
    expected_shape = ((3, 2), (4, 7)) + _REST_OF_STACK
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_one_in_one_out(self):
    layer = cb.Serial(core.Div(divisor=2.0))
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_div_div(self):
    layer = cb.Serial(core.Div(divisor=2.0), core.Div(divisor=5.0))
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_branch(self):
    input_shape = (2, 3)
    expected_shape = ((2, 3), (2, 3))
    output_shape = base.check_shape_agreement(cb.Branch([], []), input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel(self):
    input_shape = ((2, 3), (2, 3))
    expected_shape = ((2, 3), (2, 3))
    output_shape = base.check_shape_agreement(cb.Parallel([], []), input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_select(self):
    input_shape = ((2, 3), (3, 4))
    expected_shape = (3, 4)
    output_shape = base.check_shape_agreement(cb.Select(1), input_shape)
    self.assertEqual(output_shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
