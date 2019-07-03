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


class CombinatorLayerTest(absltest.TestCase):

  def test_drop(self):
    layer = cb.Drop()
    input_shape = (3, 2)
    expected_shape = ()
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_dup(self):
    layer = cb.Dup()
    input_shape = (3, 2)
    expected_shape = ((3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_swap(self):
    layer = cb.Swap()
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((4, 7), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_no_op(self):
    layer = cb.Serial(None)
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_no_op_list(self):
    layer = cb.Serial([])
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_one_in_one_out(self):
    layer = cb.Serial(core.Div(divisor=2.0))
    input_shape = (3, 2)
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_div_div(self):
    layer = cb.Serial(core.Div(divisor=2.0), core.Div(divisor=5.0))
    input_shape = (3, 2)
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_serial_dup_dup(self):
    layer = cb.Serial(cb.Dup(), cb.Dup())
    input_shape = (3, 2)
    expected_shape = ((3, 2), (3, 2), (3, 2))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_dup_dup(self):
    layer = cb.Parallel(cb.Dup(), cb.Dup())
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((3, 2), (3, 2), (4, 7), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_div_div(self):
    layer = cb.Parallel(core.Div(divisor=0.5), core.Div(divisor=3.0))
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_no_ops(self):
    layer = cb.Parallel([], None)
    input_shape = ((3, 2), (4, 7))
    expected_shape = ((3, 2), (4, 7))
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_branch_op_not_defined(self):
    with self.assertRaises(AttributeError):
      cb.Branch([], [])

  def test_select_op_not_defined(self):
    input_shape = ((3, 2), (4, 7))
    with self.assertRaises(AttributeError):
      cb.Select(1, input_shape)

if __name__ == '__main__':
  absltest.main()
