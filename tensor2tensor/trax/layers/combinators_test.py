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
from tensor2tensor.trax.layers import combinators
from tensor2tensor.trax.layers import convolution


class CombinatorLayerTest(absltest.TestCase):

  def test_branch(self):
    input_shape = (2, 3)
    expected_shape = ((2, 3), (2, 3))
    output_shape = base.check_shape_agreement(
        combinators.Branch([], []), input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_branch_named(self):
    input_shape = (2, 3)
    expected_shape = {'a': (2, 3), 'b': (2, 3)}
    output_shape = base.check_shape_agreement(
        combinators.Branch(a=[], b=[]), input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel(self):
    input_shape = ((2, 3), (2, 3))
    expected_shape = ((2, 3), (2, 3))
    output_shape = base.check_shape_agreement(
        combinators.Parallel([], []), input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_parallel_named(self):
    input_shape = {'a': (2, 3), 'b': (2, 3)}
    expected_shape = {'a': (2, 3), 'b': (2, 3)}
    output_shape = base.check_shape_agreement(
        combinators.Parallel(a=[]), input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_select(self):
    input_shape = ((2, 3), (3, 4))
    expected_shape = (3, 4)
    output_shape = base.check_shape_agreement(
        combinators.Select(1), input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_select_named(self):
    input_shape = {'a': (2, 3), 'b': (3, 4)}
    expected_shape = (3, 4)
    output_shape = base.check_shape_agreement(
        combinators.Select('b'), input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_rebatch(self):
    input_shape = (29, 5, 5, 20)
    result_shape = base.check_shape_agreement(
        convolution.Conv(30, (3, 3)), input_shape)
    self.assertEqual(result_shape, (29, 3, 3, 30))

    input_shape = (29, 5, 5, 20)
    result_shape = base.check_shape_agreement(
        combinators.Rebatch(convolution.Conv(30, (3, 3)), n_batch_dims=1),
        input_shape)
    self.assertEqual(result_shape, (29, 3, 3, 30))

    input_shape = (19, 29, 5, 5, 20)
    result_shape = base.check_shape_agreement(
        combinators.Rebatch(convolution.Conv(30, (3, 3)), n_batch_dims=2),
        input_shape)
    self.assertEqual(result_shape, (19, 29, 3, 3, 30))

if __name__ == '__main__':
  absltest.main()
