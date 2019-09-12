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

"""Tests for core layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as onp
from tensor2tensor.trax import backend
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import combinators
from tensor2tensor.trax.layers import core


class CoreLayerTest(absltest.TestCase):

  def test_flatten_n(self):
    input_shape = (29, 87, 10, 20, 30)

    layer = core.Flatten()
    expected_shape = (29, 87 * 10 * 20 * 30)
    actual_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(actual_shape, expected_shape)

    layer = core.Flatten(n_axes_to_keep=2)
    expected_shape = (29, 87, 10 * 20 * 30)
    actual_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(actual_shape, expected_shape)

    layer = core.Flatten(n_axes_to_keep=3)
    expected_shape = (29, 87, 10, 20 * 30)
    actual_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(actual_shape, expected_shape)

    layer = core.Flatten(n_axes_to_keep=4)
    expected_shape = (29, 87, 10, 20, 30)
    actual_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(actual_shape, expected_shape)

    # Not enough dimensions.
    with self.assertRaises(base.LayerError):
      base.check_shape_agreement(core.Flatten(n_axes_to_keep=5), input_shape)

    with self.assertRaises(base.LayerError):
      base.check_shape_agreement(core.Flatten(n_axes_to_keep=6), input_shape)

  def test_div(self):
    layer = core.Div(divisor=2.0)
    input_np = onp.array([[1, 2, 3], [4, 5, 6]], dtype=onp.float32)
    output_np, _ = layer(input_np)
    # absltest doesn't have ndarray equalities.
    expected_output_np = input_np / 2.0
    self.assertAlmostEqual(
        0.0,
        onp.sum((output_np - expected_output_np) ** 2),
        delta=1e-6)

  def test_div_shapes(self):
    layer = core.Div(divisor=2.0)
    input_shape = (3, 2)
    expected_shape = (3, 2)
    output_shape = base.check_shape_agreement(layer, input_shape)
    self.assertEqual(output_shape, expected_shape)

  def test_dense_param_sharing(self):
    model1 = combinators.Serial(core.Dense(32), core.Dense(32))
    layer = core.Dense(32)
    model2 = combinators.Serial(layer, layer)

    rng1, rng2 = backend.random.split(backend.random.get_prng(0), 2)
    params1, _ = model1.initialize((1, 32), onp.float32, rng1)
    params2, _ = model2.initialize((1, 32), onp.float32, rng2)
    # The first parameters have 2 kernels of size (32, 32).
    self.assertEqual((32, 32), params1[0][0].shape)
    self.assertEqual((32, 32), params1[1][0].shape)
    # The second parameters have 1 kernel of size (32, 32) and an empty dict.
    self.assertEqual((32, 32), params2[0][0].shape)
    self.assertEqual((), params2[1])

  def test_dropout(self):
    input_shape = (8, 7, 9)
    output_shape = (8, 7, 9)
    final_shape = base.check_shape_agreement(
        core.Dropout(rate=0.1, mode="train"), input_shape)
    self.assertEqual(final_shape, output_shape)
    final_shape = base.check_shape_agreement(
        core.Dropout(rate=0.1, mode="eval"), input_shape)
    self.assertEqual(final_shape, output_shape)


if __name__ == "__main__":
  absltest.main()
