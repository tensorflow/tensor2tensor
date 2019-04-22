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

"""Tests for Stax base layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as onp
from tensor2tensor.trax.backend import random
import tensor2tensor.trax.stax as stax


def random_inputs(rng, input_shape):
  if isinstance(input_shape, tuple):
    return rng.randn(*input_shape).astype(onp.float32)
  elif isinstance(input_shape, list):
    return [random_inputs(rng, shape) for shape in input_shape]
  else:
    raise TypeError(type(input_shape))


def check_shape_agreement(test_case, init_fun, apply_fun, input_shape):
  rng_key1, rng_key2 = random.split(random.get_prng(0))
  result_shape, params = init_fun(rng_key1, input_shape)
  inputs = random_inputs(onp.random.RandomState(0), input_shape)
  result = apply_fun(params, inputs, rng=rng_key2)
  test_case.assertEqual(result.shape, result_shape)
  return result_shape


def check_staxlayer(test_case, staxlayer, input_shape):
  init_fun, apply_fun = staxlayer
  return check_shape_agreement(test_case, init_fun, apply_fun, input_shape)


class SlaxTest(absltest.TestCase):

  def test_flatten_n(self):
    input_shape = (29, 87, 10, 20, 30)

    actual_shape = check_staxlayer(self, stax.Flatten(), input_shape)
    self.assertEqual(actual_shape, (29, 87 * 10 * 20 * 30))

    actual_shape = check_staxlayer(self, stax.Flatten(num_axis_to_keep=2),
                                   input_shape)
    self.assertEqual(actual_shape, (29, 87, 10 * 20 * 30))

    actual_shape = check_staxlayer(self, stax.Flatten(num_axis_to_keep=3),
                                   input_shape)
    self.assertEqual(actual_shape, (29, 87, 10, 20 * 30))

    actual_shape = check_staxlayer(self, stax.Flatten(num_axis_to_keep=4),
                                   input_shape)
    self.assertEqual(actual_shape, (29, 87, 10, 20, 30))

    # Not enough dimensions.
    with self.assertRaises(ValueError):
      check_staxlayer(self, stax.Flatten(num_axis_to_keep=5), input_shape)

    with self.assertRaises(ValueError):
      check_staxlayer(self, stax.Flatten(num_axis_to_keep=6), input_shape)

  def test_div(self):
    init_fun, apply_fun = stax.Div(divisor=2.0)
    input_np = onp.array([[1, 2, 3], [4, 5, 6]], dtype=onp.float32)
    input_shape = input_np.shape
    rng = random.get_prng(0)
    _, _ = init_fun(rng, input_shape)
    output_np = apply_fun(None, input_np)
    # absltest doesn't have ndarray equalities.
    expected_output_np = input_np / 2.0
    self.assertAlmostEqual(
        0.0,
        onp.sum((output_np - expected_output_np) ** 2),
        delta=1e-6)

  def test_dense_param_sharing(self):
    model1 = stax.Serial(stax.Dense(32), stax.Dense(32))
    layer = stax.Dense(32)
    model2 = stax.Serial(layer, layer)
    init_fun1, _ = model1
    init_fun2, _ = model2
    rng = random.get_prng(0)
    _, params1 = init_fun1(rng, [-1, 32])
    _, params2 = init_fun2(rng, [-1, 32])
    # The first parameters have 2 kernels of size (32, 32).
    self.assertEqual((32, 32), params1[0][0].shape)
    self.assertEqual((32, 32), params1[1][0].shape)
    # The second parameters have 1 kernel of size (32, 32) and an empty dict.
    self.assertEqual((32, 32), params2[0][0].shape)
    self.assertEqual((), params2[1])


if __name__ == "__main__":
  absltest.main()
