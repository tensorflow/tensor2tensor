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

"""Tests for normalization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as onp

from tensor2tensor.trax import backend
from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base
from tensor2tensor.trax.layers import normalization


class NormalizationLayerTest(absltest.TestCase):

  def test_batch_norm_shape(self):
    input_shape = (29, 5, 7, 20)
    result_shape = base.check_shape_agreement(
        normalization.BatchNorm(), input_shape)
    self.assertEqual(result_shape, input_shape)

  def test_batch_norm(self):
    input_shape = (2, 3, 4)
    input_dtype = np.float32
    eps = 1e-5
    rng = backend.random.get_prng(0)
    inp1 = np.reshape(np.arange(np.prod(input_shape), dtype=input_dtype),
                      input_shape)
    m1 = 11.5  # Mean of this random input.
    v1 = 47.9167  # Variance of this random input.
    layer = normalization.BatchNorm(axis=(0, 1, 2))
    params, state = layer.initialize(input_shape, input_dtype, rng)
    onp.testing.assert_allclose(state[0], 0)
    onp.testing.assert_allclose(state[1], 1)
    self.assertEqual(state[2], 0)
    out, state = layer(inp1, params, state)
    onp.testing.assert_allclose(state[0], m1 * 0.001)
    onp.testing.assert_allclose(state[1], 0.999 + v1 * 0.001, rtol=1e-6)
    self.assertEqual(state[2], 1)
    onp.testing.assert_allclose(out, (inp1 - m1) / np.sqrt(v1 + eps),
                                rtol=1e-6)

  def test_layer_norm_shape(self):
    input_shape = (29, 5, 7, 20)
    result_shape = base.check_shape_agreement(
        normalization.LayerNorm(), input_shape)
    self.assertEqual(result_shape, input_shape)


if __name__ == "__main__":
  absltest.main()
