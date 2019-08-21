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
    m1 = 11.5
    v1 = 47.9167
    layer = normalization.BatchNorm(axis=(0, 1, 2))
    params, state = layer.initialize(input_shape, input_dtype, rng)
    onp.testing.assert_allclose(state[0], 0)
    onp.testing.assert_allclose(state[1], 0)
    self.assertEqual(state[2], 0)
    out, state = layer(inp1, params, state)
    onp.testing.assert_allclose(state[0], m1)
    onp.testing.assert_allclose(state[1], v1, rtol=1e-6)
    self.assertEqual(state[2], 1)
    onp.testing.assert_allclose(out, (inp1 - m1) / np.sqrt(v1 + eps),
                                rtol=1e-6)
    inp2 = inp1 * 2 + 3
    m2 = m1 * 2 + 3
    v2 = v1 * 4
    m12 = (m1 + m2) / 2
    v12 = (v1 + v2) / 2
    out, state = layer(inp2, params, state)
    onp.testing.assert_allclose(state[0], m12)
    onp.testing.assert_allclose(state[1], v12, rtol=1e-6)
    self.assertEqual(state[2], 2)
    onp.testing.assert_allclose(out, (inp2 - m2) / np.sqrt(v2 + eps),
                                rtol=1e-6)
    layer = normalization.BatchNorm(axis=(0, 1, 2), mode="eval")
    inp3 = inp1 * 5 + 7
    out, state_unchanged = layer(inp3, params, state)
    for i in range(3):
      onp.testing.assert_allclose(state_unchanged[i], state[i])
    onp.testing.assert_allclose(out, (inp3 - m12) / np.sqrt(v12 + eps),
                                rtol=1e-6)

  def test_layer_norm_shape(self):
    input_shape = (29, 5, 7, 20)
    result_shape = base.check_shape_agreement(
        normalization.LayerNorm(), input_shape)
    self.assertEqual(result_shape, input_shape)


if __name__ == "__main__":
  absltest.main()
