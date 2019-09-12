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

"""Tests for tensor2tensor.trax.models.atari_cnn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator as op
import numpy as onp
from tensor2tensor.trax.backend import random as jax_random
from tensor2tensor.trax.models import atari_cnn
from tensorflow import test


class AtariCnnTest(test.TestCase):

  def test_computes(self):
    rng_key = jax_random.get_prng(0)
    hidden_size = (4, 4)
    output_size = 6
    model = atari_cnn.AtariCnn(
        hidden_sizes=hidden_size, output_size=output_size)
    B, T, OBS = 2, 2, (28, 28, 3)  # pylint: disable=invalid-name
    rng_key, key = jax_random.split(rng_key)
    params, state = model.initialize((1, 1) + OBS, onp.float32, key)
    x = onp.arange(B * (T + 1) * functools.reduce(op.mul, OBS)).reshape(
        B, T + 1, *OBS)
    rng_key, key = jax_random.split(rng_key)
    y, _ = model(x, params, state=state, rng=key)
    self.assertEqual((B, T + 1, output_size), y.shape)


class FrameStackMLPTest(test.TestCase):

  def test_computes(self):
    rng_key = jax_random.get_prng(0)
    hidden_size = (4, 4)
    output_size = 6
    model = atari_cnn.FrameStackMLP(
        hidden_sizes=hidden_size, output_size=output_size)
    B, T, OBS = 2, 2, 3  # pylint: disable=invalid-name
    rng_key, key = jax_random.split(rng_key)
    params, state = model.initialize((1, 1, OBS), onp.float32, key)
    x = onp.arange(B * (T + 1) * OBS).reshape(
        B, T + 1, OBS)
    rng_key, key = jax_random.split(rng_key)
    y, _ = model(x, params, state=state, rng=key)
    self.assertEqual((B, T + 1, output_size), y.shape)


if __name__ == "__main__":
  test.main()
