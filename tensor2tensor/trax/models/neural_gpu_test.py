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

"""Tests for google3.third_party.py.tensor2tensor.trax.models.neural_gpu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
from tensor2tensor.trax.backend import random as jax_random
from tensor2tensor.trax.models import neural_gpu
from google3.testing.pybase import googletest


class NeuralGPUTest(googletest.TestCase):

  def test_ngpu(self):
    vocab_size = 2
    in_shape = [3, 5, 7]
    source = np.ones(in_shape, dtype=np.int32)

    model = neural_gpu.NeuralGPU(
        feature_depth=30, steps=4, vocab_size=vocab_size)
    # Build params
    rng = jax_random.get_prng(0)
    logging.info(model)
    model.initialize(in_shape, rng)

    # Run network
    output = model(source)

    self.assertEqual(tuple(in_shape + [vocab_size]), output.shape)


if __name__ == '__main__':
  googletest.main()
