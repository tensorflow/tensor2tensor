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

"""Tests for google3.third_party.py.tensor2tensor.trax.layers.rnn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.trax.backend import random as jax_random
from tensor2tensor.trax.layers import rnn
import tensorflow as tf


class RnnModelTest(tf.test.TestCase):

  def _test_cell_runs(self, model, input_shape, output_shape):
    source = np.ones(input_shape, dtype=np.float32)

    # Build params
    rng = jax_random.get_prng(0)
    model.initialize(input_shape, rng)

    # Run network
    output = model(source)

    self.assertEqual(output_shape, output.shape)

  def test_conv_gru_cell(self):
    self._test_cell_runs(
        rnn.ConvGRUCell(units=9, kernel_size=(3, 3)),
        input_shape=(8, 1, 7, 9),
        output_shape=(8, 1, 7, 9))

  def test_gru_cell(self):
    self._test_cell_runs(
        rnn.GRUCell(units=9), input_shape=(8, 7, 9), output_shape=(8, 7, 9))


if __name__ == '__main__':
  tf.test.main()
