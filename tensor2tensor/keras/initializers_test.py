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

"""Tests for Keras-style initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from tensor2tensor.keras import initializers
from tensor2tensor.utils import test_utils

import tensorflow as tf
tf.compat.v1.enable_eager_execution()


class InitializersTest(tf.test.TestCase):

  @test_utils.run_in_graph_and_eager_modes
  def testTrainableHalfCauchy(self):
    shape = (3,)
    initializer = initializers.get('trainable_half_cauchy')
    half_cauchy = initializer(shape)
    self.evaluate(tf.global_variables_initializer())
    loc_value, scale_value = self.evaluate([
        # Get distribution of rv -> get distribution of Independent.
        half_cauchy.distribution.distribution.loc,
        half_cauchy.distribution.distribution.scale])
    self.assertAllClose(loc_value, np.zeros(shape), atol=1e-4)
    self.assertAllClose(scale_value, np.ones(shape), atol=1e-4)

    half_cauchy_value = self.evaluate(half_cauchy)
    self.assertAllEqual(half_cauchy_value.shape, shape)
    self.assertAllGreaterEqual(half_cauchy_value, 0.)

  @test_utils.run_in_graph_and_eager_modes
  def testTrainableNormal(self):
    shape = (100,)
    # TrainableNormal is expected to have var 1/shape[0]
    # because it by default has the fan_in mode scale normal std initializer.
    initializer = initializers.get('trainable_normal')
    normal = initializer(shape)
    self.evaluate(tf.global_variables_initializer())
    loc_value, scale_value = self.evaluate([
        # Get distribution of rv -> get distribution of Independent.
        normal.distribution.distribution.loc,
        normal.distribution.distribution.scale])
    fan_in = shape[0]
    target_scale = 1.
    target_scale /= max(1., fan_in)
    target_scale = math.sqrt(target_scale)

    self.assertAllClose(loc_value, np.zeros(shape), atol=1e-4)
    # Tolerance is larger because of the scale normal std initializer.
    # In this case it has std around 0.01 (0.1*target_scale).
    self.assertAllClose(
        scale_value, target_scale * np.ones(shape), atol=5e-2)

    # Test the TrainableNormal initializer has the specified shape.
    normal_value = self.evaluate(normal)
    self.assertAllEqual(normal_value.shape, shape)


if __name__ == '__main__':
  tf.test.main()
