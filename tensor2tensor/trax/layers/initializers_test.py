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

"""Tests for initializers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tensor2tensor.trax.backend import random
from tensor2tensor.trax.layers import initializers


class InitializersTest(absltest.TestCase):

  def test_random_normal(self):
    initializer = initializers.RandomNormalInitializer()
    input_shape = (29, 5, 7, 20)
    init_value = initializer(input_shape, random.get_prng(0))
    self.assertEqual(tuple(init_value.shape), input_shape)

  def test_random_uniform(self):
    initializer = initializers.RandomUniformInitializer()
    input_shape = (29, 5, 7, 20)
    init_value = initializer(input_shape, random.get_prng(0))
    self.assertEqual(tuple(init_value.shape), input_shape)

  def test_glorot_normal(self):
    initializer = initializers.GlorotNormalInitializer()
    input_shape = (29, 5, 7, 20)
    init_value = initializer(input_shape, random.get_prng(0))
    self.assertEqual(tuple(init_value.shape), input_shape)

  def test_glorot_uniform(self):
    initializer = initializers.GlorotUniformInitializer()
    input_shape = (29, 5, 7, 20)
    init_value = initializer(input_shape, random.get_prng(0))
    self.assertEqual(tuple(init_value.shape), input_shape)

  def test_lecun_normal(self):
    initializer = initializers.LeCunNormalInitializer()
    input_shape = (29, 5, 7, 20)
    init_value = initializer(input_shape, random.get_prng(0))
    self.assertEqual(tuple(init_value.shape), input_shape)

  def test_lecun_uniform(self):
    initializer = initializers.LeCunUniformInitializer()
    input_shape = (29, 5, 7, 20)
    init_value = initializer(input_shape, random.get_prng(0))
    self.assertEqual(tuple(init_value.shape), input_shape)

  def test_kaiming_normal(self):
    initializer = initializers.KaimingNormalInitializer()
    input_shape = (29, 5, 7, 20)
    init_value = initializer(input_shape, random.get_prng(0))
    self.assertEqual(tuple(init_value.shape), input_shape)

  def test_kaiming_uniform(self):
    initializer = initializers.KaimingUniformInitializer()
    input_shape = (29, 5, 7, 20)
    init_value = initializer(input_shape, random.get_prng(0))
    self.assertEqual(tuple(init_value.shape), input_shape)


if __name__ == "__main__":
  absltest.main()
