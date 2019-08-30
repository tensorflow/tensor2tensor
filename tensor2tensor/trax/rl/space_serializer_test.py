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

"""Tests for tensor2tensor.trax.rl.space_serializer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import gym
import numpy as np

from tensor2tensor.trax.rl import space_serializer
from tensorflow import test


class BoxSpaceSerializerTest(test.TestCase):

  def _make_space_and_serializer(self, low=-10, high=10, shape=(2,)):
    # Enough precision to represent float32s accurately.
    gin.bind_parameter("BoxSpaceSerializer.precision", 4)
    space = gym.spaces.Box(low=low, high=high, shape=shape)
    serializer = space_serializer.create(
        space,
        # Weird vocab_size to test that it doesn't only work with powers of 2.
        vocab_size=257)
    return (space, serializer)

  def _sample_batch(self, space):
    return np.reshape(space.sample(), (1,) + space.shape)

  def test_representation_length(self):
    (space, serializer) = self._make_space_and_serializer()
    input_array = self._sample_batch(space)
    representation = serializer.serialize(input_array)
    self.assertEqual(
        representation.shape, (1, serializer.representation_length))

  def test_commutes(self):
    (space, serializer) = self._make_space_and_serializer()
    input_array = self._sample_batch(space)
    representation = serializer.serialize(input_array)
    output_array = serializer.deserialize(representation)
    np.testing.assert_array_almost_equal(input_array, output_array)

  def test_representation_changes(self):
    (space, serializer) = self._make_space_and_serializer()
    array1 = self._sample_batch(space)
    array2 = -array1
    (repr1, repr2) = tuple(map(serializer.serialize, (array1, array2)))
    self.assertFalse(np.array_equal(repr1, repr2))

  def test_bounds_space(self):
    gin.bind_parameter("BoxSpaceSerializer.max_range", (-10.0, 10.0))
    (_, serializer) = self._make_space_and_serializer(
        # Too wide range to represent, need to clip.
        low=-1e18, high=1e18,
        shape=(1,))
    input_array = np.array([[1.2345]])
    representation = serializer.serialize(input_array)
    output_array = serializer.deserialize(representation)
    np.testing.assert_array_almost_equal(input_array, output_array)


class DiscreteSpaceSerializerTest(test.TestCase):

  def setUp(self):
    super(DiscreteSpaceSerializerTest, self).setUp()
    self._space = gym.spaces.Discrete(n=2)
    self._serializer = space_serializer.create(self._space, vocab_size=2)

  def _sample_batch(self):
    return np.reshape(self._space.sample(), (1,) + self._space.shape)

  def test_representation_length(self):
    input_array = self._sample_batch()
    representation = self._serializer.serialize(input_array)
    self.assertEqual(
        representation.shape, (1, self._serializer.representation_length))

  def test_commutes(self):
    input_array = self._sample_batch()
    representation = self._serializer.serialize(input_array)
    output_array = self._serializer.deserialize(representation)
    np.testing.assert_array_almost_equal(input_array, output_array)

  def test_representation_changes(self):
    array1 = self._sample_batch()
    array2 = 1 - array1
    (repr1, repr2) = tuple(map(self._serializer.serialize, (array1, array2)))
    self.assertFalse(np.array_equal(repr1, repr2))


if __name__ == "__main__":
  test.main()
