# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Tests for gym_spaces_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from tensor2tensor.envs import gym_spaces_utils
import tensorflow.compat.v1 as tf


class GymSpacesUtilsTest(tf.test.TestCase):

  def test_discrete_space_spec(self):
    discrete_space = Discrete(100)
    spec = gym_spaces_utils.gym_space_spec(discrete_space)
    self.assertIsInstance(spec, tf.FixedLenFeature)
    self.assertEqual(spec.dtype, tf.int64)
    self.assertListEqual(list(spec.shape), [1])

  def test_box_space_spec(self):
    box_space = Box(low=0, high=10, shape=[5, 6], dtype=np.float32)
    spec = gym_spaces_utils.gym_space_spec(box_space)
    self.assertIsInstance(spec, tf.FixedLenFeature)
    self.assertEqual(spec.dtype, tf.float32)
    self.assertListEqual(list(spec.shape), [5, 6])

  def test_discrete_space_encode(self):
    discrete_space = Discrete(100)
    value = discrete_space.sample()
    encoded_value = gym_spaces_utils.gym_space_encode(discrete_space, value)
    self.assertListEqual([value], encoded_value)

  def test_box_space_encode(self):
    box_space = Box(low=0, high=10, shape=[2], dtype=np.int64)
    value = np.array([2, 3])
    encoded_value = gym_spaces_utils.gym_space_encode(box_space, value)
    self.assertListEqual([2, 3], encoded_value)


if __name__ == '__main__':
  tf.test.main()
