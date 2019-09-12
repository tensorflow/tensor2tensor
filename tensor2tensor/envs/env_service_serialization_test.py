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

"""Tests for tensor2tensor.envs.env_service_serialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from tensor2tensor.envs import env_service_serialization as utils

from tensorflow import test
from tensorflow.core.framework import types_pb2  # pylint: disable=g-direct-tensorflow-import


class UtilsTest(test.TestCase):

  def test_conversion(self):
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    obs = utils.numpy_array_to_observation(np_a)

    tp_a = obs.observation
    np_tp_a = utils.tensor_proto_to_numpy_array(tp_a)

    np.testing.assert_array_equal(np_a, np_tp_a)

  def test_step_request_from_discrete_action(self):
    discrete_action = 6
    step_request = utils.step_request_from_discrete_action(discrete_action)
    action_request = step_request.action
    self.assertTrue(action_request.HasField("discrete_action"))
    self.assertEqual("discrete_action", action_request.WhichOneof("payload"))
    self.assertEqual(discrete_action, action_request.discrete_action)

  def test_gym_space_to_proto_discrete(self):
    num_actions = 77
    space = gym.spaces.Discrete(num_actions)
    space_proto = utils.gym_space_to_proto(space)

    self.assertFalse(space_proto.HasField("box"))
    self.assertTrue(space_proto.HasField("discrete"))
    self.assertEqual(num_actions, space_proto.discrete.num_actions)

  def test_gym_space_to_proto_box(self):
    space = gym.spaces.Box(low=0, high=255, shape=(28, 29, 3), dtype=np.uint8)
    space_proto = utils.gym_space_to_proto(space)

    self.assertTrue(space_proto.HasField("box"))
    self.assertEqual(types_pb2.DT_UINT8, space_proto.box.dtype)

    self.assertEqual(28, space_proto.box.shape.dim[0].size)
    self.assertEqual(29, space_proto.box.shape.dim[1].size)
    self.assertEqual(3, space_proto.box.shape.dim[2].size)

  def test_proto_to_gym_space_discrete(self):
    num_actions = 77
    space = gym.spaces.Discrete(num_actions)
    space_proto = utils.gym_space_to_proto(space)
    space_gym = utils.proto_to_gym_space(space_proto)
    space_gym.n = num_actions

  def test_proto_to_gym_space_box(self):
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(28, 29), dtype=np.float32)
    space_proto = utils.gym_space_to_proto(space)
    space_gym = utils.proto_to_gym_space(space_proto)
    self.assertEqual(np.float32, space_gym.dtype)
    self.assertAllEqual(space.shape, space_gym.shape)

  def test_reward_range_to_proto(self):
    reward_proto = utils.reward_range_to_proto((-12, +13))
    self.assertEqual(-12, reward_proto.low)
    self.assertEqual(+13, reward_proto.high)

    reward_proto = utils.reward_range_to_proto()
    self.assertEqual(-np.inf, reward_proto.low)
    self.assertEqual(np.inf, reward_proto.high)


if __name__ == "__main__":
  test.main()
