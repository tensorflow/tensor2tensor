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

"""Tests for tensor2tensor.envs.client_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import mock
import numpy as np
from tensor2tensor.envs import client_env
from tensor2tensor.envs import env_service_pb2
from tensor2tensor.envs import env_service_serialization
from tensorflow import test


class ClientEnvTest(test.TestCase):

  def configure_env_info_on_mock(self, mock_obj):
    env_info_response = env_service_pb2.EnvInfoResponse()
    env_info_response.observation_space.box.CopyFrom(
        env_service_serialization.gym_space_to_proto(
            gym.spaces.Box(low=0, high=255, shape=(28, 28, 3))).box)
    env_info_response.action_space.discrete.num_actions = 6
    env_info_response.reward_range.low = -1
    env_info_response.reward_range.high = 1
    env_info_response.batch_size = 1
    mock_obj.GetEnvInfo.return_value = env_info_response

  def test_get_env_info(self):
    mock_stub = mock.Mock()
    self.configure_env_info_on_mock(mock_stub)

    env = client_env.ClientEnv(stub=mock_stub)

    self.assertIsInstance(env.action_space, gym.spaces.Discrete)
    self.assertIsInstance(env.observation_space, gym.spaces.Box)

    self.assertEqual(6, env.action_space.n)
    self.assertEqual((28, 28, 3), env.observation_space.shape)
    self.assertEqual((-1, 1), env.reward_range)

  def test_reset(self):
    mock_stub = mock.Mock()
    self.configure_env_info_on_mock(mock_stub)
    obs_np = np.random.uniform(size=(1, 28, 28, 3))
    reset_response = env_service_pb2.ResetResponse()
    reset_response.observation.CopyFrom(
        env_service_serialization.numpy_array_to_observation(obs_np))
    mock_stub.Reset.return_value = reset_response

    env = client_env.ClientEnv(stub=mock_stub)

    self.assertAllEqual(np.squeeze(obs_np, axis=0), env.reset())

  def test_step(self):
    mock_stub = mock.Mock()
    self.configure_env_info_on_mock(mock_stub)
    obs_np = np.random.uniform(size=(1, 28, 28, 3))
    reward = 0.5
    done = True
    step_response = env_service_pb2.StepResponse(reward=reward, done=done)
    step_response.observation.CopyFrom(
        env_service_serialization.numpy_array_to_observation(obs_np))
    step_response.info.info_map["k1"] = 1
    step_response.info.info_map["k2"] = 2
    mock_stub.Step.return_value = step_response

    action = 4
    step_request = env_service_pb2.StepRequest(
        action=env_service_pb2.Action(discrete_action=action))

    env = client_env.ClientEnv(stub=mock_stub)
    step_retval = env.step(action)

    mock_stub.Step.assert_called_with(step_request)
    self.assertAllEqual(np.squeeze(obs_np, axis=0), step_retval[0])
    self.assertEqual(reward, step_retval[1])
    self.assertEqual(done, step_retval[2])
    self.assertEqual(1, step_retval[3]["k1"])
    self.assertEqual(2, step_retval[3]["k2"])


if __name__ == "__main__":
  test.main()
