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

"""Tests for tensor2tensor.envs.env_service_servicer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import mock
import numpy as np
from tensor2tensor.envs import env_service_pb2
from tensor2tensor.envs import env_service_serialization
from tensor2tensor.envs import env_service_servicer
from tensorflow import test


class EnvServiceServicerTest(test.TestCase):

  def test_get_env_info(self):
    env = gym.make("CartPole-v0")
    env_ss = env_service_servicer.EnvServiceServicer(env)
    env_info = env_ss.GetEnvInfo(None, None)

    self.assertIsInstance(env_info, env_service_pb2.EnvInfoResponse)

    self.assertTrue(env_info.observation_space.HasField("box"))
    self.assertTrue(env_info.action_space.HasField("discrete"))

    self.assertEqual(1, len(env_info.observation_space.box.shape.dim))
    self.assertEqual(4, env_info.observation_space.box.shape.dim[0].size)
    self.assertEqual(2, env_info.action_space.discrete.num_actions)

    self.assertEqual(-np.inf, env_info.reward_range.low)
    self.assertEqual(np.inf, env_info.reward_range.high)

    self.assertEqual(1, env_info.batch_size)

  def test_reset(self):
    # Set expectation on a mock.
    reset_obs = np.array([0.1, 0.2, 0.3, 0.4])
    env = mock.Mock()
    env.reset.return_value = reset_obs

    # Call reset.
    env_ss = env_service_servicer.EnvServiceServicer(env)
    reset_response = env_ss.Reset(None, None)

    # Assert the set expectation.
    self.assertIsInstance(reset_response, env_service_pb2.ResetResponse)
    self.assertAllEqual(
        reset_obs,
        env_service_serialization.tensor_proto_to_numpy_array(
            reset_response.observation.observation))

  def test_step(self):
    action = 3
    step_obs = np.array([1.1, 1.2, 1.3, 1.4])
    reward = 1.2
    done = False
    info = {"k1": 1, "k2": 2}

    env = mock.Mock()
    env.step.return_value = (step_obs, reward, done, [info])

    env_ss = env_service_servicer.EnvServiceServicer(env)
    step_request = env_service_pb2.StepRequest(
        action=env_service_pb2.Action(discrete_action=action))
    step_response = env_ss.Step(step_request, None)

    self.assertAllEqual(
        step_obs,
        env_service_serialization.tensor_proto_to_numpy_array(
            step_response.observation.observation))
    self.assertEqual(reward, step_response.reward)
    self.assertEqual(done, step_response.done)
    self.assertEqual(info["k1"], step_response.info.info_map["k1"])
    self.assertEqual(info["k2"], step_response.info.info_map["k2"])


if __name__ == "__main__":
  test.main()
