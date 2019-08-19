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

"""Client Env that connects to a distributed env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import grpc
import gym
import numpy as np
from tensor2tensor.envs import env_service_pb2
from tensor2tensor.envs import env_service_pb2_grpc
from tensor2tensor.envs import env_service_serialization as serialization


class ClientEnv(gym.Env):
  """Creates a connection to a remote env, and calls RPC methods on it."""

  @staticmethod
  def create_channel(remote_env_address):
    return grpc.insecure_channel(remote_env_address)  # pylint: disable=unreachable

  @staticmethod
  def run_step(stub, discrete_action):
    action_proto = env_service_pb2.Action(discrete_action=discrete_action)
    step_request = env_service_pb2.StepRequest()
    step_request.action.CopyFrom(action_proto)
    return stub.Step(step_request)

  @staticmethod
  def run_reset(stub):
    return stub.Reset(env_service_pb2.ResetRequest())

  @staticmethod
  def run_close(stub, channel):
    close_response = stub.Close(env_service_pb2.CloseRequest())
    channel.close()
    return close_response

  @staticmethod
  def run_render(stub, mode="rgb_array"):
    return stub.Render(env_service_pb2.RenderRequest(mode=mode))

  @staticmethod
  def run_get_env_info(stub):
    env_info_response = stub.GetEnvInfo(env_service_pb2.EnvInfoRequest())
    gym_observation_space = serialization.proto_to_gym_space(
        env_info_response.observation_space)
    gym_action_space = serialization.proto_to_gym_space(
        env_info_response.action_space)
    reward_range = (env_info_response.reward_range.low,
                    env_info_response.reward_range.high)
    return (gym_action_space, gym_observation_space, reward_range,
            env_info_response.batch_size)

  def __init__(self, remote_env_address=None, stub=None):
    self._channel = None
    self._stub = None
    self._remote_env_address = None

    if stub is not None:
      self._stub = stub
    else:
      assert remote_env_address is not None
      logging.vlog(1, "Making a ClientEnv with remote address: [%s]",
                   remote_env_address)
      self._remote_env_address = remote_env_address
      self.initialize_stub()

    assert self._stub is not None

    # We now have to do an RPC to determine spaces and reward range.
    #
    # NOTE: If all these are same across replicas, then we technically only need
    # to do this once on the 'master' replica (say 0), but `GymEnvProblem`
    # checks that they are all the same.
    (self.action_space, self.observation_space, self.reward_range,
     self._server_env_batch_size) = (
         ClientEnv.run_get_env_info(self._stub))

  def initialize_stub(self):
    self._channel = ClientEnv.create_channel(self._remote_env_address)
    # TODO(afrozm): Why is this done?
    grpc.channel_ready_future(self._channel).result()
    self._stub = env_service_pb2_grpc.EnvServiceStub(self._channel)

  def _maybe_squeeze_array(self, np_array):
    # Usually this client is talking to a server env that is running a single
    # element batch, if so, this client should strip out the batch dimension
    # before reporting the observation upstream (since this is a plain gym env,
    # not an EnvProblem), the upstream EnvProblem will then batch across
    # multiple ClientEnvs.
    if isinstance(
        np_array, np.ndarray
    ) and self._server_env_batch_size == 1 and np_array.shape[0] == 1:
      np_array = np.squeeze(np_array, axis=0)
    return np_array

  def reset(self):
    # Run the RPC.
    reset_response_proto = ClientEnv.run_reset(self._stub)
    # Convert the TensorProto to numpy.
    obs_np = serialization.tensor_proto_to_numpy_array(
        reset_response_proto.observation.observation)
    return self._maybe_squeeze_array(obs_np)

  def close(self):
    ClientEnv.run_close(self._stub, self._channel)

  def render(self, mode="rgb_array"):
    render_response = ClientEnv.run_render(self._stub, mode=mode)
    if not render_response:
      return
    # Parse out the numpy array.
    return serialization.tensor_proto_to_numpy_array(
        render_response.observation.observation)

  def step(self, action):
    step_response = ClientEnv.run_step(self._stub, action)
    observation = self._maybe_squeeze_array(
        serialization.tensor_proto_to_numpy_array(
            step_response.observation.observation))
    info = {k: v for k, v in step_response.info.info_map.items()}
    return observation, step_response.reward, step_response.done, info
