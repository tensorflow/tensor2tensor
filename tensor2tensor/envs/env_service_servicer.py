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

"""Implementation of the EnvService RPC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import grpc
import numpy as np
from tensor2tensor.envs import env_service_pb2
from tensor2tensor.envs import env_service_pb2_grpc
from tensor2tensor.envs import env_service_serialization as serialization


class EnvServiceServicer(env_service_pb2_grpc.EnvServiceServicer):
  """Implementation of the EnvService service defined in env_service.proto."""

  def __init__(self, env):
    self._env = env

  def Reset(self, request, context):
    """Reset."""
    logging.vlog(1, "EnvServiceServicer is being reset.")

    obs = self._env.reset()
    reset_response = env_service_pb2.ResetResponse()
    # Anything more efficient?
    reset_response.observation.CopyFrom(
        serialization.numpy_array_to_observation(obs))

    return reset_response

  def Step(self, step_request, context):
    """Step."""
    logging.vlog(1, "EnvServiceServicer is being stepped.")

    step_response = env_service_pb2.StepResponse()
    action = step_request.action

    if "discrete_action" != action.WhichOneof("payload"):
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details("Method not implemented for non-discrete actions!")
      return step_response

    obs, rewards, dones, infos = self._env.step(
        np.array([action.discrete_action]))

    step_response.observation.CopyFrom(
        serialization.numpy_array_to_observation(obs))
    step_response.reward = rewards
    step_response.done = dones

    # TODO(afrozm): Take care of this later. `info` is an np array of dicts.
    if len(infos) > 1:
      logging.error("Skipping adding the info for other elements in batch.")

    for k, v in infos[0].items():
      step_response.info.info_map[k] = v

    return step_response

  def Close(self, request, context):
    """Close."""

    self._env.close()
    return env_service_pb2.CloseResponse()

  def Render(self, request, context):
    """Render."""

    mode = request.mode or "rgb_array"
    rendered_value = self._env.render(mode=mode)
    response = env_service_pb2.RenderResponse()
    if (rendered_value is not None) and isinstance(rendered_value, np.ndarray):
      response.observation = serialization.numpy_array_to_observation(
          rendered_value)

    return response

  def GetEnvInfo(self, request, context):
    # Request is empty.
    del request
    del context

    response = env_service_pb2.EnvInfoResponse()

    response.observation_space.CopyFrom(
        serialization.gym_space_to_proto(self._env.observation_space))
    response.action_space.CopyFrom(
        serialization.gym_space_to_proto(self._env.action_space))
    response.reward_range.CopyFrom(
        serialization.reward_range_to_proto(self._env.reward_range))
    # Usually these envs aren't batched envs, in that case batch size = 1.
    response.batch_size = getattr(self._env, "batch_size", 1)

    return response
