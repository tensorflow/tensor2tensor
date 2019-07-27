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

"""Utilities for serializing numpy arrays, gym spaces and envs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from tensor2tensor.envs import env_service_pb2
from tensorflow.python.framework import tensor_util  # pylint: disable=g-direct-tensorflow-import


def numpy_array_to_observation(array):
  obs = env_service_pb2.Observation()
  obs.observation.CopyFrom(tensor_util.make_tensor_proto(array))
  return obs


def tensor_proto_to_numpy_array(tensor_proto):
  return tensor_util.MakeNdarray(tensor_proto)


def step_request_from_discrete_action(action):
  action_proto = env_service_pb2.Action(discrete_action=action)
  step_request = env_service_pb2.StepRequest()
  step_request.action.CopyFrom(action_proto)
  return step_request


def gym_space_to_proto(gym_space):
  """Converts a gym space to `env_service_pb2.GymSpace`."""

  if isinstance(gym_space, gym.spaces.Discrete):
    return env_service_pb2.GymSpace(
        discrete=env_service_pb2.SpaceDiscrete(num_actions=gym_space.n))
  elif isinstance(gym_space, gym.spaces.Box):
    space_proto = env_service_pb2.GymSpace()
    box_proto = space_proto.box

    # Set low & high first, we can set shape and type from it later.
    box_proto.low.CopyFrom(tensor_util.make_tensor_proto(gym_space.low))
    box_proto.high.CopyFrom(tensor_util.make_tensor_proto(gym_space.high))

    # dtype and shape.
    box_proto.dtype = box_proto.low.dtype
    box_proto.shape.CopyFrom(box_proto.low.tensor_shape)

    return space_proto

  # A space that we haven't implemented.
  return env_service_pb2.GymSpace(unimplemented_space=True)


def proto_to_gym_space(gym_space_proto):
  """Converts a `env_service_pb2.GymSpace` to a `gym.spaces`."""

  if gym_space_proto.unimplemented_space:
    return None

  if gym_space_proto.HasField("discrete"):
    return gym.spaces.Discrete(gym_space_proto.discrete.num_actions)

  assert gym_space_proto.HasField("box")

  low_np = tensor_proto_to_numpy_array(gym_space_proto.box.low)
  high_np = tensor_proto_to_numpy_array(gym_space_proto.box.high)

  return gym.spaces.Box(low=low_np, high=high_np, dtype=low_np.dtype)


def reward_range_to_proto(reward_range=None):
  if reward_range is None:
    reward_range = (-np.inf, np.inf)
  return env_service_pb2.RewardRange(low=reward_range[0], high=reward_range[1])
