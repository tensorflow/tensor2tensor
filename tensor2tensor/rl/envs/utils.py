# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Utilities for using batched environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf


class EvalVideoWrapper(gym.Wrapper):
  """Wrapper for recording videos during eval phase.

  This wrapper is designed to record videos via gym.wrappers.Monitor and
  simplifying its usage in t2t collect phase.
  It alleviate the limitation of Monitor, which doesn't allow reset on an
  active environment.

  EvalVideoWrapper assumes that only every second trajectory (after every
  second reset) will be used by the caller:
  - on the "active" runs it behaves as gym.wrappers.Monitor,
  - on the "inactive" runs it doesn't call underlying environment and only
    returns last seen observation.
  Videos are only generated during the active runs.
  """

  def __init__(self, env):
    super(EvalVideoWrapper, self).__init__(env)
    self._reset_counter = 0
    self._active = False
    self._last_returned = None

  def _step(self, action):
    if self._active:
      self._last_returned = self.env.step(action)
    if self._last_returned is None:
      raise Exception("Environment stepped before proper reset.")
    return self._last_returned

  def _reset(self, **kwargs):
    self._reset_counter += 1
    if self._reset_counter % 2 == 1:
      self._active = True
      return self.env.reset(**kwargs)

    self._active = False
    self._last_returned = (self._last_returned[0],
                           self._last_returned[1],
                           False,  # done = False
                           self._last_returned[3])
    return self._last_returned[0]


def get_action_space(environment_spec):
  """Get action spece associated with environment spec.

  Args:
     environment_spec:  EnvironmentSpec object

  Returns:
    OpenAi Gym action space
  """
  action_space = environment_spec.env_lambda().action_space
  action_shape = list(parse_shape(action_space))
  action_dtype = parse_dtype(action_space)

  return action_space, action_shape, action_dtype


def get_policy(observations, hparams):
  """Get a policy network.

  Args:
    observations: Tensor with observations
    hparams: parameters

  Returns:
    Tensor with policy and value function output
  """
  policy_network_lambda = hparams.policy_network
  action_space, _, _ = get_action_space(hparams.environment_spec)
  return policy_network_lambda(action_space, hparams, observations)


def parse_shape(space):
  """Get a tensor shape from a OpenAI Gym space.

  Args:
    space: Gym space.

  Returns:
    Shape tuple.
  """
  if isinstance(space, gym.spaces.Discrete):
    return ()
  if isinstance(space, gym.spaces.Box):
    return space.shape
  raise NotImplementedError()


def parse_dtype(space):
  """Get a tensor dtype from a OpenAI Gym space.

  Args:
    space: Gym space.

  Returns:
    TensorFlow data type.
  """
  if isinstance(space, gym.spaces.Discrete):
    return tf.int32
  if isinstance(space, gym.spaces.Box):
    return tf.float32
  raise NotImplementedError()
