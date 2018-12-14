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

"""Utilities for interacting with Gym classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym


def make_gym_env(name, rl_env_max_episode_steps=-1):
  """Create a gym env optionally wrapped with a time limit wrapper.

  NOTE: The returned env may already be wrapped with TimeLimit!

  Args:
    name: `str` - base name of the gym env to make.
    rl_env_max_episode_steps: `int` or None - Using any value < 0 returns the
      env as-in, otherwise we impose the requested timelimit. Setting this to
      None returns a wrapped env that doesn't have a step limit.

  Returns:
    An instance of `gym.Env` or `gym.wrappers.TimeLimit` with the requested
    step limit.
  """

  # rl_env_max_episode_steps is None or int.
  assert ((not rl_env_max_episode_steps) or
          isinstance(rl_env_max_episode_steps, int))

  env = gym.make(name)

  # If nothing to do, then return the env.
  if rl_env_max_episode_steps and rl_env_max_episode_steps < 0:
    return env

  # Sometimes (mostly?) the env is already wrapped in a TimeLimit wrapper, in
  # which case unwrap it and wrap with the proper time limit requested.
  if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

  return gym.wrappers.TimeLimit(env, max_episode_steps=rl_env_max_episode_steps)


def register_gym_env(class_entry_point, version="v0"):
  """Registers the class with its snake case name in Gym and returns it."""

  split_on_colon = class_entry_point.split(":")
  assert len(split_on_colon) == 2

  class_name = split_on_colon[1]
  # We have to add the version to conform to gym's API.
  env_name = "{}-{}".format(class_name, version)
  gym.envs.register(id=env_name, entry_point=class_entry_point)

  return gym.make(env_name)
