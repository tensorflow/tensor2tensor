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

"""Environments defined in T2T. Imports here force registration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.envs.registration import register

from tensor2tensor.envs import client_env
from tensor2tensor.envs import gym_env_problem
from tensor2tensor.envs import tic_tac_toe_env
from tensor2tensor.envs import tic_tac_toe_env_problem


def register_env(env_class):
  register(
      id="{}-v0".format(env_class.__name__),
      entry_point="tensor2tensor.envs:{}".format(env_class.__name__),
  )
  return env_class


# TODO(afrozm): Register TicTacToeEnv the same way.
# register_env(tic_tac_toe_env.TicTacToeEnv)
ClientEnv = register_env(client_env.ClientEnv)  # pylint: disable=invalid-name
