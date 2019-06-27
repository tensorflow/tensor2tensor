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

"""A fake gym environment.

Can specify either:
1. A done action, i.e. the action on which the environment returns done.
2. A done time-step, i.e. the time step at which the environment returns done.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np


class FakeEnv(object):
  """A fake env which is either done with a specific action or a time-step."""

  def __init__(self,
               input_shape=(4,),
               n_actions=2,
               done_time_step=None,
               done_action=None):
    self._input_shape = input_shape
    self._done_time_step = done_time_step
    self._done_action = done_action
    self._t = 0
    self.action_space = gym.spaces.Discrete(n_actions)
    self.observation_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=input_shape)

  def _get_random_observation(self):
    return np.random.random(self._input_shape)

  def reset(self):
    self._t = 0
    return self._get_random_observation()

  def step(self, action):
    done = False
    if self._done_action is not None:
      done = action == self._done_action
    elif self._done_time_step is not None:
      done = self._t == self._done_time_step

    reward = -1.0 if not done else 1.0
    self._t += 1
    return self._get_random_observation(), reward, done, {}
