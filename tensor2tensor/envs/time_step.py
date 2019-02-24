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

"""TimeStep is a simple class that holds the information seen at a time-step.

Let:
r_t = Reward(s_{t-1}, a_{t-1}, s_t)  - reward for getting into a state.
d_t = Done(s_t)                      - is this state terminal.

Then the sequence of states, actions and rewards looks like the following:

s0, a0 s1/r1/d1, a1 s2/r2/d2, a2 s3/r3/d3, ...

TimeStep holds (s_t, d_t, r_t, a_t).

NOTE: When we call step on an environment at time-step t, we supply a_t and in
return the env gives us s_{t+1}, d_{t+1}, r_{t+1}

So, we'd have to add the actions a_t to the current time-step, but add the
observations, rewards and dones to a new time-step.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class TimeStep(
    collections.namedtuple(
        "TimeStep",
        ["observation", "done", "raw_reward", "processed_reward", "action"])):
  """This class represents the time-step as mentioned above."""

  def replace(self, **kwargs):
    """Exposes the underlying namedtuple replace."""

    # NOTE: This RETURNS a NEW time-step with the replacements, i.e. doesn't
    # modify self, since namedtuple is immutable.

    # This allows this to be called like ts.replace(action=a, raw_reward=r) etc.

    return self._replace(**kwargs)

  @classmethod
  def create_time_step(cls,
                       observation=None,
                       done=False,
                       raw_reward=None,
                       processed_reward=None,
                       action=None):
    """Creates a TimeStep with both rewards and actions as optional."""

    return cls(observation, done, raw_reward, processed_reward, action)
