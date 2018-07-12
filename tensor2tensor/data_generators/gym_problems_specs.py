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
"""Definitions of data generators for gym problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# We need gym_utils for the game environments defined there.
from tensor2tensor.data_generators import gym_utils  # pylint: disable=unused-import
# pylint: disable=g-multiple-import
from tensor2tensor.data_generators.gym_problems import GymDiscreteProblem,\
  GymSimulatedDiscreteProblem, GymRealDiscreteProblem, \
  GymDiscreteProblemWithAutoencoder, GymDiscreteProblemAutoencoded, \
  GymSimulatedDiscreteProblemAutoencoded
# pylint: enable=g-multiple-import
from tensor2tensor.utils import registry


@registry.register_problem
class GymPongRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  # Hard-coding num_actions, frame_height, frame_width to avoid loading
  # libale.so file.
  @property
  def num_actions(self):
    return 6

  @property
  def frame_height(self):
    return 210

  @property
  def frame_width(self):
    return 160

  @property
  def env_name(self):
    return "PongDeterministic-v4"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


@registry.register_problem
class GymWrappedPongRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "T2TPongWarmUp20RewSkip200Steps-v1"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


@registry.register_problem
class GymWrappedLongPongRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "T2TPongWarmUp20RewSkip2000Steps-v1"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymWrappedBreakoutRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "T2TBreakoutWarmUp20RewSkip500Steps-v1"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnPong(GymSimulatedDiscreteProblem,
                                                 GymPongRandom):
  """Simulated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_pong"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymFreewayRandom(GymDiscreteProblem):
  """Freeway game, random actions."""

  @property
  def env_name(self):
    return "FreewayDeterministic-v4"

  @property
  def min_reward(self):
    return 0

  @property
  def num_rewards(self):
    return 2


@registry.register_problem
class GymDiscreteProblemWithAgentOnPong(GymRealDiscreteProblem, GymPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnFreeway(GymRealDiscreteProblem,
                                           GymFreewayRandom):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedPong(
    GymSimulatedDiscreteProblem, GymWrappedPongRandom):
  """Similated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_pong"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedLongPong(GymRealDiscreteProblem,
                                                   GymWrappedLongPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedLongPongWithAutoencoder(
    GymDiscreteProblemWithAutoencoder, GymWrappedLongPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedLongPongAutoencoded(
    GymDiscreteProblemAutoencoded, GymWrappedLongPongRandom):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedLongPong(
    GymSimulatedDiscreteProblem, GymWrappedLongPongRandom):
  """Simulated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_long_pong"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedLongPongAutoencoded(
    GymSimulatedDiscreteProblemAutoencoded, GymWrappedLongPongRandom):
  """GymSimulatedDiscreteProblemWithAgentOnWrappedLongPongAutoencoded."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_long_pong_autoencoded"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedBreakout(GymRealDiscreteProblem,
                                                   GymWrappedBreakoutRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedBreakoutAe(
    GymDiscreteProblemWithAgentOnWrappedBreakout):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedBreakout(
    GymSimulatedDiscreteProblem, GymWrappedBreakoutRandom):
  """Similated breakout."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_breakout"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedPong(GymRealDiscreteProblem,
                                               GymWrappedPongRandom):
  """GymDiscreteProblemWithAgentOnWrappedPong."""

  # Hard-coding num_actions, frame_height, frame_width to avoid loading
  # libale.so file.
  @property
  def num_actions(self):
    return 6

  @property
  def frame_height(self):
    return 210

  @property
  def frame_width(self):
    return 160


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedPongAe(  # With autoencoder.
    GymDiscreteProblemWithAgentOnWrappedPong):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnFreeway(GymSimulatedDiscreteProblem,
                                                    GymFreewayRandom):
  """Similated freeway."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_freeway"

  @property
  def num_testing_steps(self):
    return 100

