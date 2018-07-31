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

import gym

# We need gym_utils for the game environments defined there.
from tensor2tensor.data_generators import gym_utils  # pylint: disable=unused-import
# pylint: disable=g-multiple-import
from tensor2tensor.data_generators.gym_problems import GymDiscreteProblem,\
  GymSimulatedDiscreteProblem, GymRealDiscreteProblem, \
  GymDiscreteProblemWithAutoencoder, GymDiscreteProblemAutoencoded, \
  GymSimulatedDiscreteProblemAutoencoded
# pylint: enable=g-multiple-import
from tensor2tensor.utils import registry

ATARI_GAMES = [
    "air_raid", "alien", "amidar", "assault", "asterix", "asteroids",
    "atlantis", "bank_heist", "battle_zone", "beam_rider", "berzerk", "bowling",
    "boxing", "breakout", "carnival", "centipede", "chopper_command",
    "crazy_climber", "demon_attack", "double_dunk", "elevator_action", "enduro",
    "fishing_derby", "freeway", "frostbite", "gopher", "gravitar", "hero",
    "ice_hockey", "jamesbond", "journey_escape", "kangaroo", "krull",
    "kung_fu_master", "montezuma_revenge", "ms_pacman", "name_this_game",
    "phoenix", "pitfall", "pong", "pooyan", "private_eye", "qbert", "riverraid",
    "road_runner", "robotank", "seaquest", "skiing", "solaris",
    "space_invaders", "star_gunner", "tennis", "time_pilot", "tutankham",
    "up_n_down", "venture", "video_pinball", "wizard_of_wor", "yars_revenge",
    "zaxxon"
]
# Removed because XDeterministic-v4 did not exist:
# * adventure
# * defender
# * kaboom


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
class GymWrappedFullPongRandom(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "T2TPongWarmUp20RewSkipFull-v1"

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
  """Simulated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_pong"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedFullPong(GymRealDiscreteProblem,
                                                   GymWrappedFullPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedFullPongWithAutoencoder(
    GymDiscreteProblemWithAutoencoder, GymWrappedFullPongRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnWrappedFullPongAutoencoded(
    GymDiscreteProblemAutoencoded, GymWrappedFullPongRandom):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedFullPong(
    GymSimulatedDiscreteProblem, GymWrappedFullPongRandom):
  """Simulated pong."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_full_pong"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnWrappedFullPongAutoencoded(
    GymSimulatedDiscreteProblemAutoencoded, GymWrappedFullPongRandom):
  """GymSimulatedDiscreteProblemWithAgentOnWrappedFullPongAutoencoded."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_wrapped_full_pong_autoencoded"

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
  """Simulated breakout."""

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
  """Simulated freeway."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_freeway"

  @property
  def num_testing_steps(self):
    return 100


class GymClippedRewardRandom(GymDiscreteProblem):
  """Base class for clipped reward games."""

  @property
  def env_name(self):
    raise NotImplementedError

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


def dynamically_create_gym_clipped_reward_problem(game_name):
  """Dynamically create env wrapper and Problems for game."""
  # e.g. game_name == bank_heist
  assert game_name in ATARI_GAMES
  camel_game_name = "".join(
      [w[0].upper() + w[1:] for w in game_name.split("_")])
  env_name = "%sDeterministic-v4" % camel_game_name
  wrapped_env_name = "T2T%s" % env_name

  # Register an environment that does the reward clipping
  gym.envs.register(
      id=wrapped_env_name,
      entry_point=lambda: gym_utils.wrapped_factory(  # pylint: disable=g-long-lambda
          env=env_name, reward_clipping=True))

  # Create and register the Random and WithAgent Problem classes
  problem_cls = type(camel_game_name + "Random", (GymClippedRewardRandom,),
                     {"env_name": wrapped_env_name})
  with_agent_cls = type("GymDiscreteProblemWithAgentOn%s" % camel_game_name,
                        (GymRealDiscreteProblem, problem_cls), {})
  registry.register_problem(with_agent_cls)

  # Create and register the simulated Problem
  simulated_cls = type(
      "GymSimulatedDiscreteProblemWithAgentOn%s" % camel_game_name,
      (GymSimulatedDiscreteProblem, problem_cls), {
          "initial_frames_problem": with_agent_cls.name,
          "num_testing_steps": 100
      })
  registry.register_problem(simulated_cls)
