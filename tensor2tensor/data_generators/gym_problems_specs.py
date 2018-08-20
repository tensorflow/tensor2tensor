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

# Game list from our list of ROMs
# Removed because XDeterministic-v4 did not exist:
# * adventure
# * defender
# * kaboom
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

# Subset of games with promissing results on model based training.
ATARI_WHITELIST_GAMES = [
    "amidar",
    "bank_heist",
    "berzerk",
    "boxing",
    "breakout",
    "crazy_climber",
    "freeway",
    "frostbite",
    "gopher",
    "hero",
    "kung_fu_master",
    "pong",
    "road_runner",
    "seaquest",
    # TODO(blazej): check if we get equally good results on vanilla pong.
    "wrapped_full_pong",
]

ATARI_ALL_MODES_SHORT_LIST = [
    "pong",
    "boxing",
]

# Different ATARI game modes in OpenAI Gym. Full list here:
# https://github.com/openai/gym/blob/master/gym/envs/__init__.py
ATARI_GAME_MODES = [
    "Deterministic-v0",  # 0.25 repeat action probability, 4 frame skip.
    "Deterministic-v4",  # 0.00 repeat action probability, 4 frame skip.
    "NoFrameskip-v0",    # 0.25 repeat action probability, 1 frame skip.
    "NoFrameskip-v4",    # 0.00 repeat action probability, 1 frame skip.
    "-v0",               # 0.25 repeat action probability, (2 to 5) frame skip.
    "-v4"                # 0.00 repeat action probability, (2 to 5) frame skip.
]

# List of all ATARI envs in all modes.
ATARI_PROBLEMS = {}


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


class GymClippedRewardRandom(GymDiscreteProblem):
  """Abstract base class for clipped reward games."""

  @property
  def env_name(self):
    raise NotImplementedError

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 3


def create_problems_for_game(
    game_name,
    clipped_reward=True,
    game_mode="Deterministic-v4"):
  """Create and register problems for game_name.

  Args:
    game_name: str, one of the games in ATARI_GAMES, e.g. "bank_heist".
    clipped_reward: bool, whether the rewards should be clipped. False is not
      yet supported.
    game_mode: the frame skip and sticky keys config.

  Returns:
    dict of problems with keys ("base", "agent", "simulated").

  Raises:
    ValueError: if clipped_reward=False or game_name not in ATARI_GAMES.
  """
  if not clipped_reward:
    raise ValueError("Creating problems without clipped reward is not "
                     "yet supported.")
  if game_name not in ATARI_GAMES:
    raise ValueError("Game %s not in ATARI_GAMES" % game_name)
  if game_mode not in ATARI_GAME_MODES:
    raise ValueError("Unknown ATARI game mode: %s." % game_mode)
  camel_game_name = "".join(
      [w[0].upper() + w[1:] for w in game_name.split("_")])
  camel_game_name += game_mode
  env_name = camel_game_name
  wrapped_env_name = "T2T%s" % env_name

  # Register an environment that does the reward clipping
  gym.envs.register(
      id=wrapped_env_name,
      entry_point=lambda: gym_utils.wrapped_factory(  # pylint: disable=g-long-lambda
          env=env_name, reward_clipping=True))

  # Create and register the Random and WithAgent Problem classes
  problem_cls = type("Gym%sRandom" % camel_game_name,
                     (GymClippedRewardRandom,),
                     {"env_name": wrapped_env_name})
  registry.register_problem(problem_cls)

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

  return {
      "base": problem_cls,
      "agent": with_agent_cls,
      "simulated": simulated_cls,
  }

# Register the atari games with all of the possible modes.
for game in ATARI_ALL_MODES_SHORT_LIST:
  ATARI_PROBLEMS[game] = {}
  for mode in ATARI_GAME_MODES:
    classes = create_problems_for_game(
        game,
        clipped_reward=True,
        game_mode=mode)
    ATARI_PROBLEMS[game][mode] = classes


