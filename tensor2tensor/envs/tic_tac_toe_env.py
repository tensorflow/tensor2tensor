# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Gym Tic-Tac-Toe environment.

Environment acts like the second player and first player is either environment
or the agent. The environment follows a random policy for now.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.rl import gym_utils


def encode_pos(i, j):
  """Encodes a pair (i, j) as a scalar position on the board."""
  return 3 * i + j


def decode_pos(pos):
  """Decoes a scalar position on the board as a pair (i, j)."""
  return pos // 3, pos % 3


def get_open_spaces(board):
  """Given a representation of the board, returns a list of open spaces."""
  open_spaces = []
  for i in range(3):
    for j in range(3):
      if board[i][j] == 0:
        open_spaces.append(encode_pos(i, j))
  return open_spaces


def get_reward_and_done(board):
  """Given a representation of the board, returns reward and done."""
  # Returns (reward, done) where:
  # reward: -1 means lost, +1 means win, 0 means draw or continuing.
  # done: True if the game is over, i.e. someone won or it is a draw.

  # Sum all rows ...
  all_sums = [np.sum(board[i, :]) for i in range(3)]
  # ... all columns
  all_sums.extend([np.sum(board[:, i]) for i in range(3)])
  # and both diagonals.
  all_sums.append(np.sum([board[i, i] for i in range(3)]))
  all_sums.append(np.sum([board[i, 2 - i] for i in range(3)]))

  if -3 in all_sums:
    return -1, True

  if 3 in all_sums:
    return 1, True

  done = True
  if get_open_spaces(board):
    done = False

  return 0, done


# TODO(afrozm): This should eventually subclass Problem.
class TicTacToeEnv(gym.Env):
  """Simple TicTacToe Env, starts the game randomly half of the time."""

  def __init__(self, strict=False):
    self.strict = strict

    # What about metadata and spec?
    self.reward_range = (-1.0, 1.0)

    # Action space -- 9 positions that we can chose to mark.
    self.action_space = spaces.Discrete(9)

    # Observation space -- this hopefully does what we need.
    self.observation_space = spaces.Box(
        low=-1, high=1, shape=(3, 3), dtype=np.int64)

    # Set the seed.
    self.np_random = None
    self.seed()

    # Start the game.
    self.board_state = None
    self.done = False
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  # TODO(afrozm): Parametrize by some policy so that the env plays in an optimal
  # way.
  def play_random_move(self):
    # Select open spaces.
    open_spaces = get_open_spaces(self.board_state)

    if not open_spaces:
      return False

    # Choose a space and mark it.
    pos = self.np_random.choice(open_spaces)
    i, j = decode_pos(pos)

    self.board_state[i, j] = -1

  def reset(self):
    self.board_state = np.zeros((3, 3), dtype=np.int64)

    # We"ll start with a 50% chance.
    if self.np_random.choice([0, 1]) == 0:
      self.play_random_move()

    # Return the observation.
    return self.board_state

  def render(self, mode="human"):
    # Unused.
    del mode
    board_str = ""
    for i in range(3):
      for j in range(3):
        pos = self.board_state[i, j]
        if pos == -1:
          board_str += "x"
        elif pos == 0:
          board_str += "-"
        else:
          board_str += "o"
      board_str += "\n"
    return board_str

  def step(self, action):
    # Are we already done?
    if self.strict:
      assert not self.done

    # Action has to belong to the action state.
    assert self.action_space.contains(action)

    # Is it a legitimate move, i.e. is that position open to play?
    is_legit_move = action in get_open_spaces(self.board_state)

    # Shouldn"t be an illegal action -- is a noop if not strict.
    if self.strict:
      assert is_legit_move

    # If strict mode is off, then let this be a noop and env not play either.
    if not is_legit_move:
      return self.board_state, 0, False, {}

    # This is a legit move, perform the action and check if done, etc etc.
    i, j = decode_pos(action)
    self.board_state[i, j] = 1
    reward, done = get_reward_and_done(self.board_state)

    if done:
      self.done = True
      return self.board_state, reward, True, {}

    # If not done already, play our move.
    self.play_random_move()
    reward, done = get_reward_and_done(self.board_state)
    self.done = done
    return self.board_state, reward, self.done, {}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {
        "inputs": modalities.ModalityType.IDENTITY_SYMBOL,
        "targets": modalities.ModalityType.IDENTITY_SYMBOL,
    }
    p.vocab_size = {
        "inputs": 3,  # since at each box, the input is either x, o or -.
        # nevermind that we have a 3x3 box.
        "targets": 3,  # -1, 0, 1
    }
    p.input_space_id = 0  # problem.SpaceID.GENERIC
    p.target_space_id = 0  # problem.SpaceID.GENERIC


# TODO(afrozm): Figure out how to get rid of this.
class DummyPolicyProblemTTT(problem.Problem):
  """Dummy Problem for running the policy."""

  def __init__(self):
    super(DummyPolicyProblemTTT, self).__init__()
    self._ttt_env = TicTacToeEnv()

  def hparams(self, defaults, model_hparams):
    # Update the env's hparams.
    self._ttt_env.hparams(defaults, model_hparams)
    # Do these belong here?
    defaults.modality.update({
        "input_action": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "input_reward": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "target_action": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "target_reward": modalities.ModalityType.SYMBOL_WEIGHTS_ALL,
        "target_policy": modalities.ModalityType.IDENTITY,
        "target_value": modalities.ModalityType.IDENTITY,
    })
    defaults.vocab_size.update({
        "input_action": self.num_actions,
        "input_reward": 3,  # -1, 0, +1 ?
        "target_action": self.num_actions,
        "target_reward": 3,  # -1, 0, +1 ?
        "target_policy": None,
        "target_value": None,
    })

  @property
  def num_actions(self):
    return self._ttt_env.action_space.n


def register():
  # Register this with gym.
  unused_tictactoe_id, unused_tictactoe_env = gym_utils.register_gym_env(
      "tensor2tensor.envs.tic_tac_toe_env:TicTacToeEnv", version="v0")


# TODO(afrozm): Fix the registration and make it automatic.
register()
