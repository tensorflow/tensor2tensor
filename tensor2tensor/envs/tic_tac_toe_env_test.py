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

"""Tests for tensor2tensor.envs.tic_tac_toe_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.envs import tic_tac_toe_env as ttt_env
import tensorflow.compat.v1 as tf


class TicTacToeEnvTest(tf.test.TestCase):

  def test_start(self):
    ttt = ttt_env.TicTacToeEnv(strict=True)
    self.assertFalse(ttt.done)

    # At max one move may have been played by the env.
    spaces = ttt_env.get_open_spaces(ttt.board_state)
    num_open_spaces = len(spaces)
    # i.e. either 8 or 9
    self.assertGreater(num_open_spaces, 7)

    # Play a move
    observation, reward, done, unused_info = ttt.step(spaces[0])

    # The environment should also have played a move.
    spaces = ttt_env.get_open_spaces(observation)
    self.assertEqual(num_open_spaces - 2, len(spaces))

    # Since at-max 3 moves have been played, the game can't end.
    self.assertEqual(reward, 0)
    self.assertFalse(done)

  def test_env_actions(self):
    # Environment keeps taking actions and not us, we should eventually lose.
    ttt = ttt_env.TicTacToeEnv(strict=True)
    for _ in range(9):
      ttt.play_random_move()
      if ttt.done:
        break

    reward, done = ttt_env.get_reward_and_done(ttt.board_state)
    self.assertEqual(-1, reward)
    self.assertTrue(done)

  def test_keep_playing(self):
    ttt = ttt_env.TicTacToeEnv(strict=False)
    done = False
    while not done:
      # sample an action from the action space.
      action = ttt.action_space.sample()
      # play it -- could be a no-op since we don't see if positions are empty.
      unused_observation, reward, done, unused_info = ttt.step(action)

    # done is True, so either:
    # we won
    # env won or
    # no space left

    we_won = reward == 1
    env_won = reward == -1
    space = bool(ttt_env.get_open_spaces(ttt.board_state))
    self.assertTrue(we_won or env_won or not space)


if __name__ == '__main__':
  tf.test.main()
