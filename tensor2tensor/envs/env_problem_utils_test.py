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

"""Tests for env_problem_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import tic_tac_toe_env  # pylint: disable=unused-import
from tensor2tensor.envs import tic_tac_toe_env_problem

import tensorflow as tf


class EnvProblemUtilsTest(tf.test.TestCase):

  def test_play_env_problem_randomly(self):
    batch_size = 5
    num_steps = 100

    ep = tic_tac_toe_env_problem.TicTacToeEnvProblem()
    ep.initialize(batch_size=batch_size)

    env_problem_utils.play_env_problem_randomly(ep, num_steps)

    # We've played num_steps * batch_size steps + everytime we get 'done' we
    # create another step + batch_size number of pending steps.
    self.assertEqual(
        num_steps * batch_size + len(ep.trajectories.completed_trajectories) +
        batch_size, ep.trajectories.num_time_steps)


if __name__ == '__main__':
  tf.test.main()
