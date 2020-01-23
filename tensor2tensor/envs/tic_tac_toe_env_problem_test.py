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

"""Tests for tensor2tensor.envs.tic_tac_toe_env_problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import tic_tac_toe_env  # pylint: disable=unused-import
from tensor2tensor.envs import tic_tac_toe_env_problem  # pylint: disable=unused-import
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf


class TicTacToeEnvProblemTest(tf.test.TestCase):

  def test_registration_and_interaction_with_env_problem(self):
    batch_size = 5
    # This ensures that registration has occurred.
    ep = registry.env_problem("tic_tac_toe_env_problem", batch_size=batch_size)
    ep.reset()
    num_done, num_lost, num_won, num_draw = 0, 0, 0, 0
    nsteps = 100
    for _ in range(nsteps):
      actions = np.stack([ep.action_space.sample() for _ in range(batch_size)])
      obs, rewards, dones, infos = ep.step(actions)

      # Assert that things are happening batchwise.
      self.assertEqual(batch_size, len(obs))
      self.assertEqual(batch_size, len(rewards))
      self.assertEqual(batch_size, len(dones))
      self.assertEqual(batch_size, len(infos))

      done_indices = env_problem_utils.done_indices(dones)
      ep.reset(done_indices)
      num_done += sum(dones)
      for r, d in zip(rewards, dones):
        if not d:
          continue
        if r == -1:
          num_lost += 1
        elif r == 0:
          num_draw += 1
        elif r == 1:
          num_won += 1
        else:
          raise ValueError("reward should be -1, 0, 1 but is {}".format(r))

    # Assert that something got done atleast, without that the next assert is
    # meaningless.
    self.assertGreater(num_done, 0)

    # Assert that things are consistent.
    self.assertEqual(num_done, num_won + num_lost + num_draw)


if __name__ == "__main__":
  tf.test.main()
