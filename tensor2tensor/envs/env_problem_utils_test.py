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

"""Tests for env_problem_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.envs import env_problem_utils
from tensor2tensor.envs import gym_env_problem
from tensor2tensor.envs import tic_tac_toe_env  # pylint: disable=unused-import
from tensor2tensor.envs import tic_tac_toe_env_problem

import tensorflow.compat.v1 as tf


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

  def test_play_env_problem_with_policy(self):
    env = gym_env_problem.GymEnvProblem(
        base_env_name="CartPole-v0", batch_size=2, reward_range=(-1, 1))

    # Let's make sure that at-most 4 observations come to the policy function.
    len_history_for_policy = 4

    def policy_fun(observations, lengths, state=None, rng=None):
      del lengths
      b = observations.shape[0]
      # Assert that observations from time-step len_history_for_policy onwards
      # are zeros.
      self.assertTrue(
          np.all(observations[:, len_history_for_policy:, ...] == 0))
      self.assertFalse(
          np.all(observations[:, :len_history_for_policy, ...] == 0))
      a = env.action_space.n
      p = np.random.uniform(size=(b, 1, a))
      p = np.exp(p)
      p = p / np.sum(p, axis=-1, keepdims=True)
      return np.log(p), np.mean(p, axis=-1), state, rng

    max_timestep = 15
    num_trajectories = 2
    trajectories, _, _, _ = env_problem_utils.play_env_problem_with_policy(
        env,
        policy_fun,
        num_trajectories=num_trajectories,
        max_timestep=max_timestep,
        len_history_for_policy=len_history_for_policy)

    self.assertEqual(num_trajectories, len(trajectories))

    # Check shapes within trajectories.
    traj = trajectories[0]
    T = traj[1].shape[0]  # pylint: disable=invalid-name
    self.assertEqual((T + 1, 4), traj[0].shape)  # (4,) is OBS
    self.assertEqual((T,), traj[2].shape)
    self.assertEqual(T, len(traj[4]["log_prob_actions"]))
    self.assertEqual(T, len(traj[4]["value_predictions"]))
    self.assertLessEqual(T, max_timestep)

    traj = trajectories[1]
    T = traj[1].shape[0]  # pylint: disable=invalid-name
    self.assertEqual((T + 1, 4), traj[0].shape)
    self.assertEqual((T,), traj[2].shape)
    self.assertEqual(T, len(traj[4]["log_prob_actions"]))
    self.assertEqual(T, len(traj[4]["value_predictions"]))
    self.assertLessEqual(T, max_timestep)


if __name__ == "__main__":
  tf.test.main()
