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

"""Tests for tensor2tensor.trax.rl.simple."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import gin
import gym
from matplotlib import pyplot as plt
import mock
import numpy as np

from tensor2tensor.envs import trajectory
from tensor2tensor.trax import backend
from tensor2tensor.trax import trax
from tensor2tensor.trax import utils
from tensor2tensor.trax.rl import simple
from tensor2tensor.trax.rl import simulated_env_problem
from tensor2tensor.trax.rl import space_serializer  # pylint: disable=unused-import
from tensorflow import test
from tensorflow.io import gfile


class SimpleTest(test.TestCase):

  def _make_singleton_trajectory(self, observation):
    t = trajectory.Trajectory()
    t.add_time_step(observation=observation)
    return t

  def _dump_trajectory_pickle(self, observations, path):
    pkl_module = utils.get_pickle_module()
    trajectories = list(map(self._make_singleton_trajectory, observations))
    with gfile.GFile(path, "wb") as f:
      pkl_module.dump(trajectories, f)

  def test_loads_trajectories(self):
    temp_dir = self.get_temp_dir()
    # Dump two trajectory pickles with given observations.
    self._dump_trajectory_pickle(
        observations=[0, 1, 2, 3], path=os.path.join(temp_dir, "0.pkl"))
    self._dump_trajectory_pickle(
        observations=[4, 5, 6, 7], path=os.path.join(temp_dir, "1.pkl"))
    (train_trajs, eval_trajs) = simple.load_trajectories(
        temp_dir, eval_frac=0.25)
    extract_obs = lambda t: t.last_time_step.observation
    # The order of pickles is undefined, so we compare sets.
    actual_train_obs = set(map(extract_obs, train_trajs))
    actual_eval_obs = set(map(extract_obs, eval_trajs))

    # First 3 trajectories from each pickle go to train, the last one to eval.
    expected_train_obs = {0, 1, 2, 4, 5, 6}
    expected_eval_obs = {3, 7}
    self.assertEqual(actual_train_obs, expected_train_obs)
    self.assertEqual(actual_eval_obs, expected_eval_obs)

  def test_generates_examples(self):
    observations = [0, 1, 2, 3]
    trajectories = map(self._make_singleton_trajectory, observations)
    trajectory_to_training_examples = lambda t: [t.last_time_step.observation]
    stream = simple.generate_examples(
        trajectories, trajectory_to_training_examples)

    # The examples are shuffled, so we compare sets.
    self.assertEqual(
        set(itertools.islice(stream, len(observations))), set(observations))
    # The stream is infinite, so we should be able to take a next element.
    self.assertIn(next(stream), observations)

  def test_mixes_streams_with_prob_one(self):
    # Mix infinite streams of 0s and 1s.
    stream = simple.mix_streams(
        itertools.repeat(0), itertools.repeat(1), mix_prob=1.0)
    # Mixed stream should have only 0s.
    self.assertEqual(set(itertools.islice(stream, 100)), {0})

  def test_mixes_streams_with_prob_zero(self):
    stream = simple.mix_streams(
        itertools.repeat(0), itertools.repeat(1), mix_prob=0.0)
    # Mixed stream should have only 1s.
    self.assertEqual(set(itertools.islice(stream, 100)), {1})

  def test_mixes_streams_with_prob_half(self):
    stream = simple.mix_streams(
        itertools.repeat(0), itertools.repeat(1), mix_prob=0.5)
    # Mixed stream should have both 0s and 1s.
    self.assertEqual(set(itertools.islice(stream, 100)), {0, 1})

  def test_batches_stream(self):
    stream = iter([(0, 1), (2, 3), (4, 5), (6, 7)])
    batched_stream = simple.batch_stream(stream, batch_size=2)
    np.testing.assert_equal(
        next(batched_stream), (np.array([0, 2]), np.array([1, 3])))
    np.testing.assert_equal(
        next(batched_stream), (np.array([4, 6]), np.array([5, 7])))

  def test_plays_env_problem(self):
    # Shape: (time, trajectory).
    observations = np.array([[0, 1], [2, 3], [4, 5]])
    rewards = np.array([[0, 1], [1, 0]])
    actions = np.array([[1, 2], [2, 0]])
    # We end the second environment 2 times, but we shouldn't collect the second
    # trajectory.
    dones = np.array([[False, True], [True, True]])
    infos = [{}, {}]

    mock_env = mock.MagicMock()
    mock_env.batch_size = 2
    # (observations, lengths)
    mock_env.trajectories.observations_np.return_value = (None, None)
    mock_env.reset.return_value = observations[0]
    mock_env.step.side_effect = zip(observations[1:], rewards, dones, infos)

    mock_policy_fn = mock.MagicMock()
    mock_policy_fn.side_effect = actions

    trajectories = simple.play_env_problem(mock_env, mock_policy_fn)
    self.assertEqual(len(trajectories), 2)
    expected_lengths = [3, 2]
    for (i, (traj, expected_length)) in enumerate(
        zip(trajectories, expected_lengths)):
      self.assertEqual(traj.num_time_steps, expected_length)
      np.testing.assert_array_equal(
          traj.observations_np, observations[:expected_length, i])
      np.testing.assert_array_equal(
          traj.raw_rewards_np, rewards[:(expected_length - 1), i])
      np.testing.assert_array_equal(
          traj.actions_np, actions[:(expected_length - 1), i])

  def _make_trajectory(self, observations=None, actions=None):
    t = trajectory.Trajectory()
    if observations is None:
      observations = itertools.repeat(None)
    if actions is None:
      actions = itertools.repeat(None)
    for (observation, action) in zip(observations, actions):
      t.add_time_step(observation=observation, action=action)
    return t

  def test_replay_policy(self):
    trajectories = [
        self._make_trajectory(actions=actions)
        for actions in map(np.array, [[1, 2], [3]])
    ]
    policy_fn = simple.ReplayPolicy(trajectories, out_of_bounds_action=0)
    np.testing.assert_array_equal(policy_fn(None), [1, 3])
    np.testing.assert_array_equal(policy_fn(None), [2, 0])

  def test_observation_error_zero_for_same_trajectories(self):
    observations = np.array([[0], [2], [1]])
    (traj1, traj2) = map(self._make_trajectory, (observations, observations))
    error = simple.calculate_observation_error([traj1], [traj2])
    np.testing.assert_array_almost_equal(error, [0])

  def test_observation_error_positive_for_different_trajectories(self):
    observations1 = np.array([[1], [2], [3]])
    observations2 = np.array([[0], [2], [3]])
    (traj1, traj2) = map(self._make_trajectory, (observations1, observations2))
    error = simple.calculate_observation_error([traj1], [traj2])
    np.testing.assert_array_less([0], error)

  def test_observation_error_dims_correspond_to_observation_dims(self):
    observations1 = np.array([[0, 1, 0], [0, 2, 0], [0, 3, 0]])
    observations2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    (traj1, traj2) = map(self._make_trajectory, (observations1, observations2))
    error = simple.calculate_observation_error([traj1], [traj2])
    self.assertEqual(error.shape, (3,))
    np.testing.assert_array_almost_equal(error[0], 0)
    self.assertFalse(np.allclose(error[1], 0))
    np.testing.assert_array_almost_equal(error[2], 0)

  def test_observation_error_increases_with_distance(self):
    observations_zero = np.array([[0], [0], [0]])
    observations_positive = np.array([[3], [2], [1]])
    (traj_zero, traj_positive, traj_negative) = map(
        self._make_trajectory,
        (observations_zero, observations_positive, -observations_positive),
    )
    error_small = simple.calculate_observation_error(
        [traj_zero], [traj_positive])
    error_big = simple.calculate_observation_error(
        [traj_positive], [traj_negative])
    np.testing.assert_array_less(error_small, error_big)

  def test_observation_error_increases_with_real_trajectory_length(self):
    observations_real_short = np.array([[1], [2]])
    observations_real_long = np.array([[1], [2], [3]])
    observations_sim = np.array([[0], [1]])
    (traj_real_short, traj_real_long, traj_sim) = map(
        self._make_trajectory,
        (observations_real_short, observations_real_long, observations_sim),
    )
    error_small = simple.calculate_observation_error(
        real_trajectories=[traj_real_short], sim_trajectories=[traj_sim])
    error_big = simple.calculate_observation_error(
        real_trajectories=[traj_real_long], sim_trajectories=[traj_sim])
    np.testing.assert_array_less(error_small, error_big)

  def test_observation_error_same_when_sim_trajectory_longer(self):
    observations_real = np.array([[0], [1]])
    observations_sim_short = np.array([[1], [2]])
    observations_sim_long = np.array([[1], [2], [3]])
    (traj_real, traj_sim_short, traj_sim_long) = map(
        self._make_trajectory,
        (observations_real, observations_sim_short, observations_sim_long),
    )
    error1 = simple.calculate_observation_error(
        real_trajectories=[traj_real], sim_trajectories=[traj_sim_short])
    error2 = simple.calculate_observation_error(
        real_trajectories=[traj_real], sim_trajectories=[traj_sim_long])
    np.testing.assert_array_almost_equal(error1, error2)

  def test_observation_error_reduces_over_trajectories(self):
    observations1 = np.array([[1], [2], [3]])
    observations2 = np.array([[0], [2], [3]])
    (traj1, traj2) = map(self._make_trajectory, (observations1, observations2))
    error = simple.calculate_observation_error([traj1, traj1], [traj2, traj2])
    self.assertEqual(error.shape, (1,))

  @staticmethod
  @mock.patch.object(trax, "restore_state", autospec=True)
  def _make_env(
      mock_restore_state, observation_space, action_space,
      max_trajectory_length, batch_size,
  ):
    # (model_params, opt_state)
    mock_restore_state.return_value.params = (None, None)

    gin.bind_parameter("BoxSpaceSerializer.precision", 1)

    predict_output = (np.array([[[0.0]]] * batch_size))
    mock_model_fn = mock.MagicMock()
    mock_model_fn.return_value.side_effect = itertools.repeat(predict_output)
    mock_model_fn.return_value.initialize_once.return_value = ((), ())

    return simulated_env_problem.SerializedSequenceSimulatedEnvProblem(
        model=mock_model_fn,
        reward_fn=(lambda _1, _2: np.zeros(batch_size)),
        done_fn=(lambda _1, _2: np.full((batch_size,), False)),
        vocab_size=1,
        max_trajectory_length=max_trajectory_length,
        batch_size=batch_size,
        observation_space=observation_space,
        action_space=action_space,
        reward_range=(-1, 1),
        discrete_rewards=False,
        history_stream=itertools.repeat(None),
        output_dir=None,
    )

  def test_evaluates_model_with_vector_observation_space(self):
    with backend.use_backend("numpy"):
      env = self._make_env(  # pylint: disable=no-value-for-parameter
          observation_space=gym.spaces.Box(shape=(2,), low=0, high=1),
          action_space=gym.spaces.Discrete(n=1),
          max_trajectory_length=2,
          batch_size=3,
      )
      trajectories = [
          self._make_trajectory(observations, actions)  # pylint: disable=g-complex-comprehension
          for (observations, actions) in [
              (np.array([[0, 1]]), np.array([0])),
              (np.array([[1, 2], [3, 4]]), np.array([0, 0])),
              (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 0, 0])),
          ]
      ]
      metrics = simple.evaluate_model(env, trajectories, plt)
      self.assertIsNotNone(metrics)
      self.assertEqual(len(metrics), 2)

  def test_fails_to_evaluate_model_with_matrix_observation_space(self):
    with backend.use_backend("numpy"):
      env = self._make_env(  # pylint: disable=no-value-for-parameter
          observation_space=gym.spaces.Box(shape=(2, 2), low=0, high=1),
          action_space=gym.spaces.Discrete(n=1),
          max_trajectory_length=2,
          batch_size=1,
      )
      trajectories = [
          self._make_trajectory(np.array([[0, 1], [2, 3]]), np.array([0]))]
      metrics = simple.evaluate_model(env, trajectories, plt)
      self.assertIsNone(metrics)


if __name__ == "__main__":
  test.main()
