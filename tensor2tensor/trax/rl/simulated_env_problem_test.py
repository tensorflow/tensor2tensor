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

"""Tests for tensor2tensor.trax.rl.simulated_env_problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import gin
import gym
import mock
import numpy as np

from tensor2tensor.trax import backend
from tensor2tensor.trax import trax
from tensor2tensor.trax.rl import simulated_env_problem
from tensorflow import test


class RawSimulatedEnvProblemTest(test.TestCase):

  @staticmethod
  @mock.patch.object(trax, "restore_state", autospec=True)
  def _create_env(mock_restore_state, model, histories,
                  trajectory_length):
    # (model_params, opt_state)
    mock_restore_state.return_value.params = (None, None)
    space = gym.spaces.Discrete(100)
    return simulated_env_problem.RawSimulatedEnvProblem(
        model=model,
        history_length=histories.shape[2],
        trajectory_length=trajectory_length,
        batch_size=1,
        observation_space=space,
        action_space=space,
        reward_range=(-1, 1),
        discrete_rewards=True,
        history_stream=iter(histories),
        output_dir=None,
    )

  def test_communicates_with_model(self):
    # Mock model increasing the observation by action, reward is the parity of
    # the new observation.
    def mock_transition(inputs, *args, **kwargs):
      del args
      del kwargs
      (observations, actions) = inputs
      new_observations = observations[:, -1] + actions
      rewards = np.array([[int(new_observations % 2 == 0)]])
      return (new_observations, rewards), ()

    mock_model_fn = mock.MagicMock()
    mock_model_fn.return_value.side_effect = mock_transition
    mock_model = mock_model_fn.return_value

    actions_to_take = np.array([[1], [3]])
    histories = np.array([[[0, 1, 2, 3]]])
    expected_observations = np.array([[3], [4], [7]])
    expected_rewards = np.array([[1], [0]])
    expected_dones = np.array([[False], [True]])
    expected_histories = np.array([[[0, 1, 2, 3]], [[1, 2, 3, 4]]])
    expected_actions = actions_to_take

    with backend.use_backend("numpy"):
      env = self._create_env(  # pylint: disable=no-value-for-parameter
          model=mock_model_fn,
          histories=histories,
          trajectory_length=len(actions_to_take),
      )
      actual_observations = [env.reset()]
      actual_rewards = []
      actual_dones = []
      actual_histories = []
      actual_actions = []
      for action in actions_to_take:
        (observation, reward, done, _) = env.step(action)
        actual_observations.append(observation)
        actual_rewards.append(reward)
        actual_dones.append(done)
        # Mock call is a tuple (args, kwargs). There is one positional argument,
        # which is a tuple (history, action).
        (((history, action),), _) = mock_model.call_args
        actual_actions.append(action)
        actual_histories.append(history)

    np.testing.assert_array_equal(actual_observations, expected_observations)
    np.testing.assert_array_equal(actual_rewards, expected_rewards)
    np.testing.assert_array_equal(actual_dones, expected_dones)
    np.testing.assert_array_equal(actual_histories, expected_histories)
    np.testing.assert_array_equal(actual_actions, expected_actions)

  def test_takes_new_history(self):
    histories = np.array([[[0, 1, 2]], [[3, 4, 5]]])

    with backend.use_backend("numpy"):
      env = self._create_env(  # pylint: disable=no-value-for-parameter
          model=mock.MagicMock(),
          histories=histories,
          trajectory_length=2,
      )
      env.reset()
      observation = env.reset()
      np.testing.assert_array_equal(observation, [5])


class SerializedSequenceSimulatedEnvProblemTest(test.TestCase):

  @mock.patch.object(trax, "restore_state", autospec=True)
  def test_communicates_with_model(self, mock_restore_state):
    gin.bind_parameter("BoxSpaceSerializer.precision", 1)
    vocab_size = 16
    # Mock model predicting a fixed sequence of symbols. It is made such that
    # the first two observations are equal and the last one is different.
    symbols = [
        1, 1, 2, 2,  # obs1
        1, 1, 2, 2,  # obs2
        1, 2, 2, 1,  # obs3
    ]
    def make_prediction(symbol):
      one_hot = np.eye(vocab_size)[symbol]
      log_probs = (1 - one_hot) * -100.0  # Virtually deterministic.
      # (4 obs symbols + 1 action symbol) * 3 timesteps = 15.
      return np.array([[log_probs] * 15]), ()

    mock_model_fn = mock.MagicMock()
    mock_model = mock_model_fn.return_value
    mock_model.side_effect = map(make_prediction, symbols)

    with backend.use_backend("numpy"):
      # (model_params, opt_state)
      mock_restore_state.return_value.params = (None, None)
      env = simulated_env_problem.SerializedSequenceSimulatedEnvProblem(
          model=mock_model_fn,
          reward_fn=(lambda _1, _2: np.array([0.5])),
          done_fn=(lambda _1, _2: np.array([False])),
          vocab_size=vocab_size,
          max_trajectory_length=3,
          batch_size=1,
          observation_space=gym.spaces.Box(low=0, high=5, shape=(4,)),
          action_space=gym.spaces.Discrete(2),
          reward_range=(-1, 1),
          discrete_rewards=False,
          history_stream=itertools.repeat(None),
          output_dir=None,
      )
      obs1 = env.reset()
      ((inputs,), _) = mock_model.call_args

      act1 = 0
      (obs2, reward, done, _) = env.step(np.array([act1]))
      ((inputs,), _) = mock_model.call_args
      self.assertEqual(inputs[0, 4], act1)
      np.testing.assert_array_equal(inputs[0, :4], symbols[:4])
      np.testing.assert_array_equal(obs1, obs2)
      np.testing.assert_array_equal(reward, [0.5])
      np.testing.assert_array_equal(done, [False])

      act2 = 1
      (obs3, reward, done, _) = env.step(np.array([act2]))
      ((inputs,), _) = mock_model.call_args
      self.assertEqual(inputs[0, 9], act2)
      np.testing.assert_array_equal(inputs[0, 5:9], symbols[4:8])
      self.assertFalse(np.array_equal(obs2, obs3))
      np.testing.assert_array_equal(reward, [0.5])
      np.testing.assert_array_equal(done, [True])


if __name__ == "__main__":
  test.main()
