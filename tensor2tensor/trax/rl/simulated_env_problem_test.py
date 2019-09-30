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

from tensor2tensor.envs import trajectory
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
      return (new_observations, rewards)

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

  def _make_env(
      self, observation_space, action_space, vocab_size,
      predict_fn=None, reward_fn=None, done_fn=None,
      batch_size=None, max_trajectory_length=None,
  ):
    mock_model_fn = mock.MagicMock()
    if predict_fn is not None:
      mock_model_fn.return_value = predict_fn
      mock_model_fn.return_value.initialize_once.return_value = ((), ())
    return simulated_env_problem.SerializedSequenceSimulatedEnvProblem(
        model=mock_model_fn,
        reward_fn=reward_fn,
        done_fn=done_fn,
        vocab_size=vocab_size,
        max_trajectory_length=max_trajectory_length,
        batch_size=batch_size,
        observation_space=observation_space,
        action_space=action_space,
        reward_range=(-1, 1),
        discrete_rewards=False,
        history_stream=itertools.repeat(None),
        output_dir=None,
    )

  def _make_trajectory(self, observations, actions):
    assert len(observations) == len(actions) + 1
    t = trajectory.Trajectory()
    for (obs, act) in zip(observations, actions):
      t.add_time_step(observation=obs, action=act, done=False)
    t.add_time_step(observation=observations[-1], done=True)
    return t

  @mock.patch.object(trax, "restore_state", autospec=True)
  def test_communicates_with_model(self, mock_restore_state):
    gin.bind_parameter("BoxSpaceSerializer.precision", 1)
    vocab_size = 16
    # Mock model predicting a fixed sequence of symbols. It is made such that
    # the first two observations are different and the last one is equal to the
    # first.
    symbols = [
        1, 1, 2, 2, 0, 0,  # obs1 act1
        1, 2, 2, 1, 0, 0,  # obs2 act2
        1, 1, 2, 2,        # obs3
    ]
    def make_prediction(symbol):
      one_hot = np.eye(vocab_size)[symbol]
      log_probs = (1 - one_hot) * -100.0  # Virtually deterministic.
      # (4 obs symbols + 1 action symbol) * 3 timesteps = 15.
      return np.array([[log_probs]])

    mock_predict_fn = mock.MagicMock()
    mock_predict_fn.side_effect = map(make_prediction, symbols)

    with backend.use_backend("numpy"):
      # (model_params, opt_state)
      mock_restore_state.return_value.params = (None, None)
      env = self._make_env(
          predict_fn=mock_predict_fn,
          reward_fn=(lambda _1, _2: np.array([0.5])),
          done_fn=(lambda _1, _2: np.array([False])),
          vocab_size=vocab_size,
          batch_size=1,
          max_trajectory_length=3,
          observation_space=gym.spaces.Box(low=0, high=5, shape=(4,)),
          action_space=gym.spaces.MultiDiscrete(nvec=[2, 2]),
      )

      def assert_input_suffix(expected_symbols):
        actual_symbols = np.array([
            symbol.item() for ((symbol,), _) in mock_predict_fn.call_args_list[
                -len(expected_symbols):
            ]
        ])
        np.testing.assert_array_equal(actual_symbols, expected_symbols)

      actions = [[0, 1], [1, 0]]

      obs1 = env.reset()
      assert_input_suffix(symbols[:3])

      (obs2, reward, done, _) = env.step(np.array([actions[0]]))
      # Symbols going into the decoder when predicting the next observation are:
      # the last symbol of the previous observation, all action symbols, all
      # symbols but the last one of the next observation.
      assert_input_suffix([symbols[3]] + actions[0] + symbols[6:9])
      self.assertFalse(np.array_equal(obs1, obs2))
      np.testing.assert_array_equal(reward, [0.5])
      np.testing.assert_array_equal(done, [False])

      (obs3, reward, done, _) = env.step(np.array([actions[1]]))
      assert_input_suffix([symbols[9]] + actions[1] + symbols[12:15])
      np.testing.assert_array_equal(obs1, obs3)
      np.testing.assert_array_equal(reward, [0.5])
      np.testing.assert_array_equal(done, [True])

  def test_makes_training_example(self):
    env = self._make_env(
        vocab_size=2,
        observation_space=gym.spaces.Discrete(2),
        action_space=gym.spaces.Discrete(2),
        max_trajectory_length=3,
    )
    t = self._make_trajectory(observations=[0, 1, 0], actions=[1, 0])
    examples = env.trajectory_to_training_examples(t)

    # There should be 1 example with the whole trajectory.
    self.assertEqual(len(examples), 1)
    [(inputs, targets, weights)] = examples
    # inputs == targets for autoregressive sequence prediction.
    np.testing.assert_array_equal(inputs, targets)
    # Assert array shapes and datatypes.
    self.assertEqual(inputs.shape, env.model_input_shape)
    self.assertEqual(inputs.dtype, env.model_input_dtype)
    self.assertEqual(weights.shape, env.model_input_shape)
    # Actions should be masked out.
    self.assertEqual(np.min(weights), 0)
    # At least part of the observation should have full weight.
    self.assertEqual(np.max(weights), 1)

  def test_makes_training_examples_from_trajectories_of_different_lengths(self):
    env = self._make_env(
        vocab_size=2,
        observation_space=gym.spaces.Discrete(2),
        action_space=gym.spaces.Discrete(2),
        max_trajectory_length=3,
    )
    t1 = self._make_trajectory(observations=[0, 1], actions=[1])
    [(x1, _, w1)] = env.trajectory_to_training_examples(t1)
    t2 = self._make_trajectory(observations=[0, 1, 0], actions=[1, 0])
    [(x2, _, w2)] = env.trajectory_to_training_examples(t2)

    # Examples should be padded to the same shape.
    self.assertEqual(x1.shape, x2.shape)
    self.assertEqual(w1.shape, w2.shape)
    # Cumulative weight should increase with trajectory length.
    self.assertGreater(np.sum(w2), np.sum(w1))

  def test_masked_representation_changes_with_observation(self):
    env = self._make_env(
        vocab_size=2,
        observation_space=gym.spaces.Discrete(2),
        action_space=gym.spaces.Discrete(2),
        max_trajectory_length=3,
    )
    t1 = self._make_trajectory(observations=[0, 1], actions=[1])
    [(x1, _, w1)] = env.trajectory_to_training_examples(t1)
    t2 = self._make_trajectory(observations=[0, 0], actions=[1])
    [(x2, _, w2)] = env.trajectory_to_training_examples(t2)

    self.assertFalse(np.array_equal(x1 * w1, x2 * w2))

  def test_masked_representation_doesnt_change_with_action(self):
    env = self._make_env(
        vocab_size=2,
        observation_space=gym.spaces.Discrete(2),
        action_space=gym.spaces.Discrete(2),
        max_trajectory_length=3,
    )
    t1 = self._make_trajectory(observations=[0, 1], actions=[1])
    [(x1, _, w1)] = env.trajectory_to_training_examples(t1)
    t2 = self._make_trajectory(observations=[0, 1], actions=[0])
    [(x2, _, w2)] = env.trajectory_to_training_examples(t2)

    np.testing.assert_array_equal(x1 * w1, x2 * w2)


if __name__ == "__main__":
  test.main()
