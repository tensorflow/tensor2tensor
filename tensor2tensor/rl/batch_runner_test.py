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

"""Tests for BatchRunner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl import flags
from dopamine.discrete_domains import logger
import mock
import numpy as np

from tensor2tensor.rl import dopamine_connector

import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS


def _create_mock_checkpointer():
  mock_checkpointer = mock.Mock()
  test_dictionary = {"current_iteration": 1729,
                     "logs": "logs"}
  mock_checkpointer.load_checkpoint.return_value = test_dictionary
  return mock_checkpointer


class MockEnvironment(object):
  """Mock environment for testing."""

  def __init__(self, max_steps=10, reward_multiplier=1):
    self._observation = 0
    self.max_steps = max_steps
    self.reward_multiplier = reward_multiplier
    self.game_over = False

  def reset(self):
    self._observation = 0
    return self._observation

  def step(self, action):
    self._observation += 1
    action_reward_multiplier = -1 if action > 0 else 1
    reward_multiplier = self.reward_multiplier * action_reward_multiplier
    reward = self._observation * reward_multiplier
    is_terminal = self._observation >= self.max_steps
    self.game_over = is_terminal

    unused = 0
    return (self._observation, reward, is_terminal, unused)

  def render(self, mode):
    pass


class BatchEnv(object):
  """Batch env.

  Batch of environments. Assumes that all throws "done" on the same step.

  Observations and rewards are returned as arrays, done as single value.
  """

  # TODO(kozak): this can be used for mbrl pipeline (for both simulated and
  # real env), move it to dopamine_connector.py (rename it?)
  def __init__(self, envs):
    self.env_batch = envs
    self.batch_size = len(self.env_batch)
    self.max_steps = self.env_batch[0].max_steps
    assert np.all(self.max_steps == env.max_steps for env in self.env_batch)

  def step(self, actions):
    ret = [env.step(action) for env, action in zip(self.env_batch, actions)]
    obs, rewards, dones, infos = [np.array(r) for r in zip(*ret)]
    done = dones[0]
    assert np.all(done == dones)
    self.game_over = done
    return obs, rewards, done, infos

  def reset(self):
    return np.array([env.reset() for env in self.env_batch])

  def render(self, mode):
    pass


class MockLogger(object):
  """Class to mock the experiment logger."""

  def __init__(self, test_cls=None, run_asserts=True, data=None):
    self._test_cls = test_cls
    self._run_asserts = run_asserts
    self._iter = 0
    self._calls_to_set = 0
    self._calls_to_log = 0
    self.data = data

  def __setitem__(self, key, val):
    if self._run_asserts:
      self._test_cls.assertEqual("iteration_{:d}".format(self._iter), key)
      self._test_cls.assertEqual("statistics", val)
      self._iter += 1
    self._calls_to_set += 1

  def log_to_file(self, filename_prefix, iteration_number):
    if self._run_asserts:
      self._test_cls.assertEqual(
          "prefix_{}".format(self._iter - 1),
          "{}_{}".format(filename_prefix, iteration_number))
    self._calls_to_log += 1


class BatchedRunnerTest(tf.test.TestCase):
  """Modified tests from dopamine run_experiment_test.py."""

  # TODO(kozak): decide if we want to use and modify more tests from
  # dopamine/tests/atari/run_experiment_test.py (e.g.  testRunExperiment.py)

  def _agent_step(self, rewards, observations):
    # We verify that rewards are clipped (and set by MockEnvironment as a
    # function of observation)
    # observation = observations[0]
    # expected_rewards = [1 if observation % 2 else -1]
    # self.assertEqual(expected_reward, reward)
    actions = [ob % 2 for ob in observations]
    return actions

  def prepare_mock_agent(self, batch_size):
    assert batch_size % 2 == 0, "Some of tests assume that batch_size % 2 == 0"
    self.batch_size = batch_size
    self._agent = mock.Mock()
    self._agent.begin_episode.side_effect = \
      lambda x: np.repeat(0, self.batch_size)
    self._agent.step.side_effect = self._agent_step
    self._create_agent_fn = lambda x, y, summary_writer: self._agent

  def setUp(self):
    super(BatchedRunnerTest, self).setUp()
    self._test_subdir = "/tmp/dopamine_tests"
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)
    self.prepare_mock_agent(batch_size=4)

  def testRunEpisodeBatch(self):
    max_steps_per_episode = 11
    batch_size = self.batch_size
    reward_multipliers = [-1, 1] * int(batch_size / 2)
    envs = [MockEnvironment(reward_multiplier=rm) for rm in reward_multipliers]
    environment = BatchEnv(envs)
    runner = dopamine_connector.BatchRunner(
        self._test_subdir, self._create_agent_fn,
        create_environment_fn=lambda: environment,
        max_steps_per_episode=max_steps_per_episode)
    step_number, total_rewards = runner._run_one_episode()

    self.assertEqual(self._agent.step.call_count, environment.max_steps - 1)
    self.assertEqual(self._agent.end_episode.call_count, 1)
    self.assertEqual(environment.max_steps, step_number / batch_size)
    # Expected reward will be \sum_{i=0}^{9} (-1)**i * i = -5 when reward
    # multiplier=1
    self.assertAllEqual(np.array(reward_multipliers) * -5, total_rewards)

  def testRunOneEpisodeWithLowMaxSteps(self):
    max_steps_per_episode = 2
    batch_size = self.batch_size
    reward_multipliers = [-1, 1] * int(batch_size / 2)
    envs = [MockEnvironment(reward_multiplier=rm) for rm in reward_multipliers]
    environment = BatchEnv(envs)
    runner = dopamine_connector.BatchRunner(
        self._test_subdir, self._create_agent_fn,
        create_environment_fn=lambda: environment,
        max_steps_per_episode=max_steps_per_episode)
    step_number, total_rewards = runner._run_one_episode()

    self.assertEqual(self._agent.step.call_count, max_steps_per_episode - 1)
    self.assertEqual(self._agent.end_episode.call_count, 1)
    self.assertEqual(max_steps_per_episode, step_number / batch_size)
    self.assertAllEqual(np.array(reward_multipliers) * -1, total_rewards)

  def testRunOnePhase(self):
    batch_size = self.batch_size
    environment_steps = 2
    max_steps = environment_steps * batch_size * 10

    envs = [MockEnvironment(max_steps=environment_steps)
            for _ in range(batch_size)]

    environment = BatchEnv(envs)
    runner = dopamine_connector.BatchRunner(
        self._test_subdir, self._create_agent_fn,
        create_environment_fn=lambda: environment)

    statistics = []

    step_number, sum_returns, num_episodes = runner._run_one_phase(
        max_steps, statistics, "test")
    calls_to_run_episode = int(max_steps / (environment_steps * batch_size))
    self.assertEqual(self._agent.step.call_count, calls_to_run_episode)
    self.assertEqual(self._agent.end_episode.call_count, calls_to_run_episode)
    self.assertEqual(max_steps, step_number)
    self.assertEqual(-1 * calls_to_run_episode * batch_size, sum_returns)
    self.assertEqual(calls_to_run_episode, num_episodes / batch_size)
    expected_statistics = []
    for _ in range(calls_to_run_episode * batch_size):
      expected_statistics.append({
          "test_episode_lengths": 2,
          "test_episode_returns": -1
      })
    self.assertEqual(len(expected_statistics), len(statistics))
    for expected_stats, stats in zip(expected_statistics, statistics):
      self.assertDictEqual(expected_stats, stats)

  def testRunOneIteration(self):
    environment_steps = 2
    batch_size = self.batch_size
    envs = [MockEnvironment(max_steps=environment_steps)
            for _ in range(batch_size)]

    environment = BatchEnv(envs)

    training_steps = 20 * batch_size
    evaluation_steps = 10 * batch_size

    runner = dopamine_connector.BatchRunner(
        self._test_subdir, self._create_agent_fn,
        create_environment_fn=lambda: environment,
        training_steps=training_steps, evaluation_steps=evaluation_steps
    )

    dictionary = runner._run_one_iteration(1)
    train_rollouts = int(training_steps / environment_steps)
    eval_rollouts = int(evaluation_steps / environment_steps)
    expected_dictionary = {
        "train_episode_lengths": [2 for _ in range(train_rollouts)],
        "train_episode_returns": [-1 for _ in range(train_rollouts)],
        "train_average_return": [-1],
        "eval_episode_lengths": [2 for _ in range(eval_rollouts)],
        "eval_episode_returns": [-1 for _ in range(eval_rollouts)],
        "eval_average_return": [-1]
    }
    self.assertDictEqual(expected_dictionary, dictionary)

  @mock.patch.object(logger, "Logger")
  def testLogExperiment(self, mock_logger_constructor):
    # TODO(kozak): We probably do not need this test, dopamine test
    # for Runner is enough here. Remove this?
    log_every_n = 2
    logging_file_prefix = "prefix"
    statistics = "statistics"
    experiment_logger = MockLogger(test_cls=self)
    mock_logger_constructor.return_value = experiment_logger
    runner = dopamine_connector.BatchRunner(
        self._test_subdir, self._create_agent_fn,
        create_environment_fn=mock.Mock,
        logging_file_prefix=logging_file_prefix,
        log_every_n=log_every_n)
    num_iterations = 10
    for i in range(num_iterations):
      runner._log_experiment(i, statistics)
    self.assertEqual(num_iterations, experiment_logger._calls_to_set)
    self.assertEqual((num_iterations / log_every_n),
                     experiment_logger._calls_to_log)


if __name__ == "__main__":
  tf.test.main()
