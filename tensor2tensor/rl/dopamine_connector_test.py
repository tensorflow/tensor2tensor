# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for dopamine.atari.run_experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil



from absl import flags
from dopamine.atari import run_experiment
from dopamine.common import checkpointer
from dopamine.common import logger
import numpy as np
import mock
import tensorflow as tf

import gin.tf

from tensor2tensor.rl import dopamine_connector

FLAGS = flags.FLAGS


def _create_mock_checkpointer():
  mock_checkpointer = mock.Mock()
  test_dictionary = {'current_iteration': 1729,
                     'logs': 'logs'}
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
  """

  Observations and rewards are returned as arrays, done as single value.
  """
  # TODO: this can be used for mbrl pipeline, move it.
  def __init__(self, envs):
    self.env_batch = envs
    self.batch_size = len(self.env_batch)
    self.max_steps = self.env_batch[0].max_steps
    assert np.all(self.max_steps == env.max_steps for env in self.env_batch)

  def step(self, actions):
    print('actions', actions)
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
      self._test_cls.assertEqual('iteration_{:d}'.format(self._iter), key)
      self._test_cls.assertEqual('statistics', val)
      self._iter += 1
    self._calls_to_set += 1

  def log_to_file(self, filename_prefix, iteration_number):
    if self._run_asserts:
      self._test_cls.assertEqual(
          'prefix_{}'.format(self._iter - 1),
          '{}_{}'.format(filename_prefix, iteration_number))
    self._calls_to_log += 1


class RunExperimentTest(tf.test.TestCase):

  @mock.patch.object(gin, 'parse_config_files_and_bindings')
  def testLoadGinConfigs(self, mock_parse_config_files_and_bindings):
    gin_files = ['file1', 'file2', 'file3']
    gin_bindings = ['binding1', 'binding2']
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    self.assertEqual(1, mock_parse_config_files_and_bindings.call_count)
    mock_args, mock_kwargs = mock_parse_config_files_and_bindings.call_args
    self.assertEqual(gin_files, mock_args[0])
    self.assertEqual(gin_bindings, mock_kwargs['bindings'])
    self.assertFalse(mock_kwargs['skip_unknown'])



class BatchedRunnerTest(tf.test.TestCase):

  def _agent_step(self, rewards, observations):
    # We verify that rewards are clipped (and set by MockEnvironment as a
    # function of observation)
    # observation = observations[0]
    # expected_rewards = [1 if observation % 2 else -1]
    # self.assertEqual(expected_reward, reward)
    print('observations', observations)
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

  # def testFailsWithoutGameName(self):
  #   with self.assertRaises(AssertionError):
  #     run_experiment.Runner(self._test_subdir, self._create_agent_fn)
  #
  # @mock.patch.object(checkpointer, 'get_latest_checkpoint_number')
  # def testInitializeCheckpointingWithNoCheckpointFile(self, mock_get_latest):
  #   mock_get_latest.return_value = -1
  #   base_dir = '/does/not/exist'
  #   with self.assertRaisesRegexp(tf.errors.PermissionDeniedError,
  #                                '.*/does.*'):
  #     run_experiment.Runner(base_dir, self._create_agent_fn,
  #                           game_name='Pong')
  #
  # @mock.patch.object(checkpointer, 'get_latest_checkpoint_number')
  # @mock.patch.object(checkpointer, 'Checkpointer')
  # @mock.patch.object(logger, 'Logger')
  # def testInitializeCheckpointingWhenCheckpointUnbundleFails(
  #     self, mock_logger_constructor, mock_checkpointer_constructor,
  #     mock_get_latest):
  #   mock_checkpointer = _create_mock_checkpointer()
  #   mock_checkpointer_constructor.return_value = mock_checkpointer
  #   latest_checkpoint = 7
  #   mock_get_latest.return_value = latest_checkpoint
  #   agent = mock.Mock()
  #   agent.unbundle.return_value = False
  #   mock_logger = mock.Mock()
  #   mock_logger_constructor.return_value = mock_logger
  #   runner = run_experiment.Runner(self._test_subdir,
  #                                  lambda x, y, summary_writer: agent,
  #                                  create_environment_fn=lambda x, y: x,
  #                                  game_name='Test')
  #   self.assertEqual(0, runner._start_iteration)
  #   self.assertEqual(1, mock_checkpointer.load_checkpoint.call_count)
  #   self.assertEqual(1, agent.unbundle.call_count)
  #   mock_args, _ = agent.unbundle.call_args
  #   self.assertEqual('{}/checkpoints'.format(self._test_subdir), mock_args[0])
  #   self.assertEqual(latest_checkpoint, mock_args[1])
  #   expected_dictionary = {'current_iteration': 1729,
  #                          'logs': 'logs'}
  #   self.assertDictEqual(expected_dictionary, mock_args[2])
  #
  # @mock.patch.object(checkpointer, 'get_latest_checkpoint_number')
  # def testInitializeCheckpointingWhenCheckpointUnbundleSucceeds(
  #     self, mock_get_latest):
  #   latest_checkpoint = 7
  #   mock_get_latest.return_value = latest_checkpoint
  #   logs_data = {'a': 1, 'b': 2}
  #   current_iteration = 1729
  #   checkpoint_data = {'current_iteration': current_iteration,
  #                      'logs': logs_data}
  #   checkpoint_dir = os.path.join(self._test_subdir, 'checkpoints')
  #   checkpoint = checkpointer.Checkpointer(checkpoint_dir, 'ckpt')
  #   checkpoint.save_checkpoint(latest_checkpoint, checkpoint_data)
  #   mock_agent = mock.Mock()
  #   mock_agent.unbundle.return_value = True
  #   runner = run_experiment.Runner(self._test_subdir,
  #                                  lambda x, y, summary_writer: mock_agent,
  #                                  game_name='Pong')
  #   expected_iteration = current_iteration + 1
  #   self.assertEqual(expected_iteration, runner._start_iteration)
  #   self.assertDictEqual(logs_data, runner._logger.data)
  #   mock_agent.unbundle.assert_called_once_with(
  #       checkpoint_dir, latest_checkpoint, checkpoint_data)

  def testRunEpisodeBatch(self):
    max_steps_per_episode = 11
    batch_size = self.batch_size
    reward_multipliers = [-1, 1] * int(batch_size / 2)
    envs = [MockEnvironment(reward_multiplier=rm) for rm in reward_multipliers]
    environment = BatchEnv(envs)
    runner = dopamine_connector.BatchRunner(
        self._test_subdir, self._create_agent_fn, batch_size=batch_size,
        game_name='Test',
        create_environment_fn=lambda x, y: environment,
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
        self._test_subdir, self._create_agent_fn, batch_size=batch_size,
        game_name='Test',
        create_environment_fn=lambda x, y: environment,
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
      self._test_subdir, self._create_agent_fn, batch_size=batch_size,
      game_name="Test",
      create_environment_fn=lambda x, y: environment)

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
    for i in range(len(statistics)):
      self.assertDictEqual(expected_statistics[i], statistics[i])

  def testRunOneIteration(self):
    environment_steps = 2
    batch_size = self.batch_size
    envs = [MockEnvironment(max_steps=environment_steps)
            for _ in range(batch_size)]

    environment = BatchEnv(envs)

    training_steps = 20 * batch_size
    evaluation_steps = 10 * batch_size

    runner = dopamine_connector.BatchRunner(
      self._test_subdir, self._create_agent_fn, batch_size=batch_size,
      game_name="Test",
      create_environment_fn=lambda x, y: environment,
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
    #TODO: We probably do we need this test, dopamine test for Runner is enugh
    # here. Remove this?
    log_every_n = 2
    logging_file_prefix = "prefix"
    statistics = "statistics"
    experiment_logger = MockLogger(test_cls=self)
    mock_logger_constructor.return_value = experiment_logger
    runner = dopamine_connector.BatchRunner(
        self._test_subdir, self._create_agent_fn, batch_size=self.batch_size,
        game_name="Test",
        create_environment_fn=lambda x, y: mock.Mock(),
        logging_file_prefix=logging_file_prefix,
        log_every_n=log_every_n)
    num_iterations = 10
    for i in range(num_iterations):
      runner._log_experiment(i, statistics)
    self.assertEqual(num_iterations, experiment_logger._calls_to_set)
    self.assertEqual((num_iterations / log_every_n),
                     experiment_logger._calls_to_log)

  # @mock.patch.object(checkpointer, 'Checkpointer')
  # @mock.patch.object(logger, 'Logger')
  # def testCheckpointExperiment(self, mock_logger_constructor,
  #                              mock_checkpointer_constructor):
  #   checkpoint_dir = os.path.join(self._test_subdir, 'checkpoints')
  #   test_dict = {'test': 1}
  #   iteration = 1729
  #
  #   def bundle_and_checkpoint(x, y):
  #     self.assertEqual(checkpoint_dir, x)
  #     self.assertEqual(iteration, y)
  #     return test_dict
  #
  #   self._agent.bundle_and_checkpoint.side_effect = bundle_and_checkpoint
  #   experiment_checkpointer = mock.Mock()
  #   mock_checkpointer_constructor.return_value = experiment_checkpointer
  #   logs_data = {'one': 1, 'two': 2}
  #   mock_logger = MockLogger(run_asserts=False, data=logs_data)
  #   mock_logger_constructor.return_value = mock_logger
  #   runner = run_experiment.Runner(
  #       self._test_subdir, self._create_agent_fn,
  #       game_name='Test',
  #       create_environment_fn=lambda x, y: mock.Mock())
  #   runner._checkpoint_experiment(iteration)
  #   self.assertEqual(1, experiment_checkpointer.save_checkpoint.call_count)
  #   mock_args, _ = experiment_checkpointer.save_checkpoint.call_args
  #   self.assertEqual(iteration, mock_args[0])
  #   test_dict['logs'] = logs_data
  #   test_dict['current_iteration'] = iteration
  #   self.assertDictEqual(test_dict, mock_args[1])
  #
  # @mock.patch.object(checkpointer, 'Checkpointer')
  # @mock.patch.object(logger, 'Logger')
  # def testRunExperimentWithInconsistentRange(self, mock_logger_constructor,
  #                                            mock_checkpointer_constructor):
  #   experiment_logger = MockLogger()
  #   mock_logger_constructor.return_value = experiment_logger
  #   experiment_checkpointer = mock.Mock()
  #   mock_checkpointer_constructor.return_value = experiment_checkpointer
  #   runner = run_experiment.Runner(
  #       self._test_subdir, self._create_agent_fn,
  #       game_name='Test',
  #       create_environment_fn=lambda x, y: mock.Mock(),
  #       num_iterations=0)
  #   runner.run_experiment()
  #   self.assertEqual(0, experiment_checkpointer.save_checkpoint.call_count)
  #   self.assertEqual(0, experiment_logger._calls_to_set)
  #   self.assertEqual(0, experiment_logger._calls_to_log)
  #
  # @mock.patch.object(checkpointer, 'get_latest_checkpoint_number')
  # @mock.patch.object(checkpointer, 'Checkpointer')
  # @mock.patch.object(logger, 'Logger')
  # def testRunExperiment(self, mock_logger_constructor,
  #                       mock_checkpointer_constructor,
  #                       mock_get_latest):
  #   log_every_n = 1
  #   environment = MockEnvironment()
  #   experiment_logger = MockLogger(run_asserts=False)
  #   mock_logger_constructor.return_value = experiment_logger
  #   experiment_checkpointer = mock.Mock()
  #   start_iteration = 1729
  #   mock_get_latest.return_value = start_iteration
  #   def load_checkpoint(_):
  #     return {'logs': 'log_data', 'current_iteration': start_iteration - 1}
  #
  #   experiment_checkpointer.load_checkpoint.side_effect = load_checkpoint
  #   mock_checkpointer_constructor.return_value = experiment_checkpointer
  #   def bundle_and_checkpoint(x, y):
  #     del x, y  # Unused.
  #     return {'test': 1}
  #
  #   self._agent.bundle_and_checkpoint.side_effect = bundle_and_checkpoint
  #   num_iterations = 10
  #   self._agent.unbundle.return_value = True
  #   end_iteration = start_iteration + num_iterations
  #   runner = run_experiment.Runner(
  #       self._test_subdir, self._create_agent_fn,
  #       game_name='Test',
  #       create_environment_fn=lambda x, y: environment,
  #       log_every_n=log_every_n,
  #       num_iterations=end_iteration,
  #       training_steps=1,
  #       evaluation_steps=1)
  #   self.assertEqual(start_iteration, runner._start_iteration)
  #   runner.run_experiment()
  #   self.assertEqual(num_iterations,
  #                    experiment_checkpointer.save_checkpoint.call_count)
  #   self.assertEqual(num_iterations, experiment_logger._calls_to_set)
  #   self.assertEqual(num_iterations, experiment_logger._calls_to_log)
  #   glob_string = '{}/events.out.tfevents.*'.format(self._test_subdir)
  #   self.assertGreater(len(tf.gfile.Glob(glob_string)), 0)


if __name__ == '__main__':
  tf.test.main()
