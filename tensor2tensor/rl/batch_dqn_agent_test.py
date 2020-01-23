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

"""Tests for BatchDQNAgent."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl import flags
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import atari_lib
import numpy as np

from tensor2tensor.rl import dopamine_connector

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class BatchDQNAgentTest(tf.test.TestCase):
  # TODO(kozak): add testStepTrain (and possibly other tests) from dopamine
  # dqn_agent_test.py

  def setUp(self):
    super(BatchDQNAgentTest, self).setUp()
    self._test_subdir = os.path.join('/tmp/dopamine_tests', 'ckpts')
    shutil.rmtree(self._test_subdir, ignore_errors=True)
    os.makedirs(self._test_subdir)
    self.num_actions = 4
    self.min_replay_history = 6
    self.update_period = 2
    self.target_update_period = 4
    self.epsilon_decay_period = 90
    self.epsilon_train = 0.05
    self.observation_shape = dqn_agent.NATURE_DQN_OBSERVATION_SHAPE
    self.stack_size = dqn_agent.NATURE_DQN_STACK_SIZE
    self.env_batch_size = 4

    self.zero_state = np.zeros(
        [self.env_batch_size, self.observation_shape[0],
         self.observation_shape[1], self.stack_size])

  def _create_test_agent(self, sess):
    stack_size = self.stack_size

    class MockDQNNetwork(tf.keras.Model):
      """The Keras network used in tests."""

      def __init__(self, num_actions, **kwargs):
        # This weights_initializer gives action 0 a higher weight, ensuring
        # that it gets picked by the argmax.
        super(MockDQNNetwork, self).__init__(**kwargs)
        weights_initializer = np.tile(
            np.arange(num_actions, 0, -1), (stack_size, 1))
        self.layer = tf.keras.layers.Dense(
            num_actions,
            kernel_initializer=tf.constant_initializer(weights_initializer),
            bias_initializer=tf.ones_initializer())

      def call(self, state):
        inputs = tf.constant(
            np.zeros((state.shape[0], stack_size)), dtype=tf.float32)
        return atari_lib.DQNNetworkType(self.layer((inputs)))

    agent = dopamine_connector.BatchDQNAgent(
        network=MockDQNNetwork,
        replay_capacity=100,
        buffer_batch_size=8,
        generates_trainable_dones=True,
        sess=sess,
        env_batch_size=self.env_batch_size,
        num_actions=self.num_actions,
        min_replay_history=self.min_replay_history,
        epsilon_fn=lambda w, x, y, z: 0.0,  # No exploration.
        update_period=self.update_period,
        target_update_period=self.target_update_period,
        epsilon_eval=0.0)  # No exploration during evaluation.
    # This ensures non-random action choices (since epsilon_eval = 0.0) and
    # skips the train_step.
    agent.eval_mode = True
    sess.run(tf.global_variables_initializer())
    return agent

  def testCreateAgentWithDefaults(self):
    # Verifies that we can create and train an agent with the default values.
    with tf.Session() as sess:
      agent = self._create_test_agent(sess)
      sess.run(tf.global_variables_initializer())
      observation = np.ones([84, 84, 1])
      agent.begin_episode([observation])
      agent.step(reward=[1], observation=[observation])
      agent.end_episode(reward=[1])

  def testBeginEpisode(self):
    """Test the functionality of agent.begin_episode.

    Specifically, the action returned and its effect on state.
    """
    with tf.Session() as sess:
      agent = self._create_test_agent(sess)
      # We fill up the state with 9s. On calling agent.begin_episode the state
      # should be reset to all 0s.
      agent.state_batch.fill(9)
      first_observation = np.ones(
          [self.env_batch_size, self.observation_shape[0],
           self.observation_shape[1], 1])
      self.assertTrue((agent.begin_episode(first_observation) == 0).all())
      # When the all-1s observation is received, it will be placed at the end of
      # the state.
      expected_state = self.zero_state
      expected_state[:, :, :, -1] = np.ones(
          [self.env_batch_size, self.observation_shape[0],
           self.observation_shape[1]])
      self.assertAllEqual(agent.state_batch, expected_state)
      self.assertAllEqual(agent._observation_batch, first_observation[..., 0])
      # No training happens in eval mode.
      self.assertEqual(agent.training_steps, 0)

      # This will now cause training to happen.
      agent.eval_mode = False
      # Having a low replay memory add_count will prevent any of the
      # train/prefetch/sync ops from being called.
      agent._replay.memory.add_count = 0
      second_observation = np.ones(
          [self.env_batch_size, self.observation_shape[0],
           self.observation_shape[1], 1]) * 2
      agent.begin_episode(second_observation)
      # The agent's state will be reset, so we will only be left with the all-2s
      # observation.
      expected_state[:, :, :, -1] = np.full(
          (self.env_batch_size, self.observation_shape[0],
           self.observation_shape[1]), 2
      )
      self.assertAllEqual(agent.state_batch, expected_state)
      self.assertAllEqual(agent._observation_batch,
                          second_observation[:, :, :, 0])
      # training_steps is incremented since we set eval_mode to False.
      self.assertEqual(agent.training_steps, 1)


if __name__ == '__main__':
  tf.test.main()
