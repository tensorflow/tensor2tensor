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

"""Tests for tensor2tensor.envs.trajectory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.envs import trajectory
import tensorflow as tf


class TrajectoryTest(tf.test.TestCase):

  def test_empty_trajectory(self):
    t = trajectory.Trajectory()
    self.assertFalse(t.is_active)
    self.assertEqual(0, t.num_time_steps)
    self.assertFalse(t.done)

  def test_add_time_step(self):
    t = trajectory.Trajectory()
    t.add_time_step(observation=1, done=True)

    # Test that the trajectory is now active.
    self.assertTrue(t.is_active)

    added_t = t.last_time_step
    self.assertEqual(1, added_t.observation)
    self.assertTrue(added_t.done)
    self.assertIsNone(None, added_t.raw_reward)
    self.assertIsNone(None, added_t.processed_reward)
    self.assertIsNone(None, added_t.action)

    self.assertEqual(1, t.num_time_steps)

  def test_change_last_time_step(self):
    t = trajectory.Trajectory()
    t.add_time_step(observation=1, done=False)
    t.add_time_step(observation=1, done=True)
    self.assertTrue(t.is_active)

    num_ts_old = t.num_time_steps
    self.assertEqual(2, num_ts_old)

    # Assert on what the last time-step is currently.
    ts = t.last_time_step
    self.assertEqual(1, ts.observation)
    self.assertTrue(ts.done)
    self.assertEqual(None, ts.action)

    # Change the last time-step.
    t.change_last_time_step(done=False, action=5)

    # Assert that it changed.
    ts = t.last_time_step
    self.assertEqual(1, ts.observation)  # unchanged, since we didn't change it.
    self.assertFalse(ts.done)  # was True earlier
    self.assertEqual(5, ts.action)  # was None earlier

    # Assert on the number of steps remaining the same as before.
    self.assertEqual(num_ts_old, t.num_time_steps)

  def test_reward(self):
    t = trajectory.Trajectory()
    # first time-step doesn't have rewards, since they are on entering a state.
    t.add_time_step(
        observation=1, raw_reward=None, processed_reward=None, done=False)
    t.add_time_step(
        observation=2, raw_reward=2, processed_reward=200, done=False)
    t.add_time_step(
        observation=3, raw_reward=3, processed_reward=300, done=True)

    raw_reward, processed_reward = t.reward

    self.assertEqual(5, raw_reward)
    self.assertEqual(500, processed_reward)


class BatchTrajectoryTest(tf.test.TestCase):

  BATCH_SIZE = 10
  OBSERVATION_SHAPE = (3, 4)

  def get_random_observations_rewards_actions_dones(self, batch_size=None):
    batch_size = batch_size or self.BATCH_SIZE
    # Random observations, rewards, actions, done of the expected shape.
    observations = np.random.rand(*((batch_size,) + self.OBSERVATION_SHAPE))
    raw_rewards = np.random.randn(batch_size)
    actions = np.random.randn(batch_size)
    # 40% change of being done.
    dones = np.random.random((batch_size,)) > 0.6

    return observations, raw_rewards, actions, dones

  def test_creation(self):
    bt = trajectory.BatchTrajectory(batch_size=self.BATCH_SIZE)

    self.assertEqual(self.BATCH_SIZE, len(bt.trajectories))
    self.assertEqual(0, len(bt.completed_trajectories))

  def test_reset_all(self):
    bt = trajectory.BatchTrajectory(batch_size=self.BATCH_SIZE)

    indices = np.arange(self.BATCH_SIZE)
    observations, _, _, _ = self.get_random_observations_rewards_actions_dones()

    # Call reset.
    bt.reset(indices, observations)

    # Assert that all trajectories are active and not done (reset never marks
    # anything as done).
    self.assertTrue(all(t.is_active for t in bt.trajectories))
    self.assertEqual(0, len(bt.completed_trajectories))

  def test_num_time_steps(self):
    bt = trajectory.BatchTrajectory(batch_size=self.BATCH_SIZE)

    self.assertEqual(0, bt.num_completed_time_steps)
    self.assertEqual(0, bt.num_time_steps)

  def test_reset_some(self):
    bt = trajectory.BatchTrajectory(batch_size=self.BATCH_SIZE)

    indices = np.arange(self.BATCH_SIZE // 2)
    observations, _, _, _ = self.get_random_observations_rewards_actions_dones(
        batch_size=self.BATCH_SIZE // 2)

    # Just reset the first half.
    bt.reset(indices, observations)

    # So first half are active, rest aren't.
    self.assertTrue(
        all(t.is_active for t in bt.trajectories[:self.BATCH_SIZE // 2]))
    self.assertTrue(
        all(not t.is_active for t in bt.trajectories[self.BATCH_SIZE // 2:]))

    # Nothing is done anyways.
    self.assertEqual(0, len(bt.completed_trajectories))

  def test_step(self):
    bt = trajectory.BatchTrajectory(batch_size=self.BATCH_SIZE)

    indices = np.arange(self.BATCH_SIZE)
    observations, _, _, _ = self.get_random_observations_rewards_actions_dones()

    # Have to call reset first.
    bt.reset(indices, observations)

    # Create some fake data for calling step.
    new_observations, raw_rewards, actions, dones = (
        self.get_random_observations_rewards_actions_dones())
    processed_rewards = raw_rewards.astype(np.int64)

    # Force mark the first one as done anyways, so that there is something to
    # test.
    dones[0] = True

    num_done = sum(dones)
    self.assertLessEqual(1, num_done)  # i.e. num_done is atleast 1.

    num_not_done = len(dones) - num_done

    # Finally call step.
    bt.step(new_observations, raw_rewards, processed_rewards, dones, actions)

    # Expect to see `num_done` number of completed trajectories.
    self.assertEqual(num_done, len(bt.completed_trajectories))

    # Expect to see that the rest are marked as active.
    num_active = sum(t.is_active for t in bt.trajectories)
    self.assertEqual(num_not_done, num_active)

  def test_desired_placement_of_rewards_and_actions(self):
    batch_size = 1
    bt = trajectory.BatchTrajectory(batch_size=batch_size)

    indices = np.arange(batch_size)
    observations, _, _, _ = self.get_random_observations_rewards_actions_dones(
        batch_size=batch_size)

    # Have to call reset first.
    bt.reset(indices, observations)

    # Create some fake data for calling step.
    new_observations, raw_rewards, actions, _ = (
        self.get_random_observations_rewards_actions_dones(
            batch_size=batch_size))
    processed_rewards = raw_rewards.astype(np.int64)
    dones = np.full(batch_size, False)

    # Call step.
    bt.step(new_observations, raw_rewards, processed_rewards, dones, actions)

    # Assert that nothing is done, since dones is False
    self.assertEqual(0, len(bt.completed_trajectories))

    # The only trajectory is active.
    self.assertEqual(batch_size, len(bt.trajectories))
    t = bt.trajectories[0]
    self.assertTrue(t.is_active)
    self.assertEqual(2, t.num_time_steps)

    ts = t.time_steps

    # Now assert on placements

    # i.e. the old observation/done is first and the new one comes later.
    self.assertAllEqual(observations[0], ts[0].observation)
    self.assertAllEqual(new_observations[0], ts[1].observation)

    self.assertEqual(False, ts[0].done)
    self.assertEqual(False, ts[1].done)

    # Similarly actions went to the first time-step.
    self.assertEqual(actions[0], ts[0].action)
    self.assertIsNone(ts[1].action)

    # However make sure reward went into the second time-step and not the first.
    self.assertNear(raw_rewards[0], ts[1].raw_reward, 1e-6)
    self.assertIsNone(ts[0].raw_reward)

    # Similarly with processed_rewards.
    self.assertEqual(processed_rewards[0], ts[1].processed_reward)
    self.assertIsNone(ts[0].processed_reward)


if __name__ == '__main__':
  tf.test.main()
