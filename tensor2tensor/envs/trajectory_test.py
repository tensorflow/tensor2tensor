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

"""Tests for tensor2tensor.envs.trajectory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tensor2tensor.envs import time_step
from tensor2tensor.envs import trajectory
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.io import gfile


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

  def test_observation_np(self):
    t = trajectory.Trajectory()
    ts = 5
    shape = (3, 4)
    for _ in range(ts):
      t.add_time_step(observation=np.random.uniform(size=shape), done=False)

    self.assertEqual((ts,) + shape, t.observations_np.shape)

  def test_truncate_and_last_n_observations_np(self):
    t = trajectory.Trajectory()
    ts = 5
    shape = (3, 4)
    for _ in range(ts):
      t.add_time_step(observation=np.random.uniform(size=shape), done=False)

    original_obs = np.copy(t.observations_np)
    self.assertEqual((ts,) + shape, original_obs.shape)

    # Now let's just get the observations from the last 2 steps.
    num_to_keep = 2
    truncated_original_obs = original_obs[-num_to_keep:, ...]

    # Let's get the last `num_to_keep` observations
    last_n_observations_np = np.copy(t.last_n_observations_np(n=num_to_keep))

    # Now truncate the trajectory and get the same.
    _ = t.truncate(num_to_keep=num_to_keep)
    truncated_np = np.copy(t.observations_np)

    # These should be the expected length.
    self.assertEqual((2,) + shape, last_n_observations_np.shape)
    self.assertEqual((2,) + shape, truncated_np.shape)

    # Test the last `num_to_keep` are the same.
    self.assertAllEqual(truncated_np, truncated_original_obs)
    self.assertAllEqual(last_n_observations_np, truncated_original_obs)

  def test_as_numpy(self):
    t = trajectory.Trajectory()
    shape = (3, 4)

    # We'll have `ts` observations and `ts-1` actions and rewards.
    ts = 5
    num_actions = 6
    observations = np.random.uniform(size=(ts,) + shape)
    actions = np.random.choice(range(num_actions), size=(ts - 1,))
    rewards = np.random.choice([-1, 0, 1], size=(ts - 1,))
    squares = np.arange(ts - 1)**2
    cubes = np.arange(ts - 1)**3

    def get_info(i):
      return {"sq": squares[i], "cu": cubes[i]}

    # First time-step has no reward.
    t.add_time_step(
        observation=observations[0],
        done=False,
        action=actions[0],
        info=get_info(0))
    for i in range(1, ts - 1):
      t.add_time_step(
          observation=observations[i],
          done=False,
          raw_reward=rewards[i - 1],
          processed_reward=rewards[i - 1],
          action=actions[i],
          info=get_info(i))
    # Last time-step has no action.
    t.add_time_step(
        observation=observations[-1],
        done=False,
        raw_reward=rewards[-1],
        processed_reward=rewards[-1])

    traj_np = t.as_numpy

    self.assertAllEqual(observations, traj_np[0])
    self.assertAllEqual(actions, traj_np[1])
    self.assertAllEqual(rewards, traj_np[2])

    self.assertAllEqual(squares, traj_np[4]["sq"])
    self.assertAllEqual(cubes, traj_np[4]["cu"])


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
    self.assertEqual(0, bt.num_completed_trajectories)

  def test_reset_all(self):
    bt = trajectory.BatchTrajectory(batch_size=self.BATCH_SIZE)

    indices = np.arange(self.BATCH_SIZE)
    observations, _, _, _ = self.get_random_observations_rewards_actions_dones()

    # Call reset.
    bt.reset(indices, observations)

    # Assert that all trajectories are active and not done (reset never marks
    # anything as done).
    self.assertTrue(all(t.is_active for t in bt.trajectories))
    self.assertEqual(0, bt.num_completed_trajectories)

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
    self.assertEqual(0, bt.num_completed_trajectories)

  def test_truncate(self):
    batch_size = 1
    bt = trajectory.BatchTrajectory(batch_size=batch_size)

    indices = np.arange(batch_size)
    observations, _, _, _ = (
        self.get_random_observations_rewards_actions_dones(
            batch_size=batch_size))

    # Have to call reset first.
    bt.reset(indices, observations)

    # Take a few steps.
    ts = 5
    for _ in range(ts):
      (observations, rewards, actions,
       dones) = self.get_random_observations_rewards_actions_dones(
           batch_size=batch_size)
      dones[...] = False
      bt.step(observations, rewards, rewards, dones, actions)

    self.assertEqual(0, bt.num_completed_trajectories)

    num_to_keep = 2
    bt.truncate_trajectories(indices, num_to_keep=num_to_keep)

    self.assertEqual(batch_size, bt.num_completed_trajectories)

    # Assert they are all active.
    # Since the last `num_to_keep` observations were duplicated.
    self.assertTrue(all(t.is_active for t in bt.trajectories))

    orig_obs = bt.completed_trajectories[0].observations_np
    # + 1 because of the initial reset
    self.assertEqual(ts + 1, orig_obs.shape[0])

    trunc_obs = bt.trajectories[0].observations_np
    self.assertEqual(num_to_keep, trunc_obs.shape[0])
    self.assertEqual(num_to_keep, bt.trajectories[0].num_time_steps)

    # Test that the observations are the same.
    self.assertAllEqual(orig_obs[-num_to_keep:, ...], trunc_obs)

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
    self.assertEqual(num_done, bt.num_completed_trajectories)

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
    self.assertEqual(0, bt.num_completed_trajectories)

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

  def test_observations_np(self):
    bt = trajectory.BatchTrajectory(batch_size=self.BATCH_SIZE)
    indices = np.arange(self.BATCH_SIZE)
    observations, _, _, _ = self.get_random_observations_rewards_actions_dones()

    # Have to call reset first.
    bt.reset(indices, observations)

    # Number of time-steps now looks like the following:
    # (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    lengths = np.full((self.BATCH_SIZE,), 1)

    ts = 5
    for _ in range(ts):
      (observations, rewards, actions,
       dones) = self.get_random_observations_rewards_actions_dones()
      dones[...] = False
      bt.step(observations, rewards, rewards, dones, actions)

    # Number of time-steps now looks like the following:
    # (6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
    lengths = lengths + ts

    # Now let's mark the first two as done.
    observations, _, _, _ = self.get_random_observations_rewards_actions_dones(
        batch_size=2)
    bt.reset(np.array([0, 1]), observations)

    # Number of time-steps now looks like the following:
    # (1, 1, 6, 6, 6, 6, 6, 6, 6, 6)
    lengths[0] = lengths[1] = 1

    for _ in range(ts):
      (observations, rewards, actions,
       dones) = self.get_random_observations_rewards_actions_dones()
      dones[...] = False
      bt.step(observations, rewards, rewards, dones, actions)

    # Number of time-steps now looks like the following:
    # (6, 6, 11, 11, 11, 11, 11, 11, 11, 11)
    lengths = lengths + ts

    boundary = 20
    len_history_for_policy = 40

    padded_obs_np, padded_lengths = bt.observations_np(
        boundary=boundary, len_history_for_policy=len_history_for_policy)

    # The lengths are what we expect them to be.
    self.assertAllEqual(lengths, padded_lengths)

    # The padded_observations are the shape we expect them to be.
    self.assertEqual((self.BATCH_SIZE, boundary + 1) + self.OBSERVATION_SHAPE,
                     padded_obs_np.shape)

    # Let's now request the last n = [1, 2 * boundary) steps for the history.
    for len_history_for_policy in range(1, 2 * boundary):
      # The expected lengths will now be:
      truncated_lengths = [min(l, len_history_for_policy) for l in lengths]

      padded_obs_np, padded_lengths = bt.observations_np(
          boundary=boundary, len_history_for_policy=len_history_for_policy)

      self.assertAllEqual(truncated_lengths, padded_lengths)

      # This shouldn't change, since even if we request lengths > boundary + 1
      # there are no trajectories that long.
      self.assertEqual((self.BATCH_SIZE, boundary + 1) + self.OBSERVATION_SHAPE,
                       padded_obs_np.shape)

    # Let's do 10 more steps (to go on the other side of the boundary.
    ts = 10
    for _ in range(ts):
      (observations, rewards, actions,
       dones) = self.get_random_observations_rewards_actions_dones()
      dones[...] = False
      bt.step(observations, rewards, rewards, dones, actions)

    # Number of time-steps now looks like the following:
    # (16, 16, 21, 21, 21, 21, 21, 21, 21, 21)
    lengths = lengths + ts

    len_history_for_policy = 40
    padded_obs_np, padded_lengths = bt.observations_np(
        boundary=boundary, len_history_for_policy=len_history_for_policy)

    # The lengths are what we expect them to be.
    self.assertAllEqual(lengths, padded_lengths)

    # The padded_observations are the shape we expect them to be.
    self.assertEqual(
        (self.BATCH_SIZE, (2 * boundary) + 1) + self.OBSERVATION_SHAPE,
        padded_obs_np.shape)

    # Test that the padding is the only part that is all 0s.
    # NOTE: There is almost 0 probability that the random observation is all 0s.
    zero_obs = np.full(self.OBSERVATION_SHAPE, 0.)
    for b in range(self.BATCH_SIZE):
      # The first lengths[b] will be actual data, rest is 0s.
      for ts in range(lengths[b]):
        self.assertFalse(np.all(zero_obs == padded_obs_np[b][ts]))

      for ts in range(lengths[b], len(padded_obs_np[b])):
        self.assertAllEqual(zero_obs, padded_obs_np[b][ts])

  def test_parse_trajectory_file_name(self):
    self.assertEqual(
        (12, 13, 1.0, "abc"),
        trajectory.BatchTrajectory.parse_trajectory_file_name(
            "/tmp/trajectory_epoch_000012_env_id_000013_temperature_1.0_r_abc.pkl"
        ))

    self.assertIsNone(
        trajectory.BatchTrajectory.parse_trajectory_file_name(
            "/tmp/trajectory_epoch_000012_env_id_000013.pkl"))

  def test_load_from_directory(self):
    output_dir = self.get_temp_dir()

    epochs = [0, 1, 2]
    env_ids = [0, 1, 2]
    temperatures = [0.5, 1.0]
    random_strings = ["a", "b"]

    # Write some trajectories.
    # There are 3x3x2x2 (36) trajectories, and of them 3x2x2 (12) are done.
    for epoch in epochs:
      for env_id in env_ids:
        for temperature in temperatures:
          for random_string in random_strings:
            traj = trajectory.Trajectory(time_steps=[
                time_step.TimeStep(
                    observation=epoch,
                    done=(epoch == 0),
                    raw_reward=1.0,
                    processed_reward=1.0,
                    action=env_id,
                    info={})
            ])

            trajectory_file_name = trajectory.TRAJECTORY_FILE_FORMAT.format(
                epoch=epoch,
                env_id=env_id,
                temperature=temperature,
                r=random_string)

            with gfile.GFile(
                os.path.join(output_dir, trajectory_file_name), "w") as f:
              trajectory.get_pickle_module().dump(traj, f)

    # Load everything and check.
    bt = trajectory.BatchTrajectory.load_from_directory(output_dir)

    self.assertIsInstance(bt, trajectory.BatchTrajectory)
    self.assertEqual(36, bt.num_completed_trajectories)
    self.assertEqual(36, bt.batch_size)

    bt = trajectory.BatchTrajectory.load_from_directory(output_dir, epoch=0)
    self.assertEqual(12, bt.num_completed_trajectories)
    self.assertEqual(12, bt.batch_size)

    # Get 100 trajectories, but there aren't any.
    bt = trajectory.BatchTrajectory.load_from_directory(
        output_dir, epoch=0, n_trajectories=100, max_tries=0)
    self.assertIsNone(bt)

    bt = trajectory.BatchTrajectory.load_from_directory(
        output_dir, epoch=0, temperature=0.5)
    self.assertEqual(6, bt.num_completed_trajectories)
    self.assertEqual(6, bt.batch_size)

    bt = trajectory.BatchTrajectory.load_from_directory(output_dir, epoch=1)
    self.assertEqual(12, bt.num_completed_trajectories)
    self.assertEqual(12, bt.batch_size)

    # Constraints cannot be satisfied.
    bt = trajectory.BatchTrajectory.load_from_directory(
        output_dir, epoch=1, n_trajectories=100, up_sample=False, max_tries=0)
    self.assertIsNone(bt)

    # Constraints can be satisfied.
    bt = trajectory.BatchTrajectory.load_from_directory(
        output_dir, epoch=1, n_trajectories=100, up_sample=True, max_tries=0)
    self.assertEqual(100, bt.num_completed_trajectories)
    self.assertEqual(100, bt.batch_size)

    bt = trajectory.BatchTrajectory.load_from_directory(
        output_dir, epoch=1, n_trajectories=10)
    self.assertEqual(10, bt.num_completed_trajectories)
    self.assertEqual(10, bt.batch_size)

    gfile.rmtree(output_dir)


if __name__ == "__main__":
  tf.test.main()
