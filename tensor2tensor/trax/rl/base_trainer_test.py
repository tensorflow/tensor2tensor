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

"""Tests for tensor2tensor.trax.rl.base_trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cloudpickle as pickle
import numpy as np

from tensor2tensor.envs import gym_env_problem
from tensor2tensor.trax.rl import base_trainer
from tensorflow import test


class FakeTrainer(base_trainer.BaseTrainer):
  """Fake Trainer.

  Adds one complete and one incomplete trajectory every epoch.
  """

  def __init__(self, *args, **kwargs):
    super(FakeTrainer, self).__init__(*args, **kwargs)
    self._epoch = 0
    self._should_reset = True

  @property
  def epoch(self):
    return self._epoch

  def train_epoch(self):
    trajectories = self.train_env.trajectories
    if self._should_reset:
      trajectories.reset(indices=np.arange(2), observations=np.zeros(2))
    self._should_reset = False
    trajectories.step(
        observations=np.zeros(2),
        raw_rewards=np.zeros(2),
        processed_rewards=np.zeros(2),
        dones=np.array([False, True]),
        actions=np.zeros(2),
    )
    # Reset the trajectories that are done, as
    # env_problem_utils.play_env_problem_with_policy does.
    trajectories.reset(indices=np.array([1]), observations=np.zeros(1))
    self._epoch += 1

  def evaluate(self):
    pass

  def save(self):
    pass

  def flush_summaries(self):
    pass


class BaseTrainerTest(test.TestCase):

  def _make_trainer(self, min_count_per_shard):
    train_env = gym_env_problem.GymEnvProblem(
        base_env_name="Acrobot-v1", batch_size=2)
    eval_env = gym_env_problem.GymEnvProblem(
        base_env_name="Acrobot-v1", batch_size=1)
    temp_dir = self.get_temp_dir()
    return FakeTrainer(
        train_env, eval_env,
        output_dir=temp_dir,
        trajectory_dump_dir=temp_dir,
        trajectory_dump_min_count_per_shard=min_count_per_shard,
    )

  def _assert_no_shard_exists(self, trajectory_dir):
    self.assertFalse(os.listdir(trajectory_dir))

  def _assert_single_shard_exists_and_has_trajectories(
      self, trajectory_dir, expected_trajectory_lengths):
    shard_filenames = os.listdir(trajectory_dir)
    self.assertEqual(len(shard_filenames), 1)
    shard_path = os.path.join(trajectory_dir, shard_filenames[0])
    with open(shard_path, "rb") as f:
      trajectories = pickle.load(f)
    actual_trajectory_lengths = [
        len(trajectory.time_steps) for trajectory in trajectories]
    self.assertEqual(
        list(sorted(actual_trajectory_lengths)),
        list(sorted(expected_trajectory_lengths)),
    )

  def test_dumps_full_shard(self):
    trainer = self._make_trainer(min_count_per_shard=2)
    trajectory_dir = self.get_temp_dir()

    # Add one complete trajectory to the buffer. Should not dump yet.
    trainer.train_epoch()
    trainer.dump_trajectories()
    self._assert_no_shard_exists(trajectory_dir)

    # Add the second complete trajectory. Now we should dump.
    trainer.train_epoch()
    trainer.dump_trajectories()
    self._assert_single_shard_exists_and_has_trajectories(
        trajectory_dir, [2, 2])

  def test_dumps_incomplete_trajectories_when_force_is_true(self):
    trainer = self._make_trainer(min_count_per_shard=2)
    trajectory_dir = self.get_temp_dir()

    # Add one complete and one incomplete trajectory to the buffer. Should dump.
    trainer.train_epoch()
    trainer.dump_trajectories(force=True)
    self._assert_single_shard_exists_and_has_trajectories(
        trajectory_dir, [2, 2])

  def test_dumps_incomplete_shard_when_force_is_true(self):
    trainer = self._make_trainer(min_count_per_shard=4)
    trajectory_dir = self.get_temp_dir()

    # Add one complete and one incomplete trajectory to the buffer. Should dump,
    # even though we don't have a full shard yet.
    trainer.train_epoch()
    trainer.dump_trajectories(force=True)
    self._assert_single_shard_exists_and_has_trajectories(
        trajectory_dir, [2, 2])


if __name__ == "__main__":
  test.main()
