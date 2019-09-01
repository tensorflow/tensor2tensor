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

import cloudpickle as pickle
import numpy as np
from tensor2tensor.envs import trajectory
from tensor2tensor.trax.rl import simple
from tensorflow import test
from tensorflow.io import gfile


class SimpleTest(test.TestCase):

  def _make_singleton_trajectory(self, observation):
    t = trajectory.Trajectory()
    t.add_time_step(observation=observation)
    return t

  def _dump_trajectory_pickle(self, observations, path):
    trajectories = list(map(self._make_singleton_trajectory, observations))
    with gfile.GFile(path, "wb") as f:
      pickle.dump(trajectories, f)

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


if __name__ == "__main__":
  test.main()
