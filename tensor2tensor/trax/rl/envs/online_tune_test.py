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

"""Tests for tensor2tensor.trax.rl.online_tune."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.trax import history as trax_history
from tensor2tensor.trax.rl.envs import online_tune
from tensorflow import test


class OnlineTuneTest(test.TestCase):

  def _append_metrics(self, h, metric, values):
    for (i, value) in enumerate(values):
      h.append(*metric, step=i, value=value)

  def test_retrieves_historical_metric_values(self):
    history = trax_history.History()
    self._append_metrics(history, ("train", "accuracy"), [0.1, 0.73])
    metric_values = online_tune.historical_metric_values(
        history, metric=("train", "accuracy"), observation_range=(0, 5))
    np.testing.assert_array_equal(metric_values, [0.1, 0.73])

  def test_clips_historical_metric_values(self):
    history = trax_history.History()
    self._append_metrics(history, ("train", "loss"), [-10, 10])
    metric_values = online_tune.historical_metric_values(
        history, metric=("train", "loss"), observation_range=(-1, 1))
    np.testing.assert_array_equal(metric_values, [-1, 1])

  def test_converts_history_to_observations_without_learning_rate(self):
    history = trax_history.History()
    self._append_metrics(history, ("train", "loss"), [3.0, 1.07])
    self._append_metrics(history, ("eval", "accuracy"), [0.12, 0.68])
    observations = online_tune.history_to_observations(
        history,
        metrics=(("eval", "accuracy"), ("train", "loss")),
        observation_range=(0, 5),
        include_lr=False,
    )
    np.testing.assert_array_equal(observations, [[0.12, 3.0], [0.68, 1.07]])

  def test_converts_history_to_observations_with_learning_rate(self):
    history = trax_history.History()
    self._append_metrics(
        history, ("train", "training/learning_rate"), [1e-3, 1e-4])
    observations = online_tune.history_to_observations(
        history,
        metrics=(),
        observation_range=(0, 5),
        include_lr=True,
    )
    self.assertEqual(observations.shape, (2, 1))
    ((log_lr_1,), (log_lr_2,)) = observations
    self.assertGreater(log_lr_1, log_lr_2)

  def test_clips_observations(self):
    history = trax_history.History()
    self._append_metrics(history, ("eval", "loss"), [-10, 10])
    observations = online_tune.history_to_observations(
        history,
        metrics=(("eval", "loss"),),
        observation_range=(-2, 2),
        include_lr=False,
    )
    np.testing.assert_array_equal(observations, [[-2], [2]])

  def test_calculates_new_learning_rate(self):
    history = trax_history.History()
    self._append_metrics(
        history, online_tune.LEARNING_RATE_METRIC, [1e-2, 1e-3])
    new_lr = online_tune.new_learning_rate(
        action=2,
        history=history,
        action_multipliers=(0.5, 1.0, 2.0),
        max_lr=1.0,
    )
    np.testing.assert_almost_equal(new_lr, 2e-3)

  def test_clips_new_learning_rate(self):
    history = trax_history.History()
    self._append_metrics(history, online_tune.LEARNING_RATE_METRIC, [1e-3])
    new_lr = online_tune.new_learning_rate(
        action=0,
        history=history,
        action_multipliers=(4.0, 1.0, 0.25),
        max_lr=3e-3,
    )
    np.testing.assert_almost_equal(new_lr, 3e-3)


if __name__ == "__main__":
  test.main()
