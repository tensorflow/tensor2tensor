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

"""Tests for tensor2tensor.trax.online_tune."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.trax import history as trax_history
from tensor2tensor.trax.rl import online_tune
from tensorflow import test


class OnlineTuneTest(test.TestCase):

  def _append_metrics(self, h, metric, values):
    for (i, value) in enumerate(values):
      h.append(*metric, step=i, value=value)

  def test_retrieves_historical_metric_values(self):
    history = trax_history.History()
    self._append_metrics(history, ("train", "accuracy"), [0.1, 0.73])
    metric_values = online_tune.historical_metric_values(
        history, metric=("train", "accuracy")
    )
    np.testing.assert_array_equal(metric_values, [0.1, 0.73])

  def test_converts_control_to_log_scale_without_flipping(self):
    config = ("weight_decay", None, (1e-5, 0.1), False)
    controls = np.array([0.01, 0.02, 0.04])
    obs_range = (-1, 1)
    obs = online_tune.control_to_observation(controls, config, obs_range)
    np.testing.assert_almost_equal(obs[1] - obs[0], obs[2] - obs[1])

  def test_converts_control_to_log_scale_with_flipping(self):
    config = ("momentum", None, (0.5, 0.99), True)
    controls = np.array([0.98, 0.96, 0.92])
    obs_range = (-1, 1)
    obs = online_tune.control_to_observation(controls, config, obs_range)
    np.testing.assert_almost_equal(obs[1] - obs[0], obs[2] - obs[1])

  def test_clips_control_without_flipping(self):
    config = ("weight_decay", None, (1e-5, 0.1), False)
    controls = np.array([0.0, 0.2])
    obs_range = (-1, 1)
    obs = online_tune.control_to_observation(controls, config, obs_range)
    np.testing.assert_equal(obs, [-1, 1])

  def test_clips_control_with_flipping(self):
    config = ("momentum", None, (0.5, 0.99), True)
    controls = np.array([0.4, 1.0])
    obs_range = (-1, 1)
    obs = online_tune.control_to_observation(controls, config, obs_range)
    np.testing.assert_equal(obs, [1, -1])

  def test_rescales_control(self):
    config = ("weight_decay", None, (1e-5, 0.1), False)
    controls = np.array([4e-4, 3e-3, 2e-2])
    (obs_low, obs_high) = (103, 104)
    obs = online_tune.control_to_observation(
        controls, config, observation_range=(obs_low, obs_high),
    )
    np.testing.assert_array_less(obs, [obs_high] * 3)
    np.testing.assert_array_less([obs_low] * 3, obs)

  def test_converts_history_to_observations_without_controls(self):
    history = trax_history.History()
    self._append_metrics(history, ("train", "loss"), [1.0, 0.07])
    self._append_metrics(history, ("eval", "accuracy"), [0.12, 0.68])
    observations = online_tune.history_to_observations(
        history,
        metrics=(("eval", "accuracy"), ("train", "loss")),
        observation_range=(-1, 1),
        control_configs=None,
    )
    np.testing.assert_array_almost_equal(
        observations, [[0.12, 1.0], [0.68, 0.07]]
    )

  def test_converts_history_to_observations_with_controls(self):
    history = trax_history.History()
    self._append_metrics(
        history, ("train", "training/learning_rate"), [1e-3, 1e-4])
    observations = online_tune.history_to_observations(
        history,
        metrics=(),
        observation_range=(0, 5),
        control_configs=(
            ("learning_rate", None, (1e-9, 10.0), False),
        ),
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
        control_configs=None,
    )
    np.testing.assert_array_equal(observations, [[-2], [2]])

  def test_updates_control_without_flipping(self):
    config = ("learning_rate", None, (1e-9, 10.0), False)
    history = trax_history.History()
    self._append_metrics(
        history, online_tune.control_metric("learning_rate"), [1e-2, 1e-3])
    new_control = online_tune.update_control(
        control_config=config,
        action=2,
        history=history,
        action_multipliers=(0.5, 1.0, 2.0),
    )
    np.testing.assert_almost_equal(new_control, 2e-3)

  def test_updates_control_with_flipping(self):
    config = ("momentum", None, (0.5, 0.99), True)
    history = trax_history.History()
    self._append_metrics(
        history, online_tune.control_metric("momentum"), [0.96, 0.98])
    new_control = online_tune.update_control(
        control_config=config,
        action=0,
        history=history,
        action_multipliers=(0.5, 1.0, 2.0),
    )
    np.testing.assert_almost_equal(new_control, 0.99)

  def test_clips_updated_control_without_flipping(self):
    config = ("learning_rate", None, (1e-9, 10.0), False)
    history = trax_history.History()
    self._append_metrics(
        history, online_tune.control_metric("learning_rate"), [7.0])
    new_control = online_tune.update_control(
        control_config=config,
        action=2,
        history=history,
        action_multipliers=(0.5, 1.0, 2.0),
    )
    np.testing.assert_almost_equal(new_control, 10.0)

  def test_clips_updated_control_with_flipping(self):
    config = ("momentum", None, (0.5, 0.99), True)
    history = trax_history.History()
    self._append_metrics(
        history, online_tune.control_metric("momentum"), [0.985])
    new_control = online_tune.update_control(
        control_config=config,
        action=0,
        history=history,
        action_multipliers=(0.5, 1.0, 2.0),
    )
    np.testing.assert_almost_equal(new_control, 0.99)


if __name__ == "__main__":
  test.main()
