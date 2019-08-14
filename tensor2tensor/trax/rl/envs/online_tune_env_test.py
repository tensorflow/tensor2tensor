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

"""Tests for tensor2tensor.trax.rl.online_tune_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import models
from tensor2tensor.trax import trax
from tensor2tensor.trax.rl.envs import online_tune_env
from tensorflow import test
from tensorflow.io import gfile

HISTORY_MODE = "eval"
METRIC = "metrics/accuracy"


class MockTrainer(trax.Trainer):

  def __init__(self, metrics_to_report, *args, **kwargs):
    super(MockTrainer, self).__init__(*args, **kwargs)
    self.learning_rates = []
    self.init_metrics_to_report = metrics_to_report
    self.metrics_to_report = None

  def reset(self, output_dir):
    super(MockTrainer, self).reset(output_dir)
    # Copy the sequence to a list so we can modify it later.
    self.metrics_to_report = list(self.init_metrics_to_report)

  def train_epoch(self, epoch_steps, eval_steps):
    del epoch_steps
    self.learning_rates.append(self.learning_rate)
    self.evaluate(eval_steps)

  def evaluate(self, eval_steps):
    del eval_steps
    self.state.history.append(
        mode=HISTORY_MODE,
        metric=METRIC,
        step=self.step,
        value=self.metrics_to_report.pop(0))


class OnlineTuneTest(test.TestCase):

  @staticmethod
  def _create_env(output_dir, metrics_to_report=(0.0,), action_multipliers=()):
    return online_tune_env.OnlineTuneEnv(
        trainer_class=functools.partial(MockTrainer, metrics_to_report),
        model=functools.partial(
            models.MLP, n_hidden_layers=0, n_output_classes=1),
        inputs=functools.partial(
            trax_inputs.random_inputs,
            input_shape=(1, 1),
            input_dtype=np.float32,
            output_shape=(1, 1),
            output_dtype=np.float32),
        output_dir=output_dir,
        action_multipliers=action_multipliers,
        observation_metrics=[(HISTORY_MODE, METRIC)],
        reward_metric=(HISTORY_MODE, METRIC),
        train_steps=1,
        eval_steps=1,
        env_steps=(len(metrics_to_report) - 1))

  def test_communicates_with_trainer(self):
    action_multipliers = [0.8, 1.0, 1.25]
    metrics_to_report = [0.1, 0.5, 0.8, 0.9]
    actions_to_take = [0, 1, 2]
    expected_observations = np.expand_dims(metrics_to_report, axis=1)
    # Metric difference in consecutive timesteps.
    expected_rewards = [0.4, 0.3, 0.1]
    expected_dones = [False, False, True]
    expected_learning_rates = [0.0008, 0.0008, 0.001]

    env = self._create_env(
        output_dir=self.get_temp_dir(),
        metrics_to_report=metrics_to_report,
        action_multipliers=action_multipliers)
    actual_observations = [env.reset()]
    actual_rewards = []
    actual_dones = []
    for action in actions_to_take:
      (observation, reward, done, _) = env.step(action)
      actual_observations.append(observation)
      actual_rewards.append(reward)
      actual_dones.append(done)

    np.testing.assert_allclose(actual_observations, expected_observations)
    np.testing.assert_allclose(actual_rewards, expected_rewards)
    self.assertEqual(actual_dones, expected_dones)
    np.testing.assert_allclose(env.trainer.learning_rates,
                               expected_learning_rates)

  def test_creates_new_trajectory_dirs(self):
    output_dir = self.get_temp_dir()
    env = self._create_env(output_dir=output_dir)
    self.assertEqual(set(gfile.listdir(output_dir)), set())
    env.reset()
    self.assertEqual(set(gfile.listdir(output_dir)), {"0"})
    env.reset()
    self.assertEqual(set(gfile.listdir(output_dir)), {"0", "1"})


if __name__ == "__main__":
  test.main()
