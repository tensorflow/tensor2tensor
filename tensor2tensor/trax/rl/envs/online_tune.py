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

"""Utility functions for OnlineTuneEnv."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


LEARNING_RATE_METRIC = ("train", "training/learning_rate")


def historical_metric_values(
    history, metric, observation_range=(-np.inf, np.inf)):
  """Converts a metric stream from a trax History object into a numpy array."""
  metric_sequence = history.get(*metric)
  metric_values = np.array([
      metric_value for (_, metric_value) in metric_sequence
  ])
  return np.clip(metric_values, *observation_range)


def history_to_observations(history, metrics, observation_range, include_lr):
  """Converts a trax History object into a sequence of observations."""
  observation_dimensions = [
      historical_metric_values(history, metric, observation_range)
      for metric in metrics
  ]
  if include_lr:
    # Logartihm of the learning rate.
    observation_dimensions.append(np.log(historical_metric_values(
        history, LEARNING_RATE_METRIC, observation_range
    )))
  return np.stack(observation_dimensions, axis=1)


def new_learning_rate(action, history, action_multipliers, max_lr):
  """Calculates a new learning rate based on an action."""
  learning_rates = historical_metric_values(history, LEARNING_RATE_METRIC)
  assert learning_rates.shape[0] > 0, "No last learning rate found in history."
  current_lr = learning_rates[-1]
  return min(current_lr * action_multipliers[action], max_lr)
