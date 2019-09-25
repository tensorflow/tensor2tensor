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


def historical_metric_values(history, metric):
  """Converts a metric stream from a trax History object into a numpy array."""
  metric_sequence = history.get(*metric)
  return np.array([
      metric_value for (_, metric_value) in metric_sequence
  ])


def metric_to_observation(metric_values, metric_range):
  """Clips and scales the metric to the [-1, 1] interval."""
  (low, high) = metric_range
  clipped_values = np.clip(metric_values, low, high)
  return (clipped_values - low) / (high - low) * 2 - 1


def control_to_observation(control_values, control_config):
  """Flips, logarithms, clips and scales the control to the [-1, 1] interval."""
  (_, _, (low, high), flip) = control_config
  def transform(x):
    return np.log(maybe_flip(x, flip))
  (log_control_values, log_low, log_high) = map(
      transform, (control_values, low, high)
  )
  if flip:
    (log_low, log_high) = (log_high, log_low)
  return metric_to_observation(log_control_values, (log_low, log_high))


def control_metric(name):
  """Returns the (mode, metric) pair in History for the given control."""
  return ("train", "training/{}".format(name))


def maybe_flip(value, flip):
  """Flips a control (or not).

  Meant to translate controls that naturally take values close to 1
  (e.g. momentum) to a space where multiplication makes sense (i.e. close to 0).

  Args:
    value: float or numpy array, value of the control.
    flip: bool, whether to flip or not.

  Returns:
    Either value or 1 - value based on flip.
  """
  if flip:
    value = 1 - value
  return value


def history_to_observations(
    history, metrics, observation_range, control_configs=None):
  """Converts a trax History object into a sequence of observations."""
  observation_dimensions = [
      metric_to_observation(  # pylint: disable=g-complex-comprehension
          historical_metric_values(history, metric), observation_range
      )
      for metric in metrics
  ]
  if control_configs is not None:
    for control_config in control_configs:
      (control_name, _, _, _) = control_config
      observation_dimensions.append(control_to_observation(
          historical_metric_values(history, control_metric(control_name)),
          control_config,
      ))
  return np.stack(observation_dimensions, axis=1)


def update_control(control_config, action, history, action_multipliers):
  """Calculates a new value of a control based on an action."""
  (name, _, (low, high), flip) = control_config
  metric = control_metric(name)
  control_values = historical_metric_values(history, metric)
  assert control_values.shape[0] > 0, (
      "No last control {} found in history.".format(name))
  current_control = control_values[-1]
  (current_control, low, high) = maybe_flip(
      np.array([current_control, low, high]), flip
  )
  if flip:
    (low, high) = (high, low)
  new_control = np.clip(
      current_control * action_multipliers[action], low, high
  )
  return maybe_flip(new_control, flip)
