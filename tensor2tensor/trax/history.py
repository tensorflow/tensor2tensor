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

"""trax history."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class History(object):
  """History of metrics.

  History contains the metrics recorded during training and evaluation.
  Save data with history.append(metric, value, step, mode) and get a sequence
  of data by calling history.get(metric, mode). For example:

  history.append("metrics/accuracy", 0.04, 1, "train")
  history.append("metrics/accuracy", 0.31, 1000, "train")
  history.get("metrics/accuracy", "train")
  # returns [(1, 0.04), (1000, 0.31)]
  """

  def __init__(self):
    # Structure is
    # values = {
    #   "mode1": {
    #     "metric1": [val1, val2],
    #     ...
    #   },
    #   "mode2": ...
    # }
    self._values = {}

  def append(self, metric, value, step, mode):
    """Append (step, value) pair to history for the given mode and metric."""
    if mode not in self._values:
      self._values[mode] = collections.defaultdict(list)
    self._values[mode][metric].append((step, value))

  def get(self, metric, mode):
    """Get the history for the given metric and mode."""
    if mode not in self._values:
      return []
    return list(self._values[mode][metric])
