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

"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import sys

import cloudpickle
import numpy as np


def get_pickle_module():
  """Returns the appropriate pickle module based on Python version."""
  # TODO(gilmer, lukaszkaiser): figure out how to use cloudpickle in python3.
  # Currently the code throws an error when run in python3.
  if sys.version_info[0] < 3:
    return cloudpickle
  else:
    return pickle


def gumbel_sample(log_probs):
  """Gumbel sampling from a categorical distribution."""
  u = np.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
  g = -np.log(-np.log(u))
  return np.argmax(log_probs + g, axis=-1)
