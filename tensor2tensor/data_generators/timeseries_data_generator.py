# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Data generator for the timeseries problem."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def generate_data(timeseries_length, timeseries_params):
  """Generates synthetic timeseries using input parameters.

  Each generated timeseries has timeseries_length data points.
  Parameters for each timeseries are specified by timeseries_params.

  Args:
    timeseries_length: Number of data points to generate for each timeseries.
    timeseries_params: Parameters used to generate the timeseries. The following
      parameters need to be specified for each timeseries:
      m = Slope of the timeseries used to compute the timeseries trend.
      b = y-intercept of the timeseries used to compute the timeseries trend.
      A = Timeseries amplitude used to compute timeseries period.
      freqcoeff = Frequency coefficient used to compute timeseries period.
      rndA = Random amplitude used to inject noise into the timeseries.
      fn = Base timeseries function (np.cos or np.sin).
      Example params for two timeseries.
      [{"m": 0.006, "b": 300.0, "A":50.0, "freqcoeff":1500.0, "rndA":15.0,
      "fn": np.sin},
      {"m": 0.000, "b": 500.0, "A":35.0, "freqcoeff":3500.0, "rndA":25.0,
      "fn": np.cos}]

  Returns:
    Multi-timeseries (list of list).
  """
  x = range(timeseries_length)

  multi_timeseries = []
  for p in timeseries_params:
    # Trend
    y1 = [p["m"] * i + p["b"] for i in x]
    # Period
    y2 = [p["A"] * p["fn"](i / p["freqcoeff"]) for i in x]
    # Noise
    y3 = np.random.normal(0, p["rndA"], timeseries_length).tolist()
    # Sum of Trend, Period and Noise. Replace negative values with zero.
    y = [max(a + b + c, 0) for a, b, c in zip(y1, y2, y3)]
    multi_timeseries.append(y)

  return multi_timeseries
