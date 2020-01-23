# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Timeseries data generator tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.data_generators import timeseries_data_generator

import tensorflow.compat.v1 as tf


class TimeseriesDataGeneratorTest(tf.test.TestCase):

  def testGenerateData(self):
    timeseries_params = [{
        "m": 0.006,
        "b": 300.0,
        "A": 50.0,
        "freqcoeff": 1500.0,
        "rndA": 15.0,
        "fn": np.sin
    }, {
        "m": 0.000,
        "b": 500.0,
        "A": 35.0,
        "freqcoeff": 3500.0,
        "rndA": 25.0,
        "fn": np.cos
    }, {
        "m": -0.003,
        "b": 800.0,
        "A": 65.0,
        "freqcoeff": 2500.0,
        "rndA": 5.0,
        "fn": np.sin
    }, {
        "m": 0.009,
        "b": 600.0,
        "A": 20.0,
        "freqcoeff": 1000.0,
        "rndA": 1.0,
        "fn": np.cos
    }, {
        "m": 0.002,
        "b": 700.0,
        "A": 40.0,
        "freqcoeff": 2000.0,
        "rndA": 35.0,
        "fn": np.sin
    }, {
        "m": -0.008,
        "b": 1000.0,
        "A": 70.0,
        "freqcoeff": 3000.0,
        "rndA": 25.0,
        "fn": np.cos
    }, {
        "m": 0.000,
        "b": 100.0,
        "A": 25.0,
        "freqcoeff": 1500.0,
        "rndA": 10.0,
        "fn": np.sin
    }, {
        "m": 0.004,
        "b": 1500.0,
        "A": 54.0,
        "freqcoeff": 900.0,
        "rndA": 55.0,
        "fn": np.cos
    }, {
        "m": 0.005,
        "b": 2000.0,
        "A": 32.0,
        "freqcoeff": 1100.0,
        "rndA": 43.0,
        "fn": np.sin
    }, {
        "m": 0.010,
        "b": 2500.0,
        "A": 43.0,
        "freqcoeff": 1900.0,
        "rndA": 53.0,
        "fn": np.cos
    }]
    multi_timeseries = timeseries_data_generator.generate_data(
        20, timeseries_params)

    self.assertEqual(10, len(multi_timeseries))
    self.assertEqual(20, len(multi_timeseries[0]))


if __name__ == "__main__":
  tf.test.main()
