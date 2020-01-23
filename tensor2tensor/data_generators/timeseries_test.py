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

"""Timeseries generators tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from tensor2tensor.data_generators import timeseries

import tensorflow.compat.v1 as tf


class TimeseriesTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tmp_dir = tf.test.get_temp_dir()
    shutil.rmtree(cls.tmp_dir)
    os.mkdir(cls.tmp_dir)

  def testTimeseriesToyProblem(self):
    problem = timeseries.TimeseriesToyProblem()
    problem.generate_data(self.tmp_dir, self.tmp_dir)

    dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, self.tmp_dir)
    features = dataset.make_one_shot_iterator().get_next()

    examples = []
    exhausted = False
    with self.test_session() as sess:
      examples.append(sess.run(features))
      examples.append(sess.run(features))
      examples.append(sess.run(features))
      examples.append(sess.run(features))

      try:
        sess.run(features)
      except tf.errors.OutOfRangeError:
        exhausted = True

    self.assertTrue(exhausted)
    self.assertEqual(4, len(examples))

    self.assertNotEqual(
        list(examples[0]["inputs"][0, 0]), list(examples[1]["inputs"][0, 0]))

  def testTimeseriesToyProblemNoInputs(self):
    problem = timeseries.TimeseriesToyProblemNoInputs()
    problem.generate_data(self.tmp_dir, self.tmp_dir)

    dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, self.tmp_dir)
    features = dataset.make_one_shot_iterator().get_next()

    examples = []
    exhausted = False
    with self.test_session() as sess:
      examples.append(sess.run(features))
      examples.append(sess.run(features))
      examples.append(sess.run(features))
      examples.append(sess.run(features))
      examples.append(sess.run(features))

      try:
        sess.run(features)
      except tf.errors.OutOfRangeError:
        exhausted = True

    self.assertTrue(exhausted)
    self.assertEqual(5, len(examples))

  def testTimeseriesSyntheticData10Series100kSamples(self):
    problem = timeseries.TimeseriesSyntheticDataSeries10Samples100k()
    self.assertEqual(10, problem.num_series)
    self.assertEqual(250, problem.num_input_timestamps)
    self.assertEqual(100, problem.num_target_timestamps)


if __name__ == "__main__":
  tf.test.main()
