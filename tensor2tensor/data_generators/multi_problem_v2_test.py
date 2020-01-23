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

"""Tests for tensor2tensor.data_generators.multi_problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensor2tensor.data_generators import multi_problem_v2
from tensor2tensor.data_generators import problem
import tensorflow.compat.v1 as tf


class MultiProblemV2Test(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {
          'inputs': [(0.0, ['string', 12]), np.array([12, 10])],
          'targets': ((0.0, ('string', 12)), (12, 10)),
      },
      {
          'inputs': [1.0, np.ones([2, 3])],
          'targets': (1.0, ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))),
      },
  )
  def test_tuplize(self, inputs, targets):
    self.assertEqual(multi_problem_v2.tuplize(inputs), targets)

  @parameterized.parameters(
      {
          'schedule': ('step', (100,), ((0.25, 0.75),)),
          'string': 'step @100 0.25 0.75',
      },
      {
          'schedule': ('step', (100, 200), ((0.25, 0.75), (0.62, 0.38))),
          'string': 'step @100 0.25 0.75 @200 0.62 0.38',
      },
      {
          'schedule': ('linear', (100, 200), ((0.25, 0.75), (0.62, 0.38))),
          'string': 'linear @100 0.25 0.75 @200 0.62 0.38',
      },
  )
  def test_encode_decode_schedule(self, schedule, string):
    self.assertEqual(multi_problem_v2.encode_schedule(schedule), string)
    self.assertEqual(multi_problem_v2.decode_schedule(string), schedule)

  @parameterized.parameters(
      {
          'x': np.array([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0]),
          'xp': np.array([0.0, 1.0]),
          'fp': np.array([0.2, 0.4]),
          'y': np.array([0.2, 0.2, 0.25, 0.3, 0.35, 0.4, 0.4]),
      },
      {
          'x': np.array([-1.0, 0.0, 0.5, 1.0, 2.0]),
          'xp': np.array([0.0, 1.0]),
          'fp': np.array([[0.2, 0.4], [0.4, 0.2]]),
          'y': np.array(
              [[0.2, 0.4], [0.2, 0.4], [0.3, 0.3], [0.4, 0.2], [0.4, 0.2]]),
      },
  )
  def test_linear_interpolation(self, x, xp, fp, y):
    self.assertAllClose(multi_problem_v2.linear_interpolation(x, xp, fp), y)

  @parameterized.parameters(
      {
          'x': np.array([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0]),
          'xp': np.array([0.0, 0.6, 0.9]),
          'fp': np.array([0.1, 0.9, 0.6]),
          'y': np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.6, 0.6]),
      },
      {
          'x': np.array([-1.0, 0.0, 0.5, 1.0, 2.0]),
          'xp': np.array([0.0, 0.6, 0.9]),
          'fp': np.array([[0.1, 0.4], [0.9, 0.2], [0.6, 0.9]]),
          'y': np.array(
              [[0.1, 0.4], [0.1, 0.4], [0.1, 0.4], [0.6, 0.9], [0.6, 0.9]]),
      },
  )
  def test_step_interpolation(self, x, xp, fp, y):
    self.assertAllClose(multi_problem_v2.step_interpolation(x, xp, fp), y)

  @parameterized.parameters(
      {
          'schedule': ('linear', (100, 200), ((0.25, 0.75), (0.62, 0.38))),
          'steps': np.array([50, 100, 150, 200, 250]),
          'pmfs': np.array(
              [[0.25, 0.75], [0.25, 0.75], [0.435, 0.565], [0.62, 0.38],
               [0.62, 0.38]]),
      },
      {
          'schedule': ('step', (100, 200), ((0.25, 0.75), (0.62, 0.38))),
          'steps': np.array([50, 100, 150, 200, 250]),
          'pmfs': np.array(
              [[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.62, 0.38],
               [0.62, 0.38]]),
      },
  )
  def test_get_schedule_distribution(self, schedule, steps, pmfs):
    with self.test_session() as sess:
      global_step = tf.train.get_or_create_global_step()
      output = multi_problem_v2.get_schedule_distribution(schedule, global_step)
      sess.run(global_step.initializer)
      for step, pmf in zip(steps, pmfs):
        sess.run(global_step.assign(step))
        self.assertAllClose(sess.run(output), pmf)

  @parameterized.parameters(
      {
          'pmf': np.array([1.0, 0.0], np.float32),
          'fns': [lambda: 0, lambda: 1],
          'rands': np.array([0.1, 0.4, 0.6, 0.9], np.float32),
          'targets': np.array([0, 0, 0, 0], np.float32),
      },
      {
          'pmf': np.array([0.2, 0.6, 0.2], np.float32),
          'fns': [lambda: 0, lambda: 1, lambda: 2],
          'rands': np.array([0.1, 0.4, 0.6, 0.9], np.float32),
          'targets': np.array([0, 1, 1, 2], np.float32),
      },
  )
  def test_categorical_case(self, pmf, fns, rands, targets):
    with self.test_session() as sess:
      for rand, target in zip(rands, targets):
        output = multi_problem_v2.categorical_case(pmf, fns, rand)
        self.assertEqual(sess.run(output), target)

  @parameterized.parameters(
      {
          'pmf': np.array([1.0, 0.0], np.float32),
          'num_datasets': 2,
          'sample_size': 10,
      },
      {
          'pmf': np.array([0.3, 0.7], np.float32),
          'num_datasets': 2,
          'sample_size': 400,
      },
      {
          'pmf': None,
          'num_datasets': 2,
          'sample_size': 400,
      },
  )
  def test_get_multi_dataset(self, pmf, num_datasets, sample_size):
    with self.test_session() as sess:
      datasets = [tf.data.Dataset.from_tensors(i) for i in range(num_datasets)]
      multi_dataset = multi_problem_v2.get_multi_dataset(datasets, pmf)
      multi_dataset = multi_dataset.batch(sample_size)
      iterator = multi_dataset.make_initializable_iterator()
      sess.run(iterator.initializer)
      sample_pmf = tf.reduce_mean(
          tf.one_hot(iterator.get_next(), num_datasets), 0)
      if pmf is None:
        pmf = np.array([1.0 / num_datasets] * num_datasets, np.float32)
      self.assertAllClose(sess.run(sample_pmf), pmf, rtol=0.1, atol=0.1)

  @parameterized.parameters(
      {
          'schedule': ('step', (100, 200), ((1.0, 0.0), (0.0, 1.0))),
          'num_datasets': 2,
          'sample_size': 20,
      },
      {
          'schedule': ('linear', (100, 200), ((0.6, 0.4), (0.1, 0.9))),
          'num_datasets': 2,
          'sample_size': 400,
      },
  )
  def test_multi_problem_v2(self, schedule, num_datasets, sample_size):

    class DummyProblem(problem.Problem):

      def dataset(self, *args, **kwargs):
        return tf.data.Dataset.from_tensors({'targets': 0.0})

    with self.test_session() as sess:
      for mode in [problem.DatasetSplit.TRAIN, problem.DatasetSplit.EVAL]:
        p = multi_problem_v2.MultiProblemV2(
            [DummyProblem() for _ in range(num_datasets)], schedule)
        global_step = tf.train.get_or_create_global_step()
        dataset = p.dataset(mode, global_step).batch(sample_size)
        iterator = dataset.make_initializable_iterator()
        features = iterator.get_next()
        sess.run(global_step.initializer)
        sess.run(iterator.initializer)
        sess.run(features)


if __name__ == '__main__':
  tf.test.main()
