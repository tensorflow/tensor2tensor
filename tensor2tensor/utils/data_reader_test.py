# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Data reader test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

# Dependency imports

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem as problem_mod
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class TestProblem(problem_mod.Problem):

  def generator(self, data_dir, tmp_dir, is_training):
    for i in xrange(30):
      yield {"inputs": [i] * (i + 1), "targets": [i], "floats": [i + 0.5]}

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(data_dir, 1, shuffled=True)
    dev_paths = self.dev_filepaths(data_dir, 1, shuffled=True)
    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir, True), train_paths)
    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir, False), dev_paths)

  def hparams(self, defaults, model_hparams):
    pass

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
        "floats": tf.VarLenFeature(tf.float32),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def preprocess_example(self, example, unused_mode, unused_hparams):
    example["new_field"] = tf.constant([42.42])
    return example


def generate_test_data(problem, tmp_dir):
  problem.generate_data(tmp_dir, tmp_dir)
  return [problem.filepattern(tmp_dir, tf.estimator.ModeKeys.TRAIN)]


class DataReaderTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    tf.set_random_seed(1)
    cls.problem = registry.problem("test_problem")
    cls.data_dir = tempfile.gettempdir()
    cls.filepatterns = generate_test_data(cls.problem, cls.data_dir)

  @classmethod
  def tearDownClass(cls):
    # Clean up files
    for fp in cls.filepatterns:
      files = tf.gfile.Glob(fp)
      for f in files:
        os.remove(f)

  def testBasicExampleReading(self):
    dataset = self.problem.dataset(
        tf.estimator.ModeKeys.TRAIN, data_dir=self.data_dir)
    examples = dataset.make_one_shot_iterator().get_next()
    with tf.train.MonitoredSession() as sess:
      # Check that there are multiple examples that have the right fields of the
      # right type (lists of int/float).
      for _ in xrange(10):
        ex_val = sess.run(examples)
        inputs, targets, floats = (ex_val["inputs"], ex_val["targets"],
                                   ex_val["floats"])
        self.assertEqual(np.int64, inputs.dtype)
        self.assertEqual(np.int64, targets.dtype)
        self.assertEqual(np.float32, floats.dtype)
        for field in [inputs, targets, floats]:
          self.assertGreater(len(field), 0)

  def testPreprocess(self):
    dataset = self.problem.dataset(
        tf.estimator.ModeKeys.TRAIN, data_dir=self.data_dir)
    examples = dataset.make_one_shot_iterator().get_next()
    with tf.train.MonitoredSession() as sess:
      ex_val = sess.run(examples)
      # problem.preprocess_example has been run
      self.assertAllClose([42.42], ex_val["new_field"])

  def testLengthFilter(self):
    max_len = 15
    dataset = self.problem.dataset(
        tf.estimator.ModeKeys.TRAIN, data_dir=self.data_dir)
    dataset = dataset.filter(
        lambda ex: data_reader.example_valid_size(ex, 0, max_len))
    examples = dataset.make_one_shot_iterator().get_next()
    with tf.train.MonitoredSession() as sess:
      ex_lens = []
      for _ in xrange(max_len):
        ex_lens.append(len(sess.run(examples)["inputs"]))

    self.assertAllEqual(list(range(1, max_len + 1)), sorted(ex_lens))

  def testBatchingSchemeMaxLength(self):
    scheme = data_reader._batching_scheme(
        batch_size=20,
        max_length=None,
        min_length_bucket=8,
        length_bucket_step=1.1,
        drop_long_sequences=False)
    self.assertGreater(scheme["max_length"], 10000)

    scheme = data_reader._batching_scheme(
        batch_size=20,
        max_length=None,
        min_length_bucket=8,
        length_bucket_step=1.1,
        drop_long_sequences=True)
    self.assertEqual(scheme["max_length"], 20)

    scheme = data_reader._batching_scheme(
        batch_size=20,
        max_length=15,
        min_length_bucket=8,
        length_bucket_step=1.1,
        drop_long_sequences=True)
    self.assertEqual(scheme["max_length"], 15)

    scheme = data_reader._batching_scheme(
        batch_size=20,
        max_length=15,
        min_length_bucket=8,
        length_bucket_step=1.1,
        drop_long_sequences=False)
    self.assertGreater(scheme["max_length"], 10000)

  def testBatchingSchemeBuckets(self):
    scheme = data_reader._batching_scheme(
        batch_size=128,
        max_length=0,
        min_length_bucket=8,
        length_bucket_step=1.1)
    boundaries, batch_sizes = scheme["boundaries"], scheme["batch_sizes"]
    self.assertEqual(len(boundaries), len(batch_sizes) - 1)
    expected_boundaries = [
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30,
        33, 36, 39, 42, 46, 50, 55, 60, 66, 72, 79, 86, 94, 103, 113, 124
    ]
    self.assertEqual(expected_boundaries, boundaries)
    expected_batch_sizes = [
        16, 12, 12, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2,
        2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
    self.assertEqual(expected_batch_sizes, batch_sizes)

    scheme = data_reader._batching_scheme(
        batch_size=128,
        max_length=0,
        min_length_bucket=8,
        length_bucket_step=1.1,
        shard_multiplier=2)
    boundaries, batch_sizes = scheme["boundaries"], scheme["batch_sizes"]
    self.assertAllEqual([bs * 2 for bs in expected_batch_sizes], batch_sizes)
    self.assertEqual(expected_boundaries, boundaries)

    scheme = data_reader._batching_scheme(
        batch_size=128,
        max_length=0,
        min_length_bucket=8,
        length_bucket_step=1.1,
        length_multiplier=2)
    boundaries, batch_sizes = scheme["boundaries"], scheme["batch_sizes"]
    self.assertAllEqual([b * 2 for b in expected_boundaries], boundaries)
    self.assertEqual([max(1, bs // 2)
                      for bs in expected_batch_sizes], batch_sizes)

  def testBucketBySeqLength(self):

    def example_len(ex):
      return tf.shape(ex["inputs"])[0]

    boundaries = [10, 20, 30]
    batch_sizes = [10, 8, 4, 2]
    window_size = 40

    dataset = self.problem.dataset(
        tf.estimator.ModeKeys.TRAIN, data_dir=self.data_dir)
    dataset = data_reader.bucket_by_sequence_length(
        dataset, example_len, boundaries, batch_sizes, window_size)
    batch = dataset.make_one_shot_iterator().get_next()

    input_vals = []
    obs_batch_sizes = []
    with tf.train.MonitoredSession() as sess:
      # Until OutOfRangeError
      while True:
        batch_val = sess.run(batch)
        batch_inputs = batch_val["inputs"]
        batch_size, max_len = batch_inputs.shape
        obs_batch_sizes.append(batch_size)
        for inputs in batch_inputs:
          input_val = inputs[0]
          input_vals.append(input_val)
          # The inputs were constructed such that they were repeated value+1
          # times (i.e. if the inputs value is 7, the example has 7 repeated 8
          # times).
          repeat = input_val + 1
          # Check padding
          self.assertAllEqual([input_val] * repeat + [0] * (max_len - repeat),
                              inputs)

    # Check that all inputs came through
    self.assertEqual(list(range(30)), sorted(input_vals))
    # Check that we saw variable batch size
    self.assertTrue(len(set(obs_batch_sizes)) > 1)


if __name__ == "__main__":
  tf.test.main()
