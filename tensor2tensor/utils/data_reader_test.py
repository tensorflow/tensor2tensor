# Copyright 2017 Google Inc.
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
from tensor2tensor.utils import data_reader

import tensorflow as tf


class DataReaderTest(tf.test.TestCase):

  def testExamplesQueue(self):
    tf.set_random_seed(1)
    tmp_dir = self.get_temp_dir()
    (_, tmp_file_path) = tempfile.mkstemp(dir=tmp_dir)
    tmp_file_name = os.path.basename(tmp_file_path)

    # Generate a file with 100 examples.
    def test_generator():
      for i in xrange(100):
        yield {"inputs": [i], "targets": [i], "floats": [i + 0.5]}

    generator_utils.generate_files(test_generator(), tmp_file_name, tmp_dir)
    self.assertTrue(tf.gfile.Exists(tmp_file_path + "-00000-of-00001"))

    examples_train = data_reader.examples_queue(
        [tmp_file_path + "*"], {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64)
        },
        training=True)
    examples_eval = data_reader.examples_queue(
        [tmp_file_path + "*"], {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64),
            "floats": tf.VarLenFeature(tf.float32)
        },
        training=False)
    with tf.train.MonitoredSession() as session:
      # Evaluation data comes in the same order as in the file, check 10.
      for i in xrange(10):
        examples = session.run(examples_eval)
        self.assertEqual(len(examples["inputs"]), 1)
        self.assertEqual(len(examples["targets"]), 1)
        self.assertEqual(examples["inputs"][0], i)
        self.assertEqual(examples["targets"][0], i)
        self.assertEqual(examples["floats"][0], i + 0.5)
      # Training data is shuffled.
      is_shuffled = False
      for i in xrange(10):
        examples = session.run(examples_train)
        self.assertEqual(len(examples["inputs"]), 1)
        self.assertEqual(len(examples["targets"]), 1)
        self.assertEqual(examples["inputs"][0], examples["targets"][0])
        if examples["inputs"][0] != i:
          is_shuffled = True
      self.assertTrue(is_shuffled)

    # Clean up.
    os.remove(tmp_file_path + "-00000-of-00001")
    os.remove(tmp_file_path)

  def testBatchExamples(self):
    tf.set_random_seed(1)
    tmp_dir = self.get_temp_dir()
    (_, tmp_file_path) = tempfile.mkstemp(dir=tmp_dir)
    tmp_file_name = os.path.basename(tmp_file_path)

    # Generate a file with 100 examples, n-th example of length n + 1.
    def test_generator():
      for i in xrange(100):
        yield {"inputs": [i + 1 for _ in xrange(i + 1)], "targets": [i + 1]}

    generator_utils.generate_files(test_generator(), tmp_file_name, tmp_dir)
    self.assertTrue(tf.gfile.Exists(tmp_file_path + "-00000-of-00001"))

    examples_train = data_reader.examples_queue([tmp_file_path + "*"], {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64)
    }, True)
    batch_train = data_reader.batch_examples(examples_train, 4)
    examples_eval = data_reader.examples_queue([tmp_file_path + "*"], {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64)
    }, False)
    batch_eval = data_reader.batch_examples(examples_eval, 2)
    session, coord = tf.Session(), tf.train.Coordinator()
    with session.as_default():
      tf.train.start_queue_runners(coord=coord)

      # Evaluation data comes in the same order as in the file.
      # The first batch will be inputs=[[1, 0], [2, 2]], targets=[[1], [2]].
      examples = session.run(batch_eval)
      self.assertAllClose(examples["inputs"], np.array([[1, 0], [2, 2]]))
      self.assertAllClose(examples["targets"], np.array([[1], [2]]))
      # Check the second batch too.
      examples = session.run(batch_eval)
      self.assertAllClose(examples["inputs"],
                          np.array([[3, 3, 3, 0], [4, 4, 4, 4]]))
      self.assertAllClose(examples["targets"], np.array([[3], [4]]))

      # Training data is shuffled but shouldn't have too many pads.
      for _ in xrange(10):
        examples = session.run(batch_train)
        inputs = examples["inputs"]
        # Only 3 out of 4 examples in a batch have padding zeros at all.
        pad_per_example = (inputs.size - np.count_nonzero(inputs)) // 3
        # Default bucketing is in steps of 8 until 64 and 32 later.
        if int(max(examples["targets"])) < 64:
          self.assertLess(pad_per_example, 8)
        else:
          self.assertLess(pad_per_example, 32)

    # Clean up.
    coord.request_stop()
    coord.join()
    os.remove(tmp_file_path + "-00000-of-00001")
    os.remove(tmp_file_path)


if __name__ == "__main__":
  tf.test.main()
