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

"""Tests for tensor2tensor.trax.inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
from tensor2tensor.trax import inputs
import tensorflow as tf
import tensorflow_datasets as tfds


def test_dataset_ints(lengths):
  """Create a test dataset of int64 tensors of shape [length]."""
  def generator():
    """Sample generator of sequences of shape [length] of type int64."""
    for length in lengths:
      x = np.zeros([length], dtype=np.int64)
      yield (x, x)  # Inputs and targets are the same here.
  types = (tf.int64, tf.int64)
  shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
  return tf.data.Dataset.from_generator(
      generator, output_types=types, output_shapes=shapes)


class InputsTest(tf.test.TestCase):

  def setUp(self):
    gin.clear_config()

  def test_batch_fun(self):
    dataset = test_dataset_ints([32])
    dataset = dataset.repeat(10)
    batches = inputs.batch_fun(
        dataset, True, ([None], [None]), [], 1, batch_size=10)
    count = 0
    for example in tfds.as_numpy(batches):
      count += 1
      self.assertEqual(example[0].shape[0], 10)  # Batch size = 10.
    self.assertEqual(count, 1)  # Just one batch here.

  def test_batch_fun_n_devices(self):
    dataset = test_dataset_ints([32])
    dataset = dataset.repeat(9)
    batches = inputs.batch_fun(
        dataset, True, ([None], [None]), [], 9, batch_size=10)
    count = 0
    for example in tfds.as_numpy(batches):
      count += 1
      # Batch size adjusted to be divisible by n_devices.
      self.assertEqual(example[0].shape[0], 9)
    self.assertEqual(count, 1)  # Just one batch here.


if __name__ == "__main__":
  tf.test.main()
