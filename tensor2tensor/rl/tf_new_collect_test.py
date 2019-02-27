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

"""Tests for the new collect."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.rl import tf_new_collect

import tensorflow as tf

from munch import Munch


FLAGS = tf.flags.FLAGS


def test_policy(ob, batch_size):
  action = tf.random.uniform((batch_size,), maxval=9, dtype=tf.int32)
  pdf = tf.random.uniform((batch_size,), dtype=tf.float32)
  value_function = tf.random.uniform((batch_size,), dtype=tf.float32)
  return action, pdf, value_function


EPOCH_LENGTH = 10
BATCH_SIZE = 3
HISTORY = 4


class TfNewCollectTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    tf.enable_eager_execution()

  def test_basic(self):
    batch_env = tf_new_collect.NewSimulatedBatchEnv(BATCH_SIZE)
    batch_env = tf_new_collect.NewStackWrapper(batch_env, HISTORY)

    x = tf_new_collect.new_define_collect(
        batch_env, Munch(epoch_length=EPOCH_LENGTH), test_policy,
        force_beginning_resets=True
    )
    # (2, 1) is observation shape
    self.assertEqual(x[0].shape, (EPOCH_LENGTH, BATCH_SIZE, HISTORY, 2, 1))
    initial_ob = 10
    expected_obs = [initial_ob] * HISTORY + list(
        range(initial_ob + 1, initial_ob + EPOCH_LENGTH)
    )
    for offset in range(HISTORY):
      self.assertEqual(
          list(x[0][:, 0, offset, 0, 0].numpy()),
          expected_obs[offset:][:EPOCH_LENGTH]
      )


if __name__ == "__main__":
  tf.test.main()
