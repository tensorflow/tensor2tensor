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

"""BlueNet tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import bluenet

import tensorflow as tf


class BlueNetTest(tf.test.TestCase):

  def testBlueNet(self):
    vocab_size = 9
    x = np.random.random_integers(1, high=vocab_size - 1, size=(3, 5, 1, 1))
    y = np.random.random_integers(1, high=vocab_size - 1, size=(3, 1, 1, 1))
    hparams = bluenet.bluenet_tiny()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size)
    with self.test_session() as session:
      tf.train.get_or_create_global_step()
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = bluenet.BlueNet(
          hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      sharded_logits, _ = model.model_fn(features)
      logits = tf.concat(sharded_logits, 0)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (3, 5, 1, 1, vocab_size))


if __name__ == "__main__":
  tf.test.main()
