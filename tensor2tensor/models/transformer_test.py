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

"""Tests for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import transformer

import tensorflow as tf


class TransformerTest(tf.test.TestCase):

  def _testTransformer(self, net):
    batch_size = 3
    input_length = 5
    target_length = 7
    vocab_size = 9
    hparams = transformer.transformer_tiny()
    p_hparams = problem_hparams.test_problem_hparams(hparams, vocab_size,
                                                     vocab_size)
    inputs = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, input_length, 1, 1))
    targets = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, target_length, 1, 1))
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(inputs, dtype=tf.int32),
          "targets": tf.constant(targets, dtype=tf.int32),
          "target_space_id": tf.constant(1, dtype=tf.int32),
      }
      model = net(hparams, p_hparams)
      shadred_logits, _, _ = model.model_fn(features, True)
      logits = tf.concat(shadred_logits, 0)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (batch_size, target_length, 1, 1, vocab_size))

  def testTransformer(self):
    self._testTransformer(transformer.Transformer)


if __name__ == "__main__":
  tf.test.main()
