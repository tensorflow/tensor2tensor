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

"""Tests for TransformerRevnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models.research import transformer_revnet

import tensorflow as tf


def transformer_revnet_test():
  hparams = transformer_revnet.transformer_revnet_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 2
  return hparams


class TransformerRevnetTest(tf.test.TestCase):

  def testTransformer(self):
    batch_size = 3
    input_length = 5
    target_length = 7
    vocab_size = 9
    hparams = transformer_revnet_test()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size)
    hparams.problems = [p_hparams]
    inputs = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, input_length, 1, 1))
    targets = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, target_length, 1, 1))
    features = {
        "inputs": tf.constant(inputs, dtype=tf.int32),
        "targets": tf.constant(targets, dtype=tf.int32),
        "target_space_id": tf.constant(1, dtype=tf.int32),
    }
    model = transformer_revnet.TransformerRevnet(
        hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
    logits, _ = model(features)
    grads = tf.gradients(
        tf.reduce_mean(logits), [features["inputs"]] + tf.global_variables())
    grads = [g for g in grads if g is not None]

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      logits_val, _ = session.run([logits, grads])
    self.assertEqual(logits_val.shape, (batch_size, target_length, 1, 1,
                                        vocab_size))


if __name__ == "__main__":
  tf.test.main()
