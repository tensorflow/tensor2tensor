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

"""Tests for Modalities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import modalities
from tensor2tensor.utils import expert_utils

import tensorflow as tf


class ModalityTest(tf.test.TestCase):

  def testSymbolModalityInputs(self):
    batch_size = 10
    num_datashards = 5
    length = 5
    vocab_size = 5000
    hidden_size = 9
    model_hparams = common_hparams.basic_params1()
    model_hparams.hidden_size = hidden_size
    model_hparams.mode = tf.estimator.ModeKeys.TRAIN
    x = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, length, 1, 1))
    m = modalities.SymbolModality(model_hparams, vocab_size)
    data_parallelism = expert_utils.Parallelism(
        ["/device:CPU:0"] * num_datashards)
    with self.test_session() as session:
      xs = tf.split(x, num_datashards)
      sharded_output = m.bottom_sharded(xs, data_parallelism)
      output = tf.concat(sharded_output, 0)
      session.run(tf.global_variables_initializer())
      res = session.run(output)
    self.assertEqual(res.shape, (batch_size, length, 1, hidden_size))

  def testSymbolModalityTargets(self):
    batch_size = 10
    num_datashards = 5
    length = 6
    height = 7
    hidden_size = 9
    vocab_size = 11
    model_hparams = common_hparams.basic_params1()
    model_hparams.hidden_size = hidden_size
    model_hparams.mode = tf.estimator.ModeKeys.TRAIN
    body_output = -1 + np.random.random_integers(
        100, size=(batch_size, length, height, hidden_size))
    targets = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, length, height, 1))
    m = modalities.SymbolModality(model_hparams, vocab_size)
    data_parallelism = expert_utils.Parallelism(
        ["/device:CPU:0"] * num_datashards)
    with self.test_session() as session:
      sharded_body_output = tf.split(tf.to_float(body_output), num_datashards)
      sharded_targets = tf.split(targets, num_datashards)
      sharded_logits = m.top_sharded(sharded_body_output, sharded_targets,
                                     data_parallelism)
      train_loss = m.loss_sharded(sharded_logits, sharded_targets,
                                  data_parallelism)
      logits = tf.concat(sharded_logits, 0)
      session.run(tf.global_variables_initializer())
      res1, res2 = session.run((logits, train_loss))
    self.assertEqual(res1.shape, (batch_size, length, height, 1, vocab_size))
    self.assertEqual(res2.shape, ())

  def testSymbolModalityTargetsFactored(self):
    batch_size = 10
    num_datashards = 5
    length = 6
    height = 7
    hidden_size = 9
    vocab_size = 11
    model_hparams = common_hparams.basic_params1()
    model_hparams.factored_logits = True
    model_hparams.hidden_size = hidden_size
    model_hparams.mode = tf.estimator.ModeKeys.TRAIN
    body_output = -1 + np.random.random_integers(
        100, size=(batch_size, length, height, hidden_size))
    targets = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, length, height, 1))
    m = modalities.SymbolModality(model_hparams, vocab_size)
    data_parallelism = expert_utils.Parallelism(
        ["/device:CPU:0"] * num_datashards)
    with self.test_session() as session:
      sharded_body_output = tf.split(tf.to_float(body_output), num_datashards)
      sharded_targets = tf.split(targets, num_datashards)
      sharded_logits = m.top_sharded(sharded_body_output, sharded_targets,
                                     data_parallelism)
      train_loss = m.loss_sharded(sharded_logits, sharded_targets,
                                  data_parallelism)
      logits = tf.concat(sharded_logits, 0)
      session.run(tf.global_variables_initializer())
      res1, res2 = session.run((logits, train_loss))
    self.assertEqual(res1.shape, (batch_size, length, height, 1, vocab_size))
    self.assertEqual(res2.shape, ())


if __name__ == "__main__":
  tf.test.main()
