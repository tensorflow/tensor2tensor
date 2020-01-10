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

"""Tests for Modalities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import modalities
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import test_utils

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


class ModalityTest(tf.test.TestCase):

  @test_utils.run_in_graph_and_eager_modes()
  def testGetForAllModalities(self):
    for modality in modalities.ModalityType.get_choices():
      bottom = modalities.get_bottom(modality)
      loss = modalities.get_loss(modality)
      name = modalities.get_name(modality)
      targets_bottom = modalities.get_targets_bottom(modality)
      top = modalities.get_top(modality)
      weights_fn = modalities.get_weights_fn(modality)
      self.assertIsNotNone(bottom,
                           msg="{} has no default bottom".format(modality))
      self.assertIsNotNone(loss, msg="{} has no default loss".format(modality))
      self.assertIsNotNone(name, msg="{} has no default name".format(modality))
      self.assertIsNotNone(
          targets_bottom,
          msg="{} has no default targets_bottom".format(modality))
      self.assertIsNotNone(top, msg="{} has no default top".format(modality))
      self.assertIsNotNone(weights_fn,
                           msg="{} has no default weights_fn".format(modality))

  @test_utils.run_in_graph_and_eager_modes()
  def testSymbolModalityInputs(self):
    batch_size = 10
    num_datashards = 5
    length = 5
    vocab_size = 5000
    hidden_size = 9
    model_hparams = common_hparams.basic_params1()
    model_hparams.hidden_size = hidden_size
    model_hparams.mode = tf.estimator.ModeKeys.TRAIN
    x = np.random.randint(
        vocab_size, size=(batch_size, length, 1, 1))
    data_parallelism = expert_utils.Parallelism(
        ["/device:CPU:0"] * num_datashards)
    xs = tf.split(x, num_datashards)
    sharded_output = data_parallelism(
        modalities.get_bottom(modalities.ModalityType.SYMBOL),
        xs,
        model_hparams,
        vocab_size)
    output = tf.concat(sharded_output, 0)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(output)
    self.assertEqual(res.shape, (batch_size, length, 1, hidden_size))

  @test_utils.run_in_graph_and_eager_modes()
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
    body_output = np.random.randint(
        100, size=(batch_size, length, height, hidden_size))
    targets = np.random.randint(
        vocab_size, size=(batch_size, length, height, 1))
    data_parallelism = expert_utils.Parallelism(
        ["/device:CPU:0"] * num_datashards)
    sharded_body_output = tf.split(tf.to_float(body_output), num_datashards)
    sharded_targets = tf.split(targets, num_datashards)
    sharded_logits = data_parallelism(
        modalities.get_top(modalities.ModalityType.SYMBOL),
        sharded_body_output,
        sharded_targets,
        model_hparams,
        vocab_size)
    sharded_loss_num, sharded_loss_den = data_parallelism(
        modalities.get_loss(modalities.ModalityType.SYMBOL),
        sharded_logits,
        sharded_targets,
        model_hparams,
        vocab_size,
        modalities.get_weights_fn(modalities.ModalityType.SYMBOL))
    train_loss = (tf.add_n(sharded_loss_num) /
                  tf.maximum(1.0, tf.add_n(sharded_loss_den)))
    logits = tf.concat(sharded_logits, 0)
    self.evaluate(tf.global_variables_initializer())
    res1, res2 = self.evaluate((logits, train_loss))
    self.assertEqual(res1.shape, (batch_size, length, height, 1, vocab_size))
    self.assertEqual(res2.shape, ())

  @test_utils.run_in_graph_mode_only()
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
    body_output = np.random.randint(
        100, size=(batch_size, length, height, hidden_size))
    targets = np.random.randint(
        vocab_size, size=(batch_size, length, height, 1))
    data_parallelism = expert_utils.Parallelism(
        ["/device:CPU:0"] * num_datashards)
    with self.test_session() as session:
      sharded_body_output = tf.split(tf.to_float(body_output), num_datashards)
      sharded_targets = tf.split(targets, num_datashards)
      sharded_logits = data_parallelism(
          modalities.get_top(modalities.ModalityType.SYMBOL),
          sharded_body_output,
          sharded_targets,
          model_hparams,
          vocab_size)
      sharded_loss_num, sharded_loss_den = data_parallelism(
          modalities.get_loss(modalities.ModalityType.SYMBOL),
          sharded_logits,
          sharded_targets,
          model_hparams,
          vocab_size,
          modalities.get_weights_fn(modalities.ModalityType.SYMBOL))
      train_loss = (tf.add_n(sharded_loss_num) /
                    tf.maximum(1.0, tf.add_n(sharded_loss_den)))
      logits = tf.concat(sharded_logits, 0)
      session.run(tf.global_variables_initializer())
      res1, res2 = session.run((logits, train_loss))
    self.assertEqual(res1.shape, (batch_size, length, height, 1, vocab_size))
    self.assertEqual(res2.shape, ())


if __name__ == "__main__":
  tf.test.main()
