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

"""Tests for T2TModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.utils import hparam
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import test_utils

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


class T2TModelTest(tf.test.TestCase):

  @test_utils.run_in_graph_and_eager_modes()
  def testSummarizeLosses(self):
    with tf.Graph().as_default():
      model = t2t_model.T2TModel(hparam.HParams())
      losses = {"training": tf.random_normal([]),
                "extra": tf.random_normal([])}
      outputs = model._summarize_losses(losses)
      self.assertIsNone(outputs, None)
      self.assertEqual(
          len(tf.get_collection(tf.GraphKeys.SUMMARIES, scope="losses")),
          len(losses))

  def testLossSingleWeights(self):
    """Ensure _loss_single() respects optional 'weights' argument."""
    with tf.Graph().as_default():
      with self.test_session() as sess:
        batch_size = 2
        sequence_size = 16
        vocab_size = 3

        model_hparams = hparam.HParams(
            prepend_mode="none",
            loss={},
            weights_fn={},
            label_smoothing=0.0,
            shared_embedding_and_softmax_weights=False)

        ph = problem_hparams.TestProblem(
            vocab_size, vocab_size).get_hparams(model_hparams)

        model = t2t_model.T2TModel(model_hparams, problem_hparams=ph)
        logits = tf.zeros((batch_size, sequence_size, 1, 1, vocab_size))
        feature = tf.ones((batch_size, sequence_size, 1, 1))

        # all-zero weights == zero loss.
        weights = tf.zeros((batch_size, sequence_size))
        loss_num, loss_denom = model._loss_single(
            logits, "targets", feature, weights=weights)
        self.assertAllClose(tf.zeros_like(loss_num), sess.run(loss_num))
        self.assertAllClose(tf.zeros_like(loss_denom), sess.run(loss_denom))

        # non-zero weights > zero loss.
        weights = tf.ones((batch_size, sequence_size))
        loss_num, loss_denom = model._loss_single(
            logits, "targets", feature, weights=weights)
        self.assertAllLess(0.0, sess.run(loss_num))
        self.assertAllClose(batch_size * sequence_size, sess.run(loss_denom))


if __name__ == "__main__":
  tf.test.main()
