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

"""Tests for Gene Expression models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import gene_expression as gene_data
from tensor2tensor.layers import modalities  # pylint: disable=unused-import
from tensor2tensor.models import gene_expression

import tensorflow as tf


def gene_expression_conv_test():
  hparams = gene_expression.gene_expression_conv_base()
  hparams.hidden_size = 8
  hparams.num_dconv_layers = 2
  return hparams


class GeneExpressionModelsTest(tf.test.TestCase):

  def _testModel(self, hparams, model_cls):
    batch_size = 3
    target_length = 6
    target_out = 10  # GeneExpressionProblem.num_output_predictions
    input_length = target_length * 128 // 4  # chunk_size=4
    input_vocab_size = 5

    inputs = np.random.random_integers(
        input_vocab_size, size=(batch_size, input_length, 1, 1))
    targets = np.random.random_sample((batch_size, target_length, 1,
                                       target_out))

    features = {
        "inputs": tf.constant(inputs, dtype=tf.int32),
        "targets": tf.constant(targets, dtype=tf.float32),
    }
    p_hparams, = hparams.problems
    sharded_logits, _ = model_cls(hparams, tf.estimator.ModeKeys.TRAIN,
                                  p_hparams).model_fn(features)
    logits = tf.concat(sharded_logits, 0)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      res = sess.run(logits)

    self.assertEqual(res.shape, (batch_size, target_length, 1, target_out))

  def testGeneExpressionModels(self):
    models_hparams = [(gene_expression.GeneExpressionConv,
                       gene_expression_conv_test())]
    for model_cls, hparams in models_hparams:
      hparams.add_hparam("data_dir", None)
      p_hparams = gene_data.GenomicsExpressionCage10().get_hparams(hparams)
      hparams.problems = [p_hparams]
      self._testModel(hparams, model_cls)


if __name__ == "__main__":
  tf.test.main()
