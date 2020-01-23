# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Tests for tensor2tensor.models.research.transformer_vae."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models.research import transformer_vae
import tensorflow.compat.v1 as tf


class TransformerVaeTest(tf.test.TestCase):

  def testTransformerAEOnDVQ(self):
    batch_size = 3
    input_length = 5
    target_length = 16
    vocab_size = 9
    hparams = transformer_vae.transformer_ae_small()
    hparams.bottleneck_kind = "dvq"
    hparams.dp_strength = 0
    p_hparams = problem_hparams.test_problem_hparams(vocab_size,
                                                     vocab_size,
                                                     hparams)
    hparams.problem_hparams = p_hparams
    inputs = np.random.randint(
        vocab_size, size=(batch_size, input_length, 1, 1))
    targets = np.random.randint(
        vocab_size, size=(batch_size, target_length, 1, 1))
    features = {
        "inputs": tf.constant(inputs, dtype=tf.int32),
        "targets": tf.constant(targets, dtype=tf.int32),
        "target_space_id": tf.constant(1, dtype=tf.int32),
    }
    tf.train.create_global_step()
    model = transformer_vae.TransformerAE(hparams, tf.estimator.ModeKeys.TRAIN,
                                          p_hparams)
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      logits_val = session.run(logits)
      self.assertEqual(logits_val.shape,
                       (batch_size, target_length, 1, 1, vocab_size))


if __name__ == "__main__":
  tf.test.main()
