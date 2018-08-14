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
"""Vqa_attention_baseline tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models.research import vqa_attention
from tensor2tensor.utils import registry

import tensorflow as tf


class VqaAttentionBaselineTest(tf.test.TestCase):

  def testVqaAttentionBaseline(self):

    batch_size = 3
    image_size = 448
    vocab_size = 100
    num_classes = 10
    question_length = 5
    answer_length = 10
    x = 2 * np.random.rand(batch_size, image_size, image_size, 3) - 1
    q = np.random.random_integers(
        1, high=vocab_size - 1, size=(batch_size, question_length, 1, 1))
    a = np.random.random_integers(
        0, high=num_classes, size=(batch_size, answer_length, 1, 1))
    hparams = vqa_attention.vqa_attention_base()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size)
    p_hparams.input_modality["inputs"] = (registry.Modalities.IMAGE, None)
    p_hparams.input_modality["question"] = (registry.Modalities.SYMBOL,
                                            vocab_size)
    p_hparams.target_modality = (registry.Modalities.CLASS_LABEL
                                 + ":multi_label", num_classes + 1)
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.float32),
          "question": tf.constant(q, dtype=tf.int32),
          "targets": tf.constant(a, dtype=tf.int32),
      }
      model = vqa_attention.VqaAttentionBaseline(
          hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, losses = model(features)
      session.run(tf.global_variables_initializer())
      logits_, losses_ = session.run([logits, losses])

    self.assertEqual(logits_.shape, (batch_size, 1, 1, 1, num_classes + 1))
    self.assertEqual(losses_["training"].shape, ())


if __name__ == "__main__":
  tf.test.main()
