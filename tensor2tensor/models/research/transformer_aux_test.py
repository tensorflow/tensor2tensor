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

"""Tests for tensor2tensor.models.research.transformer_aux."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models.research import transformer_aux
import tensorflow.compat.v1 as tf


class TransformerAuxTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      dict(
          tensor=np.array(
              [1, 2, 3, 4]
          ),
          shift=0,
          axis=0,
          target=np.array(
              [1, 2, 3, 4]
          ),
      ),
      dict(
          tensor=np.array(
              [1, 2, 3, 4]
          ),
          shift=2,
          axis=0,
          target=np.array(
              [0, 0, 1, 2]
          ),
      ),
      dict(
          tensor=np.array(
              [1, 2, 3, 4]
          ),
          shift=-2,
          axis=0,
          target=np.array(
              [3, 4, 0, 0]
          ),
      ),
      dict(
          tensor=np.array(
              [[1, 2, 3, 4],
               [5, 6, 7, 8]]
          ),
          shift=2,
          axis=1,
          target=np.array(
              [[0, 0, 1, 2],
               [0, 0, 5, 6]]
          ),
      ),
  )
  def test_shift_and_pad(self, tensor, shift, axis, target):
    with self.test_session() as session:
      output = transformer_aux.shift_and_pad(tensor, shift, axis)
      output_val = session.run(output)
      self.assertAllEqual(output_val, target)

  def test_transformer_aux_body(self):
    batch_size = 3
    input_length = 5
    target_length = 16
    vocab_size = 9
    hparams = transformer_aux.transformer_aux_tiny()
    hparams.shift_values = "-5,1,2,3"
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
    model = transformer_aux.TransformerAux(hparams, tf.estimator.ModeKeys.TRAIN,
                                           p_hparams)
    logits, losses = model(features)

    self.assertIn("training", losses)
    self.assertIn("auxiliary", losses)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      logits_val = session.run(logits)
      self.assertEqual(logits_val.shape,
                       (batch_size, target_length, 1, 1, vocab_size))


if __name__ == "__main__":
  tf.test.main()
