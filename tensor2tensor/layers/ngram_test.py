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

"""Tests for n-gram layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import ngram

from tensor2tensor.utils import test_utils

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


class NGramTest(tf.test.TestCase):

  @test_utils.run_in_graph_and_eager_modes()
  def testNGramLayerShape(self):
    batch_size = 2
    length = 8
    vocab_size = 3
    minval = 1
    maxval = 4
    inputs = tf.random_uniform(
        [batch_size, length], minval=0, maxval=vocab_size, dtype=tf.int32)
    layer = ngram.NGram(vocab_size, minval, maxval)
    outputs = layer(inputs)
    outputs_val = self.evaluate(outputs)
    num_ngrams = sum([vocab_size**n for n in range(minval, maxval)])
    self.assertEqual(outputs_val.shape, (batch_size, num_ngrams))

  @test_utils.run_in_graph_and_eager_modes()
  def testNGramLayerOutput(self):
    inputs = tf.constant(
        [[0, 0, 0, 0, 1],
         [2, 1, 2, 1, 0]], dtype=tf.int32)
    layer = ngram.NGram(3, minval=1, maxval=3)
    outputs = layer(inputs)
    expected_outputs = tf.constant(
        [[4., 1., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 2., 2., 0., 0., 0., 0., 0., 0., 0., 2., 0.]], dtype=tf.float32)
    outputs_val, expected_outputs_val = self.evaluate(
        [outputs, expected_outputs])
    self.assertAllEqual(outputs_val, expected_outputs_val)

if __name__ == "__main__":
  tf.test.main()

