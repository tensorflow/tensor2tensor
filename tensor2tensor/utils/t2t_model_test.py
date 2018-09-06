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
"""Tests for T2TModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import t2t_model

import tensorflow as tf


class T2TModelTest(tf.test.TestCase):

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testSummarizeLosses(self):
    with tf.Graph().as_default():
      model = t2t_model.T2TModel(tf.contrib.training.HParams())
      losses = {"training": tf.random_normal([]),
                "extra": tf.random_normal([])}
      outputs = model._summarize_losses(losses)
      self.assertIsNone(outputs, None)
      self.assertLen(tf.get_collection(tf.GraphKeys.SUMMARIES, scope="losses"),
                     len(losses))

if __name__ == "__main__":
  tf.test.main()
