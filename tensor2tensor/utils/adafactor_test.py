# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

"""Tests for adafactor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import adafactor

import tensorflow as tf


class AdafactorTest(tf.test.TestCase):

  def testCallableLearningRate(self):
    def lr():
      return 0.01

    opt = adafactor.AdafactorOptimizer(learning_rate=lr)
    v1 = tf.Variable([1., 2.])
    v2 = tf.Variable([3., 4.])
    with tf.GradientTape() as tape:
      tape.watch([v1, v2])
      loss = v1 * v2
    v1_grad, v2_grad = tape.gradient(loss, [v1, v2])
    opt.apply_gradients(((v1_grad, v1), (v2_grad, v2)))


if __name__ == '__main__':
  tf.test.main()
