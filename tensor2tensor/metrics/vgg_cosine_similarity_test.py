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

"""Tests for tensor2tensor.metrics.vgg_cosine_similarity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.metrics import vgg_cosine_similarity
import tensorflow as tf


class VggCosineSimilarityTest(tf.test.TestCase):

  def test_vgg_cosine_similarity(self):
    with tf.Graph().as_default():
      rng = np.random.RandomState(0)
      x = np.asarray(rng.randn(16, 64, 64, 3), dtype=np.float32)
      cos_sim_t = vgg_cosine_similarity.vgg_cosine_similarity(x, x)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        cos_sim_np = sess.run(cos_sim_t)
        self.assertTrue(np.allclose(cos_sim_np, 1.0))


if __name__ == '__main__':
  tf.test.main()
