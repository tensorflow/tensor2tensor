# Copyright 2017 Google Inc.
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

"""Tests for common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from tensor2tensor.models import common_attention

import tensorflow as tf


class CommonAttentionTest(tf.test.TestCase):

  def testLocalAttention(self):
    #q = np.array([[[ [1.0, 0.0, 0.0, 0.0],
    #        [1.0, 0.0, 0.0, 0.0],
    #        [1.0, 0.0, 0.0, 0.0],
    #        [1.0, 0.0, 0.0, 0.0],
    #        [1.0, 0.0, 0.0, 0.0],
    #        [1.0, 0.0, 0.0, 0.0],
    #        [1.0, 0.0, 0.0, 0.0],
    #        [1.0, 0.0, 0.0, 0.0] ]]])
    #k = np.array([[[ [0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0],
    #        [0.0, 0.0, 0.0, 0.0] ]]])
    #v = np.ones((1, 1, 8, 1))

    q = np.random.rand(5, 7, 13, 3)
    k = np.random.rand(5, 7, 13, 3)
    v = np.random.rand(5, 7, 13, 11)

    with self.test_session() as session:
      q_ = tf.constant(q)
      k_ = tf.constant(k)
      v_ = tf.constant(v)
      y = common_attention.masked_local_attention_1d(q_, k_, v_, block_length=tf.constant(3))
      res = session.run(y)
    self.assertEqual(res.shape, (5, 7, 13, 11))


if __name__ == "__main__":
  tf.test.main()
