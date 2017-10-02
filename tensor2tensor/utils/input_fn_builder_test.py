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

"""Tests for tensor2tensor.utils.input_fn_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.utils import input_fn_builder
import tensorflow as tf


class InputFnBuilderTest(tf.test.TestCase):

  def testCondOnIndex(self):
    """Smoke tests of cond_on_index()."""

    z = tf.constant(1., dtype=tf.float32)
    def f(n):
      return {
          "a": z * n,
          "b": z * n * n
      }

    index = tf.placeholder(shape=[], dtype=tf.int32)
    out = input_fn_builder.cond_on_index(f, index, 3, 0)

    with self.test_session() as sess:
      # Check dispatching to the correct branch
      result = sess.run(out, feed_dict={
          index: 2
      })

      self.assertAllClose(result["a"], 2.)
      self.assertAllClose(result["b"], 4.)

      result = sess.run(out, feed_dict={
          index: 3
      })

      self.assertAllClose(result["a"], 3.)
      self.assertAllClose(result["b"], 9.)


if __name__ == "__main__":
  tf.test.main()
