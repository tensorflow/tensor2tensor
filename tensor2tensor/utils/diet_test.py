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

"""Tests for common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.utils import diet

import tensorflow as tf


class DietVarTest(tf.test.TestCase):

  def testDiet(self):

    params = diet.diet_adam_optimizer_params()

    @diet.fn_with_diet_vars(params)
    def model_fn(x):
      y = tf.layers.dense(x, 10, use_bias=False)
      return y

    @diet.fn_with_diet_vars(params)
    def model_fn2(x):
      y = tf.layers.dense(x, 10, use_bias=False)
      return y

    x = tf.random_uniform((10, 10))
    y = model_fn(x) + 10.
    y = model_fn2(y) + 10.
    grads = tf.gradients(y, [x])
    with tf.control_dependencies(grads):
      incr_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)

    train_op = tf.group(incr_step, *grads)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      orig_vals = sess.run(tf.global_variables())
      for _ in range(10):
        sess.run(train_op)
      new_vals = sess.run(tf.global_variables())

      different = []
      for old, new in zip(orig_vals, new_vals):
        try:
          self.assertAllClose(old, new)
        except AssertionError:
          different.append(True)
      self.assertEqual(len(different), len(tf.global_variables()))


if __name__ == "__main__":
  tf.test.main()
