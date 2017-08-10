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

"""Tests for RevBlock."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import rev_block

import tensorflow as tf


class RevBlockTest(tf.test.TestCase):

  def testSmoke(self):
    channels = 8
    num_layers = 4
    batch_size = 16
    use_defun = True
    tf.set_random_seed(1234)

    def f(x):
      return tf.layers.dense(x, channels // 2, use_bias=True)

    def g(x):
      return tf.layers.dense(x, channels // 2, use_bias=True)

    x = tf.random_uniform([batch_size, channels], dtype=tf.float32)
    y = rev_block.rev_block(
        x, f, g, num_layers=num_layers, is_training=use_defun)
    loss = tf.reduce_mean(y + 10.)
    grads = tf.gradients(loss, [x] + tf.global_variables())
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      _ = sess.run(grads)

  def testRevBlock(self):
    channels = 8
    num_layers = 4
    batch_size = 16
    tf.set_random_seed(1234)

    def f(x):
      return tf.layers.dense(x, channels // 2, use_bias=True)

    def g(x):
      return tf.layers.dense(x, channels // 2, use_bias=True)

    x = tf.random_uniform([batch_size, channels], dtype=tf.float32)

    with tf.variable_scope("defun") as vs:
      y_defun = rev_block.rev_block(x, f, g, num_layers=num_layers)
      fg_vars = vs.trainable_variables()

    num_vars = len(tf.global_variables())
    with tf.variable_scope(vs, reuse=True):
      y = rev_block.rev_block(x, f, g, num_layers=num_layers, is_training=False)
    # Ensure no new vars were created - full reuse
    assert len(tf.global_variables()) == num_vars

    loss_defun = tf.reduce_mean(y_defun + 10.)
    loss = tf.reduce_mean(y + 10.)

    grads_defun = tf.gradients(loss_defun, [x] + fg_vars)
    grads = tf.gradients(loss, [x] + fg_vars)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      y_val, yd_val, gd_val, g_val = sess.run([y, y_defun, grads_defun, grads])
      self.assertAllClose(y_val, yd_val)
      for g1, g2 in zip(gd_val, g_val):
        self.assertAllClose(g1, g2)


if __name__ == "__main__":
  tf.test.main()
