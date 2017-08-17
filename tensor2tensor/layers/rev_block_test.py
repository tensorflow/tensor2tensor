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
  CHANNELS = 8
  NUM_LAYERS = 4
  BATCH_SIZE = 16

  def _testRevBlock(self,
                    x=None,
                    f=None,
                    g=None,
                    f_side_input=None,
                    g_side_input=None):
    tf.set_random_seed(1234)

    if f is None:

      def f(x):  # pylint: disable=function-redefined
        return tf.layers.dense(x, self.CHANNELS // 2, use_bias=True)

    if g is None:

      def g(x):  # pylint: disable=function-redefined
        return tf.layers.dense(x, self.CHANNELS // 2, use_bias=True)

    if f_side_input is None:
      f_side_input = []

    if g_side_input is None:
      g_side_input = []

    x = tf.random_uniform([self.BATCH_SIZE, self.CHANNELS], dtype=tf.float32)
    x1, x2 = tf.split(x, 2, axis=1)

    with tf.variable_scope("rev_test") as vs:
      y1_rev, y2_rev = rev_block.rev_block(
          x1,
          x2,
          f,
          g,
          f_side_input=f_side_input,
          g_side_input=g_side_input,
          num_layers=self.NUM_LAYERS)
      y_rev = tf.concat([y1_rev, y2_rev], axis=1)
      fg_vars = vs.trainable_variables()

    num_vars = len(tf.global_variables())
    with tf.variable_scope(vs, reuse=True):
      y1, y2 = rev_block.rev_block(
          x1,
          x2,
          f,
          g,
          f_side_input=f_side_input,
          g_side_input=g_side_input,
          num_layers=self.NUM_LAYERS,
          is_training=False)
      y = tf.concat([y1, y2], axis=1)
    # Ensure no new vars were created - full reuse
    assert len(tf.global_variables()) == num_vars

    loss_rev = tf.reduce_mean(y_rev + 10.)
    loss = tf.reduce_mean(y + 10.)

    wrt = [x] + f_side_input + g_side_input + fg_vars
    grads_rev = tf.gradients(loss_rev, wrt)
    grads = tf.gradients(loss, wrt)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      y_val, yd_val, gd_val, g_val = sess.run([y, y_rev, grads_rev, grads])
      self.assertAllClose(y_val, yd_val)
      for g1, g2 in zip(gd_val, g_val):
        self.assertAllClose(g1, g2)

  def testRevBlock(self):
    self._testRevBlock()

  def testSideInput(self):
    f_side_input = tf.random_uniform([self.BATCH_SIZE, self.CHANNELS // 2])

    def f(x, side_input):
      return tf.layers.dense(
          x, self.CHANNELS // 2, use_bias=True) + side_input[0]

    self._testRevBlock(f=f, f_side_input=[f_side_input])

  def testMultipleFns(self):

    def f1(x):
      return tf.layers.dense(x, self.CHANNELS // 2)

    def f2(x):
      return tf.layers.dense(x, self.CHANNELS // 2, activation=tf.nn.relu)

    self._testRevBlock(f=[f1, f2, f1, f2])


class FnWithCustomGradTest(tf.test.TestCase):

  def testCorrectness(self):

    w = tf.random_uniform([6, 10])

    def fn(a, b, c):
      return tf.layers.dense(
          a,
          10,
          use_bias=False,
          kernel_initializer=lambda shape, dtype, partition_info: w
      ) + tf.matmul(b, c)

    def grad_fn(inputs, variables, outputs, grad_outputs):
      outputs = outputs[0]
      grad_outputs = grad_outputs[0]
      grad_inputs = tf.gradients(outputs, inputs, grad_ys=grad_outputs)
      grad_vars = tf.gradients(outputs, variables, grad_ys=grad_outputs)
      return grad_inputs, grad_vars

    custom_fn = rev_block.fn_with_custom_grad(grad_fn)(fn)

    a = tf.random_uniform([11, 6])
    b = tf.random_uniform([11, 7])
    c = tf.random_uniform([7, 10])

    out = fn(a, b, c)
    custom_out = custom_fn(a, b, c)
    self.assertEqual(out.get_shape().as_list(),
                     custom_out.get_shape().as_list())

    loss = tf.reduce_mean(out)
    custom_loss = tf.reduce_mean(custom_out)

    grads = tf.gradients(loss, [a, b, c] + [tf.trainable_variables()[0]])
    custom_grads = tf.gradients(custom_loss,
                                [a, b, c] + [tf.trainable_variables()[1]])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out_val, custom_out_val, grads_val, custom_grads_val = sess.run(
          [out, custom_out, grads, custom_grads])
      self.assertAllClose(out_val, custom_out_val)
      for g1, g2 in zip(grads_val, custom_grads_val):
        self.assertAllClose(g1, g2)

  def testCustomGrad(self):

    def fn(a, b, c):
      return tf.layers.dense(a, 10, use_bias=False) + tf.matmul(b, c)

    def grad_fn(inputs, variables, unused_outputs, unused_grad_outputs):
      grad_inputs = [tf.ones_like(t) * (i + 1.) for i, t in enumerate(inputs)]
      grad_vars = [
          tf.ones_like(t) * (i + len(inputs) + 1.)
          for i, t in enumerate(variables)
      ]
      return grad_inputs, grad_vars

    a = tf.random_uniform([11, 6])
    b = tf.random_uniform([11, 7])
    c = tf.random_uniform([7, 10])
    w = tf.random_uniform([6, 10])
    out = rev_block.fn_with_custom_grad(grad_fn)(fn)(a, b, c)
    loss = tf.reduce_mean(out)
    grads = tf.gradients(loss, [a, b, c, tf.trainable_variables()[0]])
    expected_grads = [
        tf.ones_like(t) * (i + 1.) for i, t in enumerate([a, b, c, w])
    ]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      g_val, eg_val = sess.run([grads, expected_grads])
      for g1, g2 in zip(g_val, eg_val):
        self.assertAllClose(g1, g2)


if __name__ == "__main__":
  tf.test.main()
