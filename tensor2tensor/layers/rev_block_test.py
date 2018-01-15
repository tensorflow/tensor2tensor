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

  def testForwardBackward(self):

    def f(x):
      return tf.layers.dense(x, self.CHANNELS // 2, use_bias=True)

    def g(x):
      return tf.layers.dense(x, self.CHANNELS // 2, use_bias=True)

    x = tf.random_uniform([self.BATCH_SIZE, self.CHANNELS], dtype=tf.float32)
    x1, x2 = tf.split(x, 2, axis=-1)

    block = rev_block.RevBlock(f, g, num_layers=3)
    y1, y2 = block.forward(x1, x2)
    x1_inv, x2_inv = block.backward(y1, y2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      x1, x2, x1_inv, x2_inv = sess.run([x1, x2, x1_inv, x2_inv])

      self.assertAllClose(x1, x1_inv)
      self.assertAllClose(x2, x2_inv)

  def testBackwardForward(self):

    def f(x):
      return tf.layers.dense(x, self.CHANNELS // 2, use_bias=True)

    def g(x):
      return tf.layers.dense(x, self.CHANNELS // 2, use_bias=True)

    y = tf.random_uniform([self.BATCH_SIZE, self.CHANNELS], dtype=tf.float32)
    y1, y2 = tf.split(y, 2, axis=-1)

    block = rev_block.RevBlock(f, g, num_layers=3)
    x1, x2 = block.backward(y1, y2)
    y1_inv, y2_inv = block.forward(x1, x2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      y1, y2, y1_inv, y2_inv = sess.run([y1, y2, y1_inv, y2_inv])

      self.assertAllClose(y1, y1_inv)
      self.assertAllClose(y2, y2_inv)

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

    if x is None:
      x = tf.random_uniform([self.BATCH_SIZE, self.CHANNELS], dtype=tf.float32)
    x1, x2 = tf.split(x, 2, axis=-1)

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

  # TODO(rsepassi): Recent change to conv seems to have broken this test. Find
  # out why.
  def _testConvAndBatchNorm(self):

    x = tf.random_uniform(
        [self.BATCH_SIZE, 10, self.CHANNELS], dtype=tf.float32)

    def f(x):
      x = tf.layers.conv1d(x, self.CHANNELS // 2, 3, padding="same")
      x = tf.layers.batch_normalization(x, training=True)
      x = tf.layers.conv1d(x, self.CHANNELS // 2, 3, padding="same")
      x = tf.layers.batch_normalization(x, training=True)
      return x

    self._testRevBlock(x=x, f=f)


class RecomputeTest(tf.test.TestCase):

  def testRecompute(self):

    def layer(x, name=None):
      with tf.variable_scope(name, default_name="layer"):
        x = tf.contrib.layers.layer_norm(x)
        x = tf.layers.conv1d(
            x,
            10,
            1,
            use_bias=False,
            kernel_initializer=tf.constant_initializer(42.42))
        x = tf.nn.relu(x)
        return x

    def fn(x):
      out = x
      for _ in range(3):
        out = layer(out)
      return out

    @rev_block.recompute_grad
    def fn_recompute(x):
      return fn(x)

    x = tf.random_uniform((3, 1, 3))
    recompute_vars = None
    with tf.variable_scope("recompute") as vs:
      out1 = tf.reduce_sum(fn_recompute(x))
      recompute_vars = vs.trainable_variables()
    reg_vars = None
    with tf.variable_scope("regular") as vs:
      out2 = tf.reduce_sum(fn(x))
      reg_vars = vs.trainable_variables()

    grad1 = tf.gradients(out1, recompute_vars)
    grad2 = tf.gradients(out2, reg_vars)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outs = sess.run([out1, out2, grad1, grad2])
      self.assertAllClose(outs[0], outs[1])
      for g1, g2 in zip(outs[2], outs[3]):
        self.assertAllClose(g1, g2)


if __name__ == "__main__":
  tf.test.main()
