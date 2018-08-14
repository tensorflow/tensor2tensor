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
"""Tests for tensor2tensor.models.research.glow_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.models.research import glow_ops
import tensorflow as tf


class GlowOpsTest(tf.test.TestCase):

  def test_get_variable_ddi(self):
    with tf.Graph().as_default():
      x_t = tf.random_normal((5, 5))
      ddi = glow_ops.get_variable_ddi(
          "x", (5, 5), x_t, init=True)
      with tf.Session() as session:
        diff = ddi - x_t
        self.assertTrue(np.allclose(session.run(diff), 0.0))

  def test_actnorm(self):
    """Test that actnorm provides activations with zero channel-mean."""
    with tf.Graph().as_default():
      x_t = tf.random_normal((16, 32, 32, 3), mean=50.0, stddev=2.0)
      x_act = glow_ops.actnorm("actnorm", x_t, init=True)
      with tf.Session() as session:
        x_act_np, _ = session.run(x_act)
        channel_mean = np.mean(x_act_np, axis=(0, 1, 2))
        channel_var = np.var(x_act_np, axis=(0, 1, 2))
        self.assertTrue(np.allclose(channel_mean, 0.0, atol=1e-3))
        self.assertTrue(np.allclose(channel_var, 1.0, atol=1e-3))

  def check_invertibility(self, op, name):
    with tf.Graph().as_default():
      x = tf.random_uniform(shape=(16, 32, 32, 4))

      x_inv, _ = op(name, x, reverse=False)
      x_inv_inv, _ = op(name, x_inv, reverse=True)
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        diff = session.run(x - x_inv_inv)
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-5))

  def test_invertibility(self):
    rev_ops = [glow_ops.invertible_1x1_conv, glow_ops.affine_coupling,
               glow_ops.actnorm]
    names = ["inv_1X1_conv", "affine_coupling", "actnorm"]
    for rev_op, name in zip(rev_ops, names):
      self.check_invertibility(rev_op, name)

  def test_add_edge_bias(self):
    with tf.Graph().as_default():
      x = tf.random_uniform(shape=(16, 32, 32, 3))
      x_pad = glow_ops.add_edge_bias(x, [3, 3])
      with tf.Session() as session:
        x_pad_np = session.run(x_pad)

        # Test expected output shape.
        self.assertEqual(x_pad_np.shape, (16, 34, 34, 4))

  def test_conv2d(self):
    with tf.Graph().as_default():
      x = 10.0 * tf.random_uniform(shape=(16, 5, 5, 32))
      actnorm_conv2d = glow_ops.conv2d(
          "actnorm_conv2d", x, output_channels=64, init=True,
          apply_actnorm=True)
      actnorm_zeros2d = glow_ops.conv2d(
          "actnorm_zeros2d", x, output_channels=64, init=True,
          apply_actnorm=False)
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # test if apply_actnorm is set to True, the first minibatch has
        # zero mean and unit variance.
        actnorm_np, zeros_np = session.run([actnorm_conv2d, actnorm_zeros2d])
        self.assertEqual(actnorm_np.shape, (16, 5, 5, 64))
        mean = np.mean(actnorm_np, axis=(0, 1, 2))
        var = np.var(actnorm_np, axis=(0, 1, 2))
        self.assertTrue(np.allclose(mean, 0.0, atol=1e-5))
        self.assertTrue(np.allclose(var, 1.0, atol=1e-5))

        # test shape in case apply_actnorm is set to False,
        self.assertEqual(zeros_np.shape, (16, 5, 5, 64))

  def test_nn(self):
    """Test output shape."""
    with tf.Graph().as_default():
      x = 10.0 * tf.random_uniform(shape=(16, 5, 5, 32))
      nn = glow_ops.nn("nn", x, 512, 64)

      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        nn_np = session.run(nn)
        self.assertEqual(nn_np.shape, (16, 5, 5, 64))

        # Initialized with zeros.
        self.assertTrue(np.allclose(nn_np, 0.0))

  def test_split_prior(self):
    with tf.Graph().as_default():
      x = tf.random_uniform(shape=(16, 5, 5, 32))
      x_prior = glow_ops.split_prior("split_prior", x)
      mean_t, scale_t = x_prior.loc, x_prior.scale
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        mean, scale = session.run([mean_t, scale_t])
        self.assertTrue(np.allclose(mean, 0.0))
        self.assertTrue(np.allclose(scale, 1.0))

  def test_split(self):
    with tf.Graph().as_default():
      x = tf.random_uniform(shape=(16, 5, 5, 32))
      x_inv, _, eps = glow_ops.split("split", x)
      x_inv_inv = glow_ops.split("split", x_inv, reverse=True, eps=eps)
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        x_inv_np, diff = session.run([x_inv, x - x_inv_inv])
        self.assertEqual(x_inv_np.shape, (16, 5, 5, 16))
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-5))

  def check_revnet_reversibility(self, op, name):
    with tf.Graph().as_default():
      hparams = glow_ops.glow_hparams()
      hparams.depth = 2
      x = tf.random_uniform(shape=(16, 32, 32, 4), seed=0)
      x_inv, _ = op(name, x, hparams, reverse=False)
      x_inv_inv, _ = op(name, x_inv, hparams, reverse=True)
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        diff = session.run(x - x_inv_inv)
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-3))

  def test_revnet_reversibility(self):
    ops = [glow_ops.revnet_step, glow_ops.revnet]
    names = ["revnet_step", "revnet"]
    for op, name in zip(ops, names):
      self.check_revnet_reversibility(op, name)

  def test_encoder_decoder(self):
    with tf.Graph().as_default():
      hparams = glow_ops.glow_hparams()
      hparams.n_levels = 2
      hparams.depth = 2

      x = tf.random_uniform(shape=(16, 64, 64, 4), seed=0)
      x_inv, _, eps = glow_ops.encoder_decoder(
          "encoder_decoder", x, hparams, reverse=False)
      x_inv_inv, _ = glow_ops.encoder_decoder(
          "encoder_decoder", x_inv, hparams, eps=eps, reverse=True)

      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        diff, x_inv_np = session.run([x - x_inv_inv, x_inv])
        self.assertTrue(x_inv_np.shape, (16, 8, 8, 64))
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-2))


if __name__ == "__main__":
  tf.test.main()
