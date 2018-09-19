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

import os
import tempfile
import numpy as np
from tensor2tensor.models.research import glow
from tensor2tensor.models.research import glow_ops
import tensorflow as tf

arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope


arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope


class GlowOpsTest(tf.test.TestCase):

  def test_get_variable_ddi(self):
    with tf.Graph().as_default():
      x_t = tf.random_normal((5, 5))
      ddi = glow_ops.get_variable_ddi(
          "x", (5, 5), initial_value=x_t, init=True)
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

      with arg_scope([glow_ops.actnorm], init=True):
        actnorm_conv2d = glow_ops.conv2d(
            "actnorm_conv2d", x, output_channels=64, apply_actnorm=True)
        actnorm_zeros2d = glow_ops.conv2d(
            "actnorm_zeros2d", x, output_channels=64, apply_actnorm=False)

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

  def check_tensor_to_dist(self, architecture):
    with tf.Graph().as_default():
      x = tf.random_uniform(shape=(16, 5, 5, 32))
      x_prior = glow_ops.tensor_to_dist("split_prior", x,
                                        architecture=architecture,
                                        output_channels=64)
      mean_t, scale_t = x_prior.loc, x_prior.scale
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        mean, scale = session.run([mean_t, scale_t])
        self.assertEqual(mean.shape, (16, 5, 5, 64))
        self.assertEqual(scale.shape, (16, 5, 5, 64))
        self.assertTrue(np.allclose(mean, 0.0))
        self.assertTrue(np.allclose(scale, 1.0))

  def test_tensor_to_dist(self):
    for architecture in ["single_conv", "glow_nn"]:
      self.check_tensor_to_dist(architecture)

  def test_split(self):
    with tf.Graph().as_default():
      x = tf.random_uniform(shape=(16, 5, 5, 32))
      x_inv, _, eps, z = glow_ops.split("split", x)
      x_inv_inv, _ = glow_ops.split("split", x_inv, reverse=True, eps=eps)
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        x_inv_np, diff, z_np = session.run([x_inv, x - x_inv_inv, z])
        self.assertEqual(z_np.shape, (16, 5, 5, 16))
        self.assertEqual(x_inv_np.shape, (16, 5, 5, 16))
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-5))

  def check_revnet_reversibility(self, op, name):
    with tf.Graph().as_default():
      hparams = glow.glow_hparams()
      hparams.depth = 2
      x = tf.random_uniform(shape=(16, 32, 32, 4), seed=0)
      x_inv, _ = op(name, x, hparams, reverse=False)
      x_inv_inv, _ = op(name, x_inv, hparams, reverse=True)
      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        diff = session.run(x - x_inv_inv)
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-2))

  def test_revnet_reversibility(self):
    ops = [glow_ops.revnet_step, glow_ops.revnet]
    names = ["revnet_step", "revnet"]
    for op, name in zip(ops, names):
      self.check_revnet_reversibility(op, name)

  def test_encoder_decoder(self):
    with tf.Graph().as_default():
      hparams = glow.glow_hparams()
      hparams.n_levels = 3
      hparams.depth = 2

      x = tf.random_uniform(shape=(16, 64, 64, 4), seed=0)
      x_inv, _, eps, z_levels = glow_ops.encoder_decoder(
          "encoder_decoder", x, hparams, reverse=False)
      x_inv_inv, _, z_inv_levels = glow_ops.encoder_decoder(
          "encoder_decoder", x_inv, hparams, eps=eps, reverse=True)

      with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        diff, x_inv_np, z_levels_np, z_inv_levels_np = session.run(
            [x - x_inv_inv, x_inv, z_levels, z_inv_levels])

        self.assertEqual(len(z_levels_np), 2)
        self.assertEqual(len(z_inv_levels_np), 2)
        # (h_i, w_i, c_i) = (h_{i-1}/f, w_{i-1}/f, c_{i-1}*(2f)/2) where (f=2)
        self.assertEqual(z_levels_np[0].shape, (16, 32, 32, 8))
        self.assertEqual(z_levels_np[1].shape, (16, 16, 16, 16))
        self.assertEqual(z_inv_levels_np[0].shape, (16, 32, 32, 8))
        self.assertEqual(z_inv_levels_np[1].shape, (16, 16, 16, 16))
        self.assertTrue(x_inv_np.shape, (16, 8, 8, 64))
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-2))

  def test_encoder_decoder_practical_usage(self):
    """Tests the following sequence of operations.

    1. Define forward network with arg_scope(init=True).
    2. Run one-forward pass to do data-dependent initialization and save.
    3. Define forward and reverse network with arg_scope(init=False)
    4. Check that reverse(forward(x)) == x
    """
    hparams = glow.glow_hparams()
    hparams.n_levels = 2
    hparams.depth = 12

    with tf.Graph().as_default():
      rng = np.random.RandomState(0)
      x_rand = np.asarray(rng.rand(1, 4, 4, 4), dtype=np.float32)
      x_t = tf.convert_to_tensor(x_rand)

      ops = [glow_ops.get_variable_ddi, glow_ops.actnorm]
      with arg_scope(ops, init=True):
        x_inv, _, _, _ = glow_ops.encoder_decoder(
            "revnet", x_t, hparams, reverse=False)
      curr_dir = tempfile.mkdtemp()
      model_path = os.path.join(curr_dir, "model")

      with tf.Session() as session:
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        session.run(x_inv)
        saver.save(session, model_path)

    with tf.Graph().as_default():
      rng = np.random.RandomState(0)
      x_rand = np.asarray(rng.rand(1, 4, 4, 4), dtype=np.float32)
      x_t = tf.convert_to_tensor(x_rand)
      ops = [glow_ops.get_variable_ddi, glow_ops.actnorm]
      with arg_scope(ops, init=False):
        x_inv2, _, all_eps, _ = glow_ops.encoder_decoder(
            "revnet", x_t, hparams, reverse=False)
        x_inv_inv_, _, _ = glow_ops.encoder_decoder(
            "revnet", x_inv2, hparams, eps=all_eps, reverse=True)

      with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, model_path)
        x_inv_inv_np = session.run(x_inv_inv_)
        diff = np.abs(x_inv_inv_np - x_rand)
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-3))

  def test_scale_gaussian_prior(self):
    with tf.Graph().as_default():
      rng = np.random.RandomState(0)
      img_shape = (16, 2, 2, 2)
      x_rand = np.asarray(rng.randint(0, 10, img_shape), dtype=np.float32)
      z_rand = np.asarray(rng.randint(0, 10, img_shape), dtype=np.float32)
      x_t = tf.convert_to_tensor(x_rand)
      z_t = tf.convert_to_tensor(z_rand)
      dist = glow_ops.scale_gaussian_prior(
          "scale_gaussian_prior", z_t, x_t, trainable=True)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mean, scale = sess.run([dist.loc, dist.scale])
        self.assertTrue(np.allclose(mean, z_rand))
        self.assertTrue(np.allclose(scale, 1.0))

  def check_split_latent_conditioning(self, merge_std):
    with tf.Graph().as_default():
      rng = np.random.RandomState(0)
      x_rand = rng.randn(12, 32, 32, 32).astype(np.float32)
      latent_rand = rng.randn(12, 32, 32, 16).astype(np.float32)
      x_t = tf.convert_to_tensor(x_rand)
      latent_t = tf.convert_to_tensor(latent_rand)
      hparams = glow.glow_hparams()
      hparams.level_scale = merge_std
      hparams.add_hparam("latent_dist_encoder", "pointwise")

      # Test initalization.
      # x2 ~ N(scale * latent, 1.0) where initial scale is 1.0
      exp_x2 = x_rand[:, :, :, 16:]
      exp_eps = x_rand[:, :, :, 16:] - latent_rand
      x_inv, _, eps, x2_t = glow_ops.split(
          merge_std, x_t, cond_latents=latent_t, hparams=hparams)
      # Test reversibility.
      x_inv_inv, _ = glow_ops.split(
          merge_std, x_inv, cond_latents=latent_t, eps=eps, reverse=True,
          hparams=hparams)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actual_eps, actual_x2, diff_np = sess.run([eps, x2_t, x_inv_inv - x_t])
        self.assertTrue(np.allclose(diff_np, 0.0, atol=1e-5))
        self.assertTrue(np.allclose(actual_eps, exp_eps))
        self.assertTrue(np.allclose(exp_x2, actual_x2))

  def test_split_latent_conditioning(self):
    for merge_std in ["normal", "prev_level", "prev_step"]:
      self.check_split_latent_conditioning(merge_std)


if __name__ == "__main__":
  tf.test.main()
