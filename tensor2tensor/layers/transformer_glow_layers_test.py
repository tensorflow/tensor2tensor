# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Tests for tensor2tensor.layers.transformer_glow_layers.

1. Actnorm test (zero mean and unit variance).
2. Invertibility tests for:
  * actnorm
  * actnorm with weight normalization
  * 1x1 invertible convolution
  * multi-head 1x1 invertible convolution
  * affine coupling
  * split
  * 1 step of flow
  * k steps of flow
  * entire pipeline (tested up to 3 levels, 32 steps: tca/tca/ca, 12/12/8)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
from absl.testing import parameterized
import numpy as np

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import transformer_glow_layers as glow
from tensor2tensor.layers import transformer_glow_layers_ops as gops
from tensor2tensor.models import transformer
import tensorflow.compat.v1 as tf

BATCH_SIZE = 20
INPUT_LENGTH = 3
TARGET_LENGTH = 16
N_CHANNELS = 256
HIDDEN_SIZE = 64
N_1X1_HEADS = 4
DTYPE = tf.float32


def float32_bottleneck(x):
  return tf.cast(tf.cast(x, tf.float32), tf.float64)


def get_diff(l1, l2):
  l2 = l2[::-1]
  for i1, i2 in zip(l1, l2):
    print (i1 - i2)
  for i1, i2 in zip(l1, l2):
    print (np.max(np.abs(i1 - i2)))


class TransformerGlowLayersTest(parameterized.TestCase, tf.test.TestCase):

  def get_hparams(self):
    hparams = transformer.transformer_small()
    hparams.add_hparam("prior_type", "affine")
    hparams.add_hparam("factor", 2)  # squeezing factor
    hparams.add_hparam("n_layers_transform_params", 1)
    hparams.add_hparam("n_1x1_heads", N_1X1_HEADS)
    hparams.add_hparam("flow_num_1x1_heads", 4)
    hparams.add_hparam("flow_num_heads", 4)
    hparams.add_hparam("flow_hidden_size", 64)
    hparams.add_hparam("flow_filter_size", 128)
    hparams.add_hparam("flow_layer_prepostprocess_dropout", 0.0)
    hparams.add_hparam("flow_attention_dropout", 0.0)
    hparams.add_hparam("flow_relu_dropout", 0.0)
    hparams.add_hparam("latent_size", N_CHANNELS)
    hparams.add_hparam("use_weightnorm", True)
    hparams.add_hparam("kl_startup_steps", 2000)
    hparams.add_hparam("affine_scale", "glow")
    hparams.add_hparam("scale_width", 0.999)
    hparams.add_hparam("step_fn", "glow")  # glow / chunting
    hparams.add_hparam("conv_fn", "np")  # np / tf
    hparams.add_hparam("posterior_type", "diagonal_normal")
    hparams.causal_decoder_self_attention = False
    hparams.hidden_size = HIDDEN_SIZE
    hparams.weight_dtype = "float32"
    hparams.add_hparam("pos_attn", False)
    return hparams

  def get_data(self):
    x = tf.random_normal(
        (BATCH_SIZE, TARGET_LENGTH, N_CHANNELS), dtype=DTYPE)
    x_lengths = np.random.randint(
        low=1, high=TARGET_LENGTH+1, size=BATCH_SIZE)
    x_lengths = np.ceil(x_lengths / 4.0) * 4.0
    x_lengths = x_lengths.astype(int)
    x_mask = tf.sequence_mask(x_lengths, maxlen=TARGET_LENGTH, dtype=DTYPE)
    return x, x_mask, x_lengths

  def get_kwargs(self, x_mask, hparams=None):
    if hparams is None:
      hparams = self.get_hparams()
    encoder_output = tf.random.uniform(
        (BATCH_SIZE, INPUT_LENGTH, HIDDEN_SIZE), dtype=DTYPE)
    encoder_decoder_attention_bias = tf.zeros(
        (BATCH_SIZE, 1, 1, INPUT_LENGTH), dtype=DTYPE)
    decoder_self_attention_bias = 1.0 - x_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_self_attention_bias *= -1e9
    kwargs = {"hparams": hparams,
              "encoder_output": encoder_output,
              "encoder_decoder_attention_bias": encoder_decoder_attention_bias,
              "decoder_self_attention_bias": decoder_self_attention_bias}
    return kwargs

  def test_actnorm(self):
    _, x_mask, _ = self.get_data()
    x = tf.random_normal((BATCH_SIZE, TARGET_LENGTH, N_CHANNELS),
                         mean=50.0, stddev=10.0, dtype=DTYPE)
    x_act, logabsdet = glow.actnorm(
        "actnorm", x, x_mask, inverse=False, init=True)

    x_act_nopad = tf.boolean_mask(x_act, x_mask)
    x_mean, x_var = tf.nn.moments(x_act_nopad, axes=[0])
    self.evaluate(tf.global_variables_initializer())
    x, x_act, logabsdet, x_mean, x_var = (
        self.evaluate([x, x_act, logabsdet, x_mean, x_var]))
    self.assertEqual(x_act.shape, (BATCH_SIZE, TARGET_LENGTH, N_CHANNELS))
    self.assertEqual(logabsdet.shape, (BATCH_SIZE,))
    self.assertTrue(np.allclose(x_mean, 0.0, atol=1e-5))
    self.assertTrue(np.allclose(x_var, 1.0, atol=1e-5))

  def test_actnorm_invertibility(self):
    name = "actnorm"
    x, x_mask, _ = self.get_data()

    x_inv, logabsdet = glow.actnorm(
        name, x, x_mask, inverse=False, init=False)
    x_inv_inv, logabsdet_inv = glow.actnorm(
        name, x_inv, x_mask, inverse=True, init=False)
    self.evaluate(tf.global_variables_initializer())
    x, x_inv, x_inv_inv, x_mask, logabsdet, logabsdet_inv = (
        self.evaluate(
            [x, x_inv, x_inv_inv, x_mask, logabsdet, logabsdet_inv]))
    diff = x - x_inv_inv
    logabsdet_sum = logabsdet + logabsdet_inv
    self.assertEqual(x.shape, (BATCH_SIZE, TARGET_LENGTH, N_CHANNELS))
    self.assertEqual(x_inv.shape, (BATCH_SIZE, TARGET_LENGTH, N_CHANNELS))
    self.assertEqual(x_inv_inv.shape, (BATCH_SIZE, TARGET_LENGTH, N_CHANNELS))
    self.assertTrue(np.allclose(diff, 0.0, atol=1e-5))
    self.assertTrue(np.allclose(logabsdet_sum, 0.0, atol=1e-5))

  @parameterized.parameters(
      (glow.multihead_invertible_1x1_conv_np, "a"),
      (glow.multihead_invertible_1x1_conv_np, "c"),
      )
  def test_multi_1x1_invertibility(
      self, func, multihead_split):
    name = "multi_1x1"
    x, x_mask, _ = self.get_data()

    x_inv, logabsdet = func(
        name, x, x_mask, multihead_split, inverse=False, dtype=DTYPE)
    x_inv_inv, logabsdet_inv = func(
        name, x_inv, x_mask, multihead_split, inverse=True, dtype=DTYPE)
    self.evaluate(tf.global_variables_initializer())
    x, x_mask, x_inv, x_inv_inv, logabsdet, logabsdet_inv = (
        self.evaluate(
            [x, x_mask, x_inv, x_inv_inv, logabsdet, logabsdet_inv]))
    diff = x - x_inv_inv
    logabsdet_sum = logabsdet + logabsdet_inv
    logabsdet_ = logabsdet / np.sum(x_mask, -1)
    self.assertTrue(np.allclose(diff, 0.0, atol=1e-5))
    self.assertTrue(np.allclose(logabsdet_, 0.0, atol=1e-5))
    self.assertTrue(np.allclose(logabsdet_sum, 0.0, atol=1e-5))

  @parameterized.parameters(
      (glow.additive_coupling, "c"),
      (glow.additive_coupling, "t"),
      (glow.additive_coupling, "a"),
      (glow.affine_coupling, "c"),
      (glow.affine_coupling, "t"),
      (glow.affine_coupling, "a"),
      )
  def test_coupling_invertibility(self, func, split_dim):
    name = "affine"
    x, x_mask, _ = self.get_data()
    kwargs = self.get_kwargs(x_mask)

    x_inv, logabsdet = func(
        name, x, x_mask, split_dim=split_dim,
        identity_first=True, inverse=False, init=False, disable_dropout=True,
        **kwargs)
    x_inv_inv, logabsdet_inv = func(
        name, x_inv, x_mask, split_dim=split_dim,
        identity_first=True, inverse=True, init=False, disable_dropout=True,
        **kwargs)
    self.evaluate(tf.global_variables_initializer())
    x, x_mask, x_inv, x_inv_inv, logabsdet, logabsdet_inv = (
        self.evaluate(
            [x, x_mask, x_inv, x_inv_inv, logabsdet, logabsdet_inv]))
    diff = x - x_inv_inv
    logabsdet_sum = logabsdet + logabsdet_inv
    self.assertTrue(np.allclose(diff, 0.0, atol=1e-5))
    self.assertTrue(np.allclose(logabsdet_sum, 0.0, atol=1e-5))

  def test_split(self):
    x, x_mask, _ = self.get_data()

    x_inv, z, log_p = glow.split(
        "split", x, x_mask, inverse=False)
    x_inv_inv, _, log_p_inv = glow.split(
        "split", x_inv, x_mask, z=z, inverse=True)
    self.evaluate(tf.global_variables_initializer())
    x, x_inv, x_inv_inv, z, log_p, log_p_inv = self.evaluate(
        [x, x_inv, x_inv_inv, z, log_p, log_p_inv])
    diff = x - x_inv_inv
    log_p_diff = log_p - log_p_inv
    self.assertEqual(
        x_inv.shape, (BATCH_SIZE, TARGET_LENGTH, N_CHANNELS//2))
    self.assertEqual(
        z.shape, (BATCH_SIZE, TARGET_LENGTH, N_CHANNELS//2))
    self.assertTrue(np.allclose(diff, 0.0, atol=1e-5))
    self.assertTrue(np.allclose(log_p_diff, 0.0, atol=1e-5))

  def test_flow_invertibility(self):
    name = "flow_step"
    split_dims = "cat"
    x, x_mask, _ = self.get_data()
    kwargs = self.get_kwargs(x_mask)
    x_inv, logabsdet = glow.flow_step_glow(
        name, x, x_mask, split_dims, inverse=False, init=False, dtype=DTYPE,
        disable_dropout=True, **kwargs)
    x_inv_inv, logabsdet_inv = glow.flow_step_glow(
        name, x_inv, x_mask, split_dims, inverse=True, init=False,
        dtype=DTYPE, disable_dropout=True, **kwargs)
    self.evaluate(tf.global_variables_initializer())
    x, x_mask, x_inv, x_inv_inv, logabsdet, logabsdet_inv = (
        self.evaluate(
            [x, x_mask, x_inv, x_inv_inv, logabsdet, logabsdet_inv]))
    diff = x - x_inv_inv
    logabsdet_sum = logabsdet + logabsdet_inv
    self.assertTrue(np.allclose(diff, 0.0, atol=1e-5))
    self.assertTrue(np.allclose(logabsdet_sum, 0.0, atol=1e-5))

  @parameterized.parameters(
      ("1", "cat", "affine"),
      ("1/1", "cat/cat", "affine"),
      ("1/1/1", "cat/cat/ca", "affine"),
      )
  def test_aaa_glow_training(self, depths, split_plans, prior_type):
    with tf.Graph().as_default():
      _, x_mask, _ = self.get_data()
      x = tf.random_normal((BATCH_SIZE, TARGET_LENGTH, N_CHANNELS),
                           mean=10.0, stddev=3.0, dtype=DTYPE)
      bias = common_attention.attention_bias_ignore_padding(1.0 - x_mask)
      hparams = self.get_hparams()
      hparams.prior_type = prior_type
      hparams.depths = depths
      hparams.split_plans = split_plans
      n_levels = len(hparams.depths.split("/"))
      kwargs = self.get_kwargs(x_mask, hparams)
      _ = kwargs.pop("decoder_self_attention_bias")

      x_inv, _, _, _ = glow.glow(
          "glow", x, x_mask, bias, inverse=False, init=True,
          disable_dropout=True, **kwargs)
      curr_dir = tempfile.mkdtemp()
      model_path = os.path.join(curr_dir, "model")

      with tf.Session() as session:
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        session.run(x_inv)
        saver.save(session, model_path)

    with tf.Graph().as_default():
      _, x_mask, _ = self.get_data()
      x = tf.random_normal((BATCH_SIZE, TARGET_LENGTH, N_CHANNELS),
                           mean=10.0, stddev=3.0, dtype=DTYPE)
      bias = common_attention.attention_bias_ignore_padding(1.0 - x_mask)
      hparams = self.get_hparams()
      hparams.depths = depths
      hparams.split_plans = split_plans
      kwargs = self.get_kwargs(x_mask, hparams)
      _ = kwargs.pop("decoder_self_attention_bias")
      log_q_z = gops.standard_normal_density(x, x_mask)
      log_q_z = tf.reduce_sum(log_q_z) / tf.reduce_sum(x_mask)

      x_inv, logabsdets, log_ps, zs = glow.glow(
          "glow", x, x_mask, bias, inverse=False, init=False,
          disable_dropout=True, **kwargs)
      x_inv_inv, logabsdets_inv, log_ps_inv, _ = glow.glow(
          "glow", x_inv, x_mask, bias, inverse=True, split_zs=zs, init=False,
          disable_dropout=True, **kwargs)
      logabsdets = tf.reduce_sum(
          logabsdets, axis=0) / tf.reduce_sum(x_mask)
      logabsdets_inv = tf.reduce_sum(
          logabsdets_inv, axis=0) / tf.reduce_sum(x_mask)
      log_ps = tf.reduce_sum(log_ps, axis=0) / tf.reduce_sum(x_mask)
      log_ps_inv = tf.reduce_sum(log_ps_inv, axis=0) / tf.reduce_sum(x_mask)

      with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, model_path)
        (x, x_inv, x_inv_inv, log_q_z, logabsdets, log_ps,
         logabsdets_inv, log_ps_inv) = session.run([
             x, x_inv, x_inv_inv, log_q_z, logabsdets, log_ps,
             logabsdets_inv, log_ps_inv])
        diff = x - x_inv_inv
        log_ps_diff = log_ps - log_ps_inv
        logabsdets_sum = logabsdets + logabsdets_inv
        self.assertEqual(
            x_inv.shape,
            (BATCH_SIZE, TARGET_LENGTH//(2**(n_levels-1)), N_CHANNELS))
        print (np.max(np.abs(diff)))
        print (np.max(np.abs(log_ps_diff)))
        print (np.max(np.abs(logabsdets_sum)))
        self.assertTrue(np.allclose(diff, 0.0, atol=1e-4),
                        msg=np.max(np.abs(diff)))
        self.assertTrue(np.allclose(log_ps_diff, 0.0, atol=1e-4),
                        msg=np.max(np.abs(log_ps_diff)))
        self.assertTrue(np.allclose(logabsdets_sum, 0.0, atol=1e-4),
                        msg=np.max(np.abs(logabsdets_sum)))


if __name__ == "__main__":
  tf.test.main()
