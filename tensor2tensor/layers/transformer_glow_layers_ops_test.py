# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
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

"""Tests for tensor2tensor.layers.transformer_flow_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensor2tensor.layers import transformer_glow_layers_ops as gops
from tensor2tensor.models import transformer
import tensorflow.compat.v1 as tf

BATCH_SIZE = 10
INPUT_LENGTH = 3
TARGET_LENGTH = 16
N_CHANNELS = 24
HIDDEN_SIZE = 64
N_1X1_HEADS = 4


class TransformerFlowOpsTest(parameterized.TestCase, tf.test.TestCase):

  def get_data(self):
    x = tf.random_normal((BATCH_SIZE, TARGET_LENGTH, N_CHANNELS),
                         mean=0.0, stddev=1.0)
    x_lengths = np.random.randint(low=1, high=TARGET_LENGTH+1, size=BATCH_SIZE)
    x_mask = tf.sequence_mask(x_lengths, maxlen=TARGET_LENGTH, dtype=tf.float32)
    return x, x_mask

  def get_hparams(self):
    hparams = transformer.transformer_small()
    hparams.add_hparam("prior_type", "affine")
    hparams.add_hparam("depths", "12")  # infer n_levels from depths
    hparams.add_hparam("split_plans", "tca")
    hparams.add_hparam("factor", 2)  # squeezing factor
    hparams.add_hparam("n_layers_transform_params", 1)
    hparams.add_hparam("n_layers_multiscale_prior", 3)
    hparams.add_hparam("flow_num_heads", 4)
    hparams.add_hparam("flow_num_1x1_heads", N_1X1_HEADS)
    hparams.add_hparam("flow_hidden_size", 64)
    hparams.add_hparam("flow_filter_size", 128)
    hparams.add_hparam("cond_prior_on_src", True)
    hparams.add_hparam("bottom_prior_std", False)
    hparams.add_hparam("latent_size", N_CHANNELS)
    hparams.add_hparam("scale_width", 0.999)
    hparams.add_hparam("coupling_transform_ratio", 0.5)
    hparams.add_hparam("actnorm_type", "actnorm")
    hparams.add_hparam("actnorm_weightnorm", True)
    hparams.add_hparam("perm_type", "1x1")
    hparams.add_hparam("init_permutation", True)
    hparams.causal_decoder_self_attention = False
    hparams.hidden_size = HIDDEN_SIZE
    return hparams

  def get_kwargs(self, hparams=None):
    if hparams is None:
      hparams = self.get_hparams()
    encoder_output = tf.random.uniform(
        (BATCH_SIZE, INPUT_LENGTH, HIDDEN_SIZE))
    encoder_decoder_attention_bias = tf.random.uniform(
        (BATCH_SIZE, 1, 1, INPUT_LENGTH))
    decoder_self_attention_bias = tf.random.uniform(
        (BATCH_SIZE, 1, 1, TARGET_LENGTH))
    kwargs = {"hparams": hparams,
              "encoder_output": encoder_output,
              "encoder_decoder_attention_bias": encoder_decoder_attention_bias,
              "decoder_self_attention_bias": decoder_self_attention_bias}
    return kwargs

  def test_dense_weightnorm(self):
    x, x_mask = self.get_data()
    x = tf.random_normal((BATCH_SIZE, TARGET_LENGTH, HIDDEN_SIZE),
                         mean=0.0, stddev=1.0)
    y = gops.dense_weightnorm("wn", x, N_CHANNELS, x_mask,
                              init_scale=1.0, init=True)

    y_nopad = tf.boolean_mask(y, x_mask)
    mean, var = tf.nn.moments(y_nopad, axes=[0])
    self.evaluate(tf.global_variables_initializer())
    x, x_mask, y, y_nopad, mean, var = (
        self.evaluate([x, x_mask, y, y_nopad, mean, var]))
    self.assertEqual(y.shape, (BATCH_SIZE, TARGET_LENGTH, N_CHANNELS))
    self.assertTrue(np.allclose(mean, 0.0, atol=1e-5))
    self.assertTrue(np.allclose(var, 1.0, atol=1e-5))

if __name__ == "__main__":
  tf.test.main()
