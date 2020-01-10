# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Tests for layers in latent variable models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.layers import discretization
from tensor2tensor.layers import latent_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import test_utils

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


def imagetransformer_latent_tiny():
  """Tiny set of hparams for a latent image model."""
  hparams = transformer.transformer_small()
  hparams.batch_size = 2
  hparams.num_hidden_layers = 3
  hparams.hidden_size = 16
  hparams.filter_size = 32
  hparams.compress_filter_size = 64
  hparams.ffn_layer = "conv_hidden_relu"
  hparams.layer_prepostprocess_dropout = 0.2
  hparams.layer_preprocess_sequence = "none"
  hparams.layer_postprocess_sequence = "dan"
  hparams.dropout = 0.3
  hparams.pos = "timing"
  hparams.num_encoder_layers = 1
  hparams.num_decoder_layers = 2
  hparams.use_pad_remover = False
  hparams.add_hparam("logit_normalization", True)
  hparams.add_hparam("bottleneck_kind", "dvq")
  hparams.add_hparam("bottleneck_bits", 4)
  hparams.add_hparam("num_residuals", 1)
  hparams.add_hparam("use_gold_targets", False)
  hparams.add_hparam("do_compress_attend", False)
  hparams.add_hparam("do_decompress_attend", False)
  hparams.add_hparam("drop_inputs", False)
  hparams.add_hparam("num_compress_steps", 2)
  hparams.add_hparam("startup_steps", 10000)
  hparams.add_hparam("mask_startup_steps", 50000)
  hparams.add_hparam("latent_dropout", 0.0)
  hparams.add_hparam("decode_autoregressive", False)
  hparams.add_hparam("vq_beta", 0.25)
  hparams.add_hparam("vq_epsilon", 1e-5)
  hparams.add_hparam("vq_decay", 0.999)
  hparams.add_hparam("ema", False)
  hparams.add_hparam("soft_em", True)
  hparams.add_hparam("num_samples", 1)
  hparams.add_hparam("num_latent_layers", 2)
  hparams.add_hparam("num_res_layers", 2)
  hparams.add_hparam("res_kernel_size", 3)
  hparams.add_hparam("num_blocks", 1)
  hparams.add_hparam("reshape_method", "slice")
  hparams.add_hparam("shared_rel", False)
  hparams.add_hparam("block_size", 1)
  hparams.add_hparam("kernel_size", 3)
  hparams.add_hparam("img_len", 8)
  hparams.add_hparam("num_channels", 1)
  hparams.add_hparam("local_and_global_att", False)
  hparams.add_hparam("block_length", 32)
  hparams.add_hparam("block_width", 128)
  hparams.add_hparam("dec_attention_type", cia.AttentionType.LOCAL_1D)
  hparams.add_hparam("latent_attention_type", cia.AttentionType.GLOBAL)
  hparams.add_hparam("block_raster_scan", False)
  hparams.add_hparam("num_latents", 1)
  hparams.add_hparam("q_filter_width", 1)
  hparams.add_hparam("kv_filter_width", 1)
  return hparams


class LatentLayersTest(tf.test.TestCase):

  @test_utils.run_in_graph_and_eager_modes()
  def testComputeBitsAndNats(self):
    reconstruction_loss = tf.random_uniform(())
    prior_loss = tf.random_uniform(())
    data_dim = tf.random_uniform((), maxval=1000, dtype=tf.int32)
    latent_dim = tf.random_uniform((), maxval=1000, dtype=tf.int32)
    nats_per_dim, bits_per_dim = latent_layers.compute_nats_and_bits_per_dim(
        data_dim,
        latent_dim,
        reconstruction_loss,
        prior_loss)

    nats_per_dim_py, bits_per_dim_conv_py = self.evaluate(
        [nats_per_dim, bits_per_dim * tf.log(2.)])
    self.assertAllClose(nats_per_dim_py, bits_per_dim_conv_py)

  @test_utils.run_in_graph_and_eager_modes()
  def testTransformerAutoencoder(self):
    hparams = imagetransformer_latent_tiny()
    hparams.mode = tf.estimator.ModeKeys.TRAIN
    block_dim = int(hparams.hidden_size // hparams.num_blocks)
    block_v_size = 2**(hparams.bottleneck_bits /
                       (hparams.num_residuals * hparams.num_blocks))
    block_v_size = int(block_v_size)
    means = tf.get_variable(
        name="means",
        shape=[hparams.num_residuals,
               hparams.num_blocks,
               block_v_size,
               block_dim],
        initializer=tf.uniform_unit_scaling_initializer())
    hparams.bottleneck = functools.partial(
        discretization.discrete_bottleneck,
        hidden_size=hparams.hidden_size,
        z_size=hparams.bottleneck_bits,
        filter_size=hparams.filter_size,
        startup_steps=hparams.startup_steps,
        bottleneck_kind=hparams.bottleneck_kind,
        num_blocks=hparams.num_blocks,
        num_residuals=hparams.num_residuals,
        reshape_method=hparams.reshape_method,
        beta=hparams.vq_beta,
        decay=hparams.vq_decay,
        soft_em=hparams.soft_em,
        num_samples=hparams.num_samples,
        epsilon=hparams.vq_epsilon,
        ema=hparams.ema,
        means=means)

    inputs = None
    batch_size = hparams.batch_size
    targets = tf.random_uniform([batch_size,
                                 hparams.img_len,
                                 hparams.img_len,
                                 hparams.hidden_size],
                                minval=-1., maxval=1.)
    target_space_id = None

    tf.train.create_global_step()
    decoder_output, losses, cache = latent_layers.transformer_autoencoder(
        inputs, targets, target_space_id, hparams)

    self.assertEqual(set(losses), {"extra", "extra_loss", "latent_pred"})

    self.evaluate(tf.global_variables_initializer())
    decoder_output_, extra_loss_, latent_pred_ = self.evaluate(
        [decoder_output, losses["extra_loss"], losses["latent_pred"]])
    self.assertEqual(decoder_output_.shape, (batch_size,
                                             hparams.img_len,
                                             hparams.img_len,
                                             hparams.hidden_size))
    self.assertEqual(extra_loss_.shape, (batch_size,))
    self.assertEqual(latent_pred_.shape, (batch_size,))
    self.assertAllGreaterEqual(extra_loss_, 0.)
    self.assertAllGreaterEqual(latent_pred_, 0.)
    self.assertEqual(cache, None)


if __name__ == "__main__":
  tf.test.main()
