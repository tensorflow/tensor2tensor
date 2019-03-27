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

"""Tests for common image attention utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_image_attention
from tensor2tensor.utils.hparam import HParams

import tensorflow as tf


class CommonImageAttentionTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (common_image_attention.DistributionType.DMOL, 5, 50),
      (common_image_attention.DistributionType.CAT, None, 256),
  )
  def testPostProcessImageTrainMode(self, likelihood, num_mixtures, depth):
    batch = 1
    rows = 8
    cols = 24
    hparams = HParams(
        hidden_size=2,
        likelihood=likelihood,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_mixtures=num_mixtures,
    )
    inputs = tf.random_uniform([batch, rows, cols, hparams.hidden_size],
                               minval=-1., maxval=1.)
    outputs = common_image_attention.postprocess_image(
        inputs, rows, cols, hparams)
    self.assertEqual(outputs.shape, (batch, rows, cols, depth))

  @parameterized.parameters(
      (common_image_attention.DistributionType.DMOL, 5, 50),
      (common_image_attention.DistributionType.CAT, None, 256),
  )
  def testPostProcessImageInferMode(self, likelihood, num_mixtures, depth):
    batch = 1
    rows = 8
    cols = 24
    block_length = 4
    block_width = 2
    hparams = HParams(
        block_raster_scan=True,
        hidden_size=2,
        likelihood=likelihood,
        mode=tf.estimator.ModeKeys.PREDICT,
        num_mixtures=num_mixtures,
        query_shape=[block_length, block_width],
    )
    inputs = tf.random_uniform([batch, rows, cols, hparams.hidden_size],
                               minval=-1., maxval=1.)
    outputs = common_image_attention.postprocess_image(
        inputs, rows, cols, hparams)
    num_blocks_rows = rows // block_length
    num_blocks_cols = cols // block_width
    self.assertEqual(outputs.shape,
                     (batch, num_blocks_rows, num_blocks_cols,
                      block_length, block_width, depth))

  @parameterized.parameters(
      (common_image_attention.DistributionType.DMOL, 5, 50),
      (common_image_attention.DistributionType.CAT, None, 256),
  )
  def testCreateOutputTrainMode(self, likelihood, num_mixtures, depth):
    batch = 1
    height = 8
    width = 8
    channels = 3
    rows = height
    if likelihood == common_image_attention.DistributionType.CAT:
      cols = channels * width
    else:
      cols = width
    hparams = HParams(
        hidden_size=2,
        likelihood=likelihood,
        num_channels=channels,
        mode=tf.estimator.ModeKeys.TRAIN,
        num_mixtures=num_mixtures,
    )
    decoder_output = tf.random_normal([batch, rows, cols, hparams.hidden_size])
    targets = tf.random_uniform([batch, height, width, channels],
                                minval=-1., maxval=1.)
    output = common_image_attention.create_output(
        decoder_output, rows, cols, targets, hparams)
    if hparams.likelihood == common_image_attention.DistributionType.CAT:
      self.assertEqual(output.shape, (batch, height, width, channels, depth))
    else:
      self.assertEqual(output.shape, (batch, height, width, depth))

  def testTransformerDecoderLayersGlobal(self):
    one_hot_data = tf.constant([[[0., 1.], [1., 0.]],
                                [[0., 1.], [1., 0.]],
                                [[1., 0.], [1., 0.]]])

    hparams = common_hparams.basic_params1()
    hparams.hidden_size = 4
    hparams.num_layers = 1
    hparams.layer_prepostprocess_dropout = 0.

    hparams.add_hparam("attention_key_channels", None)
    hparams.add_hparam("attention_value_channels", None)
    hparams.add_hparam("num_heads", 1)
    hparams.add_hparam("attention_dropout", 0.)
    hparams.add_hparam("shared_rel", False)
    hparams.add_hparam("block_width", 1)
    hparams.add_hparam("block_length", 1)
    hparams.add_hparam("q_filter_width", 1)
    hparams.add_hparam("kv_filter_width", 1)
    hparams.add_hparam("filter_size", 16)
    hparams.add_hparam("ffn_layer", "conv_hidden_relu")
    hparams.add_hparam("relu_dropout", 0.)

    conv_1d = tf.keras.layers.Conv1D(filters=hparams.hidden_size,
                                     kernel_size=1,
                                     use_bias=False)
    shifted_data = tf.pad(one_hot_data, [[0, 0], [1, 0], [0, 0]])[..., :-1, :]
    net = conv_1d(shifted_data)
    output = common_image_attention.transformer_decoder_layers(
        inputs=net,
        encoder_output=None,
        num_layers=hparams.num_layers,
        hparams=hparams,
        self_attention_bias=common_image_attention.get_self_attention_bias(net),
        attention_type=common_image_attention.AttentionType.GLOBAL)
    self.evaluate(tf.global_variables_initializer())
    output_val = self.evaluate(output)
    # The outputs for the padded dimension should be equal across all data.
    self.assertAllEqual(output_val[0, 0], output_val[1, 0])
    self.assertAllEqual(output_val[1, 0], output_val[2, 0])
    # The first and second elements of the batch are identical, so they should
    # have the same outputs for the second latent dimension as well.
    self.assertAllEqual(output_val[0, 1], output_val[1, 1])


if __name__ == "__main__":
  tf.test.main()
