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
"""Tests for common image attention utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2tensor.layers import common_image_attention

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
    hparams = tf.contrib.training.HParams(
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
    hparams = tf.contrib.training.HParams(
        block_raster_scan=True,
        hidden_size=2,
        likelihood=likelihood,
        mode=tf.contrib.learn.ModeKeys.INFER,
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
    hparams = tf.contrib.training.HParams(
        hidden_size=2,
        likelihood=likelihood,
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

if __name__ == "__main__":
  tf.test.main()
