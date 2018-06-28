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
"""Transformer Sketch for im2sketch problems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class TransformerSketch(transformer.Transformer):
  """Transformer with strided convolutions."""

  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Add layers of strided convolutions on top of encoder."""
    with tf.variable_scope("downstride"):
      hparams = self.hparams
      kernel, strides = (4, 4), (2, 2)
      x = inputs
      # Down-convolutions.
      for i in range(hparams.num_compress_steps):
        x = common_layers.make_even_size(x)
        x = tf.layers.conv2d(
            x, hparams.hidden_size, kernel, strides=strides,
            padding="SAME", activation=common_layers.belu, name="conv_%d" % i)
        x = common_layers.layer_norm(x)

    encoder_output, encoder_decoder_attention_bias = super(
        TransformerSketch, self).encode(
            x, target_space, hparams, features=features, losses=losses)
    return encoder_output, encoder_decoder_attention_bias


@registry.register_hparams
def transformer_sketch():
  """Basic transformer_sketch hparams."""
  hparams = transformer.transformer_small()
  hparams.num_compress_steps = 4
  hparams.batch_size = 32
  hparams.clip_grad_norm = 2.
  hparams.sampling_method = "random"
  return hparams
