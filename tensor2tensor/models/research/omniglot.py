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

"""Omniglot."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers
from tensor2tensor.models.research import transformer_moe
from tensor2tensor.utils import modality
from tensor2tensor.utils import registry

import tensorflow as tf


def conv_layer(x,
               hidden_size,
               kernel_size,
               stride,
               pooling_window,
               dropout_rate,
               dilation_rate,
               name="conv"):
  with tf.variable_scope(name):
    out = x
    out = common_layers.conv_block(
      out,
      hidden_size, [(dilation_rate, kernel_size)],
      strides=stride,
      first_relu=False,
      padding="same")
  out = tf.contrib.layers.layer_norm(out)
  out = tf.nn.relu(out)
  if pooling_window:
    out = tf.layers.max_pooling2d(
      out, pooling_window, pooling_window, padding="same")
  out = tf.layers.dropout(out, dropout_rate)
  return out


def fc_layer(x, num_out, dropout_rate, name="fc"):
  with tf.variable_scope(name):
    out = x
    out = tf.layers.dense(out, num_out)
    out = tf.contrib.layers.layer_norm(out)
    out = tf.nn.relu(out)
    out = tf.layers.dropout(out, dropout_rate)
    return out


@registry.register_model("omniglot_transformer_moe")
class OmniglotTansformerMoe(transformer_moe.TransformerMoe):
  """transformer (attention seq-seq model) with mixtures of experts."""

  def frame_level_features(self, inputs):
    """Transform input from data space to model space.

    Based on "Matching Networks" model from
    https://arxiv.org/abs/1606.04080
    Uses layer_norm instead of batch_norm.

    Args:
      inputs: A Tensor with shape [batch, length, height, width, channels]
      name: string, scope.

    Returns:
      body_input: A Tensor with shape [batch, length, body_input_depth].
    """
    inputs.get_shape().assert_has_rank(5)
    hp = self._hparams #_model_hparams

    def conv_body(inputs):
      inputs.get_shape().assert_has_rank(4)
      # reuse conv_body variables for each image
      with tf.variable_scope("conv_body", reuse=tf.AUTO_REUSE):
        ret = inputs
        for i in xrange(hp.bottom_num_conv_layers):
          ret = conv_layer(ret,
                           hp.bottom_hidden_size,
                           hp.bottom_kernel_size,
                           hp.bottom_stride,
                           hp.bottom_pooling_window,
                           hp.dropout,
                           hp.bottom_dilation,
                           name="conv_%d" % (i + 1))
        ret = tf.contrib.layers.flatten(ret)
        ret.get_shape().assert_has_rank(2)
        if common_layers.shape_list(ret)[1] == hp.bottom_hidden_size:
          return ret
        else:
          ret = fc_layer(ret, hp.bottom_hidden_size, hp.dropout, name="fc")
          return ret

    ret = tf.to_float(inputs)
    # ret is a Tensor with shape [length, batch, ...]
    ret = tf.transpose(ret, [1, 0, 2, 3, 4])
    ret = tf.map_fn(conv_body, ret)
    ret.get_shape().assert_has_rank(3)
    # ret is a Tensor with shape [batch, length, ...]
    ret = tf.transpose(ret, [1, 0, 2])
    return ret

  def bottom(self, features):
    """Transform features to feed into body."""
    inputs = features["inputs"]
    inputs = self.frame_level_features(inputs)

    lagged_targets = features["targets/background/one_hot"]
    lagged_targets = tf.pad(lagged_targets, [[0, 0], [1, 0], [0, 0]])
    lagged_targets = tf.to_float(lagged_targets)

    inputs = tf.concat([inputs, lagged_targets], axis=2)
    inputs = tf.expand_dims(inputs, axis=2)
    inputs.get_shape().assert_has_rank(4)

    transformed_features = super(OmniglotTansformerMoe, self).bottom(features)
    transformed_features["inputs"] = inputs
    return transformed_features


def add_default_omniglot_hparams(hparams):
  """Adds the hparams used by the Omniglot image feature function."""

  # Hparams for Omniglot image feature convnet.
  hparams.add_hparam("bottom_num_conv_layers", 4)
  hparams.add_hparam("bottom_hidden_size", 64)
  hparams.add_hparam("bottom_kernel_size", (3, 3))
  hparams.add_hparam("bottom_stride", (1, 1))
  hparams.add_hparam("bottom_dilation", (1, 1))
  hparams.add_hparam("bottom_pooling_window", (2, 2))
  return hparams


@registry.register_hparams("omniglot_hparams")
def omniglot_hparams():
  """Omniglot hparams.

  Will have the following architecture by default:
  * No encoder.
  * Decoder architecture:
    * Layer 0: a - sepm  (masked self-attention/masked separable convolutions)
    * Layer 1: a - sepm
    * Layer 2: a - sepm
    * Layer 3: a - sepm

  Returns:
    hparams
  """
  hparams = transformer_moe.transformer_moe_base()
  hparams = add_default_omniglot_hparams(hparams)

  hparams.batch_size = 64
  hparams.default_ff = "sep"

  # hparams.layer_types contains the network architecture:
  encoder_archi = ""
  decoder_archi = "a-sepm/a-sepm/a-sepm/a-sepm"
  hparams.layer_types = "{}#{}".format(encoder_archi, decoder_archi)

  return hparams


@registry.register_hparams("omniglot_moe_hparams")
def omniglot_moe_hparams():
  """Base transformers model with moe.

  Will have the following architecture:
  * No encoder.
  * Decoder architecture:
    * Layer 0: a - a - sepm  (self-attention - enco/deco-attention - masked sep)
    * Layer 1: a - a - sepm
    * Layer 2: a - a - moe  (mixture of expert layers in the middle)
    * Layer 3: a - a - sepm
    * Layer 4: a - a - sepm

  Returns:
    hparams
  """
  hparams = transformer_moe.transformer_moe_8k()
  hparams = add_default_omniglot_hparams(hparams)

  hparams.batch_size = 128
  hparams.default_ff = "sep"

  # hparams.layer_types contains the network architecture:
  encoder_archi = ""
  decoder_archi = "a-sepm/a-sepm/a-moe/a-sepm/a-sepm"
  hparams.layer_types = "{}#{}".format(encoder_archi, decoder_archi)

  return hparams
