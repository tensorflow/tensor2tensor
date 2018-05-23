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
"""Models for gene expression from DNA."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class GeneExpressionConv(t2t_model.T2TModel):
  """Gene expression conv net.

  Based on "Basenji" model from
  http://www.biorxiv.org/content/early/2017/07/10/161851

  Uses layer_norm instead of batch_norm.

  Model expects that if targets are of length m, inputs are of length 32*m.  The
  original data expected that inputs would be of length 128*m, but the data has
  been preprocessed to chunk every 4 bases into 1 ID (see
  data_generators/gene_expression.py).

  The magnitude of the length reduction is controlled by the pooling sizes
  (hparams.pooling_windows) at each conv layer (hparams.num_conv_layers).
  """

  def body(self, features):
    inputs = features["inputs"]
    inputs.get_shape().assert_has_rank(4)

    hp = self._hparams

    out = inputs
    out = common_layers.flatten4d3d(out)

    # Conv layers
    assert hp.num_conv_layers == len(hp.pooling_windows)
    for i in range(hp.num_conv_layers):
      out = conv_layer(
          out,
          hp.hidden_size,
          hp.kernel_width,
          hp.stride,
          hp.pooling_windows[i],
          hp.dropout,
          dilation_rate=1,
          name="conv_%d" % (i + 1))

    # Dense dilated conv layers
    for i in range(hp.num_dconv_layers):
      dilation_rate = 2**(i + 1)
      dconv_out = conv_layer(
          out,
          hp.hidden_size,
          hp.kernel_width,
          stride=1,
          pooling_window=0,
          dropout_rate=hp.dropout,
          dilation_rate=dilation_rate,
          name="dconv_%d" % (i + 1))
      out = tf.concat([out, dconv_out], axis=2)

    # Fully connected layer
    out = fc_layer(out, hp.hidden_size, hp.dropout, name="fc")

    out.get_shape().assert_has_rank(3)
    out = tf.expand_dims(out, 2)
    return out


def conv_layer(x,
               hidden_size,
               kernel_size,
               stride,
               pooling_window,
               dropout_rate,
               dilation_rate,
               name="conv"):
  """Single conv layer with relu, optional pooling, and dropout."""
  with tf.variable_scope(name):
    out = x
    out = common_layers.conv1d_block(
        out,
        hidden_size, [(dilation_rate, kernel_size)],
        strides=stride,
        first_relu=False,
        padding="same")
    out = tf.nn.relu(out)
    if pooling_window:
      out = tf.layers.max_pooling1d(
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


@registry.register_hparams
def gene_expression_conv_base():
  """Hparams for GeneExpressionConv model."""
  hparams = common_hparams.basic_params1()

  batch_size = 10
  output_length = 2048
  inputs_per_output = 128
  chunk_size = 4
  input_length = output_length * inputs_per_output // chunk_size
  hparams.batch_size = input_length * batch_size

  hparams.dropout = 0.1
  hparams.add_hparam("num_conv_layers", 4)
  hparams.add_hparam("num_dconv_layers", 7)
  # The product of these pooling windows should match
  # input_length/target_length.
  hparams.add_hparam("pooling_windows", [2, 2, 2, 4])

  hparams.hidden_size = 256
  hparams.kernel_width = 20
  hparams.add_hparam("stride", 1)
  return hparams
