# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

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
  """

  def model_fn_body(self, features):
    inputs = features["inputs"]
    inputs.get_shape().assert_has_rank(4)

    hp = self._hparams

    out = inputs
    out = common_layers.flatten4d3d(out)

    # Conv layers
    for i in xrange(hp.num_conv_layers):
      out = conv_layer(
          out,
          hp.hidden_size,
          hp.kernel_width,
          hp.stride,
          hp.pooling_windows[i],
          hp.dropout,
          1,
          name="conv_%d" % (i + 1))

    # Dense dilated conv layers
    for i in xrange(hp.num_dconv_layers):
      dilation_rate = 2**(i + 1)
      dconv_out = conv_layer(
          out,
          hp.hidden_size,
          hp.kernel_width,
          1,
          0,
          hp.dropout,
          dilation_rate,
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
  hparams.add_hparam("num_conv_layers", 4)
  hparams.add_hparam("num_dconv_layers", 7)
  hparams.add_hparam("pooling_windows", [2, 4, 4, 4])

  # TODO(rsepassi): Correct the values of these hyperparameters
  hparams.hidden_size = 128
  hparams.kernel_width = 128
  hparams.add_hparam("stride", 1)
  return hparams
