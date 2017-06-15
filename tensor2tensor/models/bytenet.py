# Copyright 2017 Google Inc.
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

"""ByteNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.models import common_hparams
from tensor2tensor.models import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def residual_dilated_conv(x, repeat, padding, name, hparams, train):
  """A stack of convolution blocks with residual connections."""
  with tf.variable_scope(name):
    k = (hparams.kernel_height, hparams.kernel_width)
    dilations_and_kernels = [((2**i, 1), k)
                             for i in xrange(hparams.num_hidden_layers)]
    for i in xrange(repeat):
      with tf.variable_scope("repeat_%d" % i):
        y = common_layers.conv_block(
            x,
            hparams.hidden_size,
            dilations_and_kernels,
            padding=padding,
            name="residual_conv")
        x = common_layers.layer_norm(x + y, hparams.hidden_size, name="lnorm")
        x = tf.nn.dropout(x, 1.0 - hparams.dropout * tf.to_float(train))
    return x


def bytenet_internal(inputs, targets, hparams, train):
  """ByteNet, main step used for training."""
  with tf.variable_scope("bytenet"):
    # Flatten inputs and extend length by 50%.
    inputs = tf.expand_dims(common_layers.flatten4d3d(inputs), axis=2)
    extend_length = tf.to_int32(0.5 * tf.to_float(tf.shape(inputs)[1]))
    inputs_shape = inputs.shape.as_list()
    inputs = tf.pad(inputs, [[0, 0], [0, extend_length], [0, 0], [0, 0]])
    inputs_shape[1] = None
    inputs.set_shape(inputs_shape)  # Don't lose the other shapes when padding.
    # Pad inputs and targets to be the same length, divisible by 50.
    inputs, targets = common_layers.pad_to_same_length(
        inputs, targets, final_length_divisible_by=50)
    final_encoder = residual_dilated_conv(
        inputs, hparams.num_block_repeat, "SAME", "encoder", hparams, train)

    shifted_targets = common_layers.shift_left(targets)
    kernel = (hparams.kernel_height, hparams.kernel_width)
    decoder_start = common_layers.conv_block(
        tf.concat([final_encoder, shifted_targets], axis=3),
        hparams.hidden_size, [((1, 1), kernel)],
        padding="LEFT")

    return residual_dilated_conv(
        decoder_start, hparams.num_block_repeat,
        "LEFT", "decoder", hparams, train)


@registry.register_model
class ByteNet(t2t_model.T2TModel):

  def model_fn_body(self, features, train):
    return bytenet_internal(features["inputs"], features["targets"],
                            self._hparams, train)


@registry.register_hparams
def bytenet_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 2048
  hparams.hidden_size = 768
  hparams.dropout = 0.2
  hparams.symbol_dropout = 0.2
  hparams.label_smoothing = 0.1
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 4
  hparams.kernel_height = 3
  hparams.kernel_width = 1
  hparams.learning_rate_decay_scheme = "exp50k"
  hparams.learning_rate = 0.05
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 3.0
  hparams.num_sampled_classes = 0
  hparams.sampling_method = "argmax"
  hparams.optimizer_adam_epsilon = 1e-6
  hparams.optimizer_adam_beta1 = 0.85
  hparams.optimizer_adam_beta2 = 0.997
  hparams.add_hparam("num_block_repeat", 4)
  return hparams
