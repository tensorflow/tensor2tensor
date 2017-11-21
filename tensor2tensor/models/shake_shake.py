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

"""Shake-shake model for CIFAR."""

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


def shake_shake_block_branch(x, conv_filters, stride):
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(
      x, conv_filters, (3, 3), strides=(stride, stride), padding="SAME")
  x = tf.layers.batch_normalization(x)
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(x, conv_filters, (3, 3), strides=(1, 1), padding="SAME")
  x = tf.layers.batch_normalization(x)
  return x


def downsampling_residual_branch(x, conv_filters):
  x = tf.nn.relu(x)
  x1 = tf.layers.average_pooling2d(x, pool_size=(1, 1), strides=(2, 2))
  x1 = tf.layers.conv2d(x1, conv_filters / 2, (1, 1), padding="SAME")
  x2 = tf.pad(x[:, 1:, 1:], [[0, 0], [0, 1], [0, 1], [0, 0]])
  x2 = tf.layers.average_pooling2d(x2, pool_size=(1, 1), strides=(2, 2))
  x2 = tf.layers.conv2d(x2, conv_filters / 2, (1, 1), padding="SAME")
  return tf.concat([x1, x2], axis=3)


def shake_shake_block(x, conv_filters, stride, hparams):
  """A shake-shake block."""
  with tf.variable_scope("branch_1"):
    branch1 = shake_shake_block_branch(x, conv_filters, stride)
  with tf.variable_scope("branch_2"):
    branch2 = shake_shake_block_branch(x, conv_filters, stride)
  if x.shape[-1] == conv_filters:
    skip = tf.identity(x)
  else:
    skip = downsampling_residual_branch(x, conv_filters)

  # TODO(rshin): Use different alpha for each image in batch.
  if hparams.mode == tf.estimator.ModeKeys.TRAIN:
    if hparams.shakeshake_type == "batch":
      shaken = common_layers.shakeshake2(branch1, branch2)
    elif hparams.shakeshake_type == "image":
      shaken = common_layers.shakeshake2_indiv(branch1, branch2)
    elif hparams.shakeshake_type == "equal":
      shaken = common_layers.shakeshake2_py(branch1, branch2, equal=True)
    else:
      raise ValueError("Invalid shakeshake_type: {!r}".format(shaken))
  else:
    shaken = common_layers.shakeshake2_py(branch1, branch2, equal=True)
  shaken.set_shape(branch1.get_shape())

  return skip + shaken


def shake_shake_stage(x, num_blocks, conv_filters, initial_stride, hparams):
  with tf.variable_scope("block_0"):
    x = shake_shake_block(x, conv_filters, initial_stride, hparams)
  for i in xrange(1, num_blocks):
    with tf.variable_scope("block_{}".format(i)):
      x = shake_shake_block(x, conv_filters, 1, hparams)
  return x


@registry.register_model
class ShakeShake(t2t_model.T2TModel):
  """Implements the Shake-Shake architecture.

  From <https://arxiv.org/pdf/1705.07485.pdf>
  This is intended to match the CIFAR-10 version, and correspond to
  "Shake-Shake-Batch" in Table 1.
  """

  def model_fn_body(self, features):
    hparams = self._hparams
    inputs = features["inputs"]
    assert (hparams.num_hidden_layers - 2) % 6 == 0
    blocks_per_stage = (hparams.num_hidden_layers - 2) // 6

    # For canonical Shake-Shake, the entry flow is a 3x3 convolution with 16
    # filters then a batch norm. Instead we will rely on the one in
    # SmallImageModality, which seems to instead use a layer norm.
    x = inputs
    with tf.variable_scope("shake_shake_stage_1"):
      x = shake_shake_stage(x, blocks_per_stage, hparams.base_filters, 1,
                            hparams)
    with tf.variable_scope("shake_shake_stage_2"):
      x = shake_shake_stage(x, blocks_per_stage, hparams.base_filters * 2, 2,
                            hparams)
    with tf.variable_scope("shake_shake_stage_3"):
      x = shake_shake_stage(x, blocks_per_stage, hparams.base_filters * 4, 2,
                            hparams)

    # For canonical Shake-Shake, we should perform 8x8 average pooling and then
    # have a fully-connected layer (which produces the logits for each class).
    # Instead, we rely on the Xception exit flow in ClassLabelModality.
    #
    # Also, this model_fn does not return an extra_loss. However, TensorBoard
    # reports an exponential moving average for extra_loss, where the initial
    # value for the moving average may be a large number, so extra_loss will
    # look large at the beginning of training.
    return x


@registry.register_hparams
def shakeshake_cifar10():
  """Parameters for CIFAR-10."""
  tf.logging.warning("shakeshake_cifar10 hparams have not been verified to "
                     "achieve good performance.")
  hparams = common_hparams.basic_params1()
  # This leads to effective batch size 128 when number of GPUs is 1
  hparams.batch_size = 4096 * 8
  hparams.hidden_size = 16
  hparams.dropout = 0
  hparams.label_smoothing = 0.0
  hparams.clip_grad_norm = 2.0
  hparams.num_hidden_layers = 26
  hparams.kernel_height = -1  # Unused
  hparams.kernel_width = -1  # Unused
  hparams.learning_rate_decay_scheme = "cosine"
  # Model should be run for 700000 steps with batch size 128 (~1800 epochs)
  hparams.learning_rate_cosine_cycle_steps = 700000
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 3000
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  # TODO(rshin): Adjust so that effective value becomes ~1e-4
  hparams.weight_decay = 3.0
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  hparams.add_hparam("base_filters", 16)
  hparams.add_hparam("shakeshake_type", "batch")
  return hparams
