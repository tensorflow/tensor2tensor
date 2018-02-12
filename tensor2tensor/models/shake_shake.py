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

"""Shake-shake model for CIFAR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


def shake_shake_skip_connection(x, output_filters, stride, is_training):
  """Adds a residual connection to the filter x for the shake-shake model."""
  curr_filters = common_layers.shape_list(x)[-1]
  if curr_filters == output_filters:
    return x
  stride_spec = [1, stride, stride, 1]
  # Skip path 1.
  path1 = tf.nn.avg_pool(x, [1, 1, 1, 1], stride_spec, "VALID")
  path1 = tf.layers.conv2d(path1, int(output_filters / 2), (1, 1),
                           padding="SAME", name="path1_conv")

  # Skip path 2.
  pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]  # First pad with 0's then crop.
  path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
  path2 = tf.nn.avg_pool(path2, [1, 1, 1, 1], stride_spec, "VALID")
  path2 = tf.layers.conv2d(path2, int(output_filters / 2), (1, 1),
                           padding="SAME", name="path2_conv")

  # Concat and apply BN.
  final_path = tf.concat(values=[path1, path2], axis=-1)
  final_path = tf.layers.batch_normalization(
      final_path, training=is_training, name="final_path_bn")
  return final_path


def shake_shake_branch(x, output_filters, stride, rand_forward, rand_backward,
                       hparams):
  """Building a 2 branching convnet."""
  is_training = hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(x, output_filters, (3, 3), strides=(stride, stride),
                       padding="SAME", name="conv1")
  x = tf.layers.batch_normalization(x, training=is_training, name="bn1")
  x = tf.nn.relu(x)
  x = tf.layers.conv2d(x, output_filters, (3, 3), padding="SAME", name="conv2")
  x = tf.layers.batch_normalization(x, training=is_training, name="bn2")
  if is_training:
    x = x * rand_backward + tf.stop_gradient(x * rand_forward -
                                             x * rand_backward)
  else:
    x *= 1.0 / hparams.shake_shake_num_branches
  return x


def shake_shake_block(x, output_filters, stride, hparams):
  """Builds a full shake-shake sub layer."""
  is_training = hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
  batch_size = common_layers.shape_list(x)[0]

  # Generate random numbers for scaling the branches.
  rand_forward = [
      tf.random_uniform(
          [batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32)
      for _ in range(hparams.shake_shake_num_branches)
  ]
  rand_backward = [
      tf.random_uniform(
          [batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32)
      for _ in range(hparams.shake_shake_num_branches)
  ]
  # Normalize so that all sum to 1.
  total_forward = tf.add_n(rand_forward)
  total_backward = tf.add_n(rand_backward)
  rand_forward = [samp / total_forward for samp in rand_forward]
  rand_backward = [samp / total_backward for samp in rand_backward]
  zipped_rand = zip(rand_forward, rand_backward)

  branches = []
  for branch, (r_forward, r_backward) in enumerate(zipped_rand):
    with tf.variable_scope("branch_{}".format(branch)):
      b = shake_shake_branch(x, output_filters, stride, r_forward, r_backward,
                             hparams)
      b = tf.nn.dropout(b, 1.0 - hparams.layer_prepostprocess_dropout)
      branches.append(b)
  res = shake_shake_skip_connection(x, output_filters, stride, is_training)
  if hparams.shake_shake_concat:
    concat_values = [res] + branches
    concat_output = tf.concat(values=concat_values, axis=-1)
    concat_output = tf.nn.relu(concat_output)
    concat_output = tf.layers.conv2d(
        concat_output, output_filters, (1, 1), name="concat_1x1")
    concat_output = tf.layers.batch_normalization(
        concat_output, training=is_training, name="concat_bn")
    return concat_output
  else:
    return res + tf.add_n(branches)


def shake_shake_layer(x, output_filters, num_blocks, stride, hparams):
  """Builds many sub layers into one full layer."""
  for block_num in range(num_blocks):
    curr_stride = stride if (block_num == 0) else 1
    with tf.variable_scope("layer_{}".format(block_num)):
      x = shake_shake_block(x, output_filters, curr_stride, hparams)
  return x


@registry.register_model
class ShakeShake(t2t_model.T2TModel):
  """Implements the Shake-Shake architecture.

  From <https://arxiv.org/pdf/1705.07485.pdf>
  This is intended to match the CIFAR-10 version, and correspond to
  "Shake-Shake-Batch" in Table 1.
  """

  def body(self, features):
    hparams = self._hparams
    is_training = hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
    inputs = features["inputs"]
    assert (hparams.num_hidden_layers - 2) % 6 == 0
    assert hparams.hidden_size % 16 == 0
    k = hparams.hidden_size // 16
    n = (hparams.num_hidden_layers - 2) // 6
    x = inputs

    x = tf.layers.conv2d(x, 16, (3, 3), padding="SAME", name="init_conv")
    x = tf.layers.batch_normalization(x, training=is_training, name="init_bn")
    with tf.variable_scope("L1"):
      x = shake_shake_layer(x, 16 * k, n, 1, hparams)
    with tf.variable_scope("L2"):
      x = shake_shake_layer(x, 32 * k, n, 2, hparams)
    with tf.variable_scope("L3"):
      x = shake_shake_layer(x, 64 * k, n, 2, hparams)
    x = tf.nn.relu(x)

    # Global avg on [1, 2] (we're nhwc) and dense to num_classes done by top.
    return x


@registry.register_hparams
def shakeshake_small():
  """Parameters for CIFAR-10. Gets to about 96% accuracy@700K steps, 1 GPU."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 128
  hparams.hidden_size = 32
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.dropout = 0
  hparams.label_smoothing = 0.0
  hparams.clip_grad_norm = 0.0  # No clipping for now, one can also try 2.0.
  hparams.num_hidden_layers = 26
  hparams.learning_rate_decay_scheme = "cosine"
  # Model should be run for 700000 steps with batch size 128 (~1800 epochs)
  hparams.learning_rate_cosine_cycle_steps = 700000
  hparams.learning_rate = 0.2
  hparams.learning_rate_warmup_steps = 100  # That's basically unused.
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 1e-4
  hparams.optimizer = "Momentum"
  hparams.optimizer_momentum_momentum = 0.9
  hparams.add_hparam("shake_shake_num_branches", 2)
  hparams.add_hparam("shake_shake_concat", int(False))
  return hparams


@registry.register_hparams
def shake_shake_quick():
  hparams = shakeshake_small()
  hparams.optimizer = "Adam"
  hparams.learning_rate_cosine_cycle_steps = 1000
  hparams.learning_rate = 0.5
  hparams.batch_size = 100
  return hparams


@registry.register_hparams
def shakeshake_big():
  hparams = shakeshake_small()
  hparams.layer_prepostprocess_dropout = 0.0
  hparams.hidden_size = 96
  return hparams


@registry.register_hparams
def shakeshake_tpu():
  hparams = shakeshake_big()
  hparams.learning_rate_cosine_cycle_steps = 180000
  hparams.learning_rate = 0.6
  return hparams
