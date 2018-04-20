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
"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class BasicConvGen(t2t_model.T2TModel):
  """Basic convolutional next-frame model."""

  def body(self, features):
    hparams = self.hparams
    filters = hparams.hidden_size
    kernel1, kernel2 = (3, 3), (4, 4)

    # Pad to make size powers of 2 as needed.
    x = features["inputs"]
    inputs_shape = common_layers.shape_list(x)
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2**hparams.num_compress_steps, axis=1)
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2**hparams.num_compress_steps, axis=2)

    # Down-stride.
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("downstride%d" % i):
        x = tf.layers.conv2d(x, filters, kernel2, activation=common_layers.belu,
                             strides=(2, 2), padding="SAME")
        x = common_layers.layer_norm(x)
        filters *= 2

    # Add embedded action.
    action = tf.reshape(features["input_action"][:, 1, :],
                        [-1, 1, 1, hparams.hidden_size])
    zeros = tf.zeros(common_layers.shape_list(x)[:-1] + [hparams.hidden_size],
                     dtype=tf.float32)
    x = tf.concat([x, action + zeros], axis=-1)

    # Run a stack of convolutions.
    for i in range(hparams.num_hidden_layers):
      with tf.variable_scope("layer%d" % i):
        y = tf.layers.conv2d(x, filters, kernel1, activation=common_layers.belu,
                             strides=(1, 1), padding="SAME")
        y = tf.nn.dropout(y, 1.0 - hparams.dropout)
        if i == 0:
          x = y
        else:
          x = common_layers.layer_norm(x + y)

    # Up-convolve.
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("upstride%d" % i):
        filters //= 2
        x = tf.layers.conv2d_transpose(
            x, filters, kernel2, activation=common_layers.belu,
            strides=(2, 2), padding="SAME")
        x = common_layers.layer_norm(x)
        x = tf.nn.dropout(x, 1.0 - hparams.dropout)

    # Cut down to original size.
    x = x[:, :inputs_shape[1], :inputs_shape[2], :]

    # Reward prediction.
    reward_pred = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    return {"targets": x, "target_reward": reward_pred}

  def infer(self, features=None, decode_length=50, beam_size=1, top_beams=1,
            alpha=0.0):
    """Produce predictions from the model by running it."""
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    # Get predictions.
    try:
      num_channels = self._hparams.problem.num_channels
    except AttributeError:
      num_channels = 1
    features["targets"] = tf.zeros(
        [self._hparams.batch_size, 1, 1, 1, num_channels], dtype=tf.int32)
    features["target_reward"] = tf.zeros(
        [self._hparams.batch_size, 1, 1], dtype=tf.int32)
    logits, _ = self(features)  # pylint: disable=not-callable
    if isinstance(logits, dict):
      results = {}
      for k, v in six.iteritems(logits):
        # Argmax in TF doesn't handle more than 5 dimensions yet.
        v_shape = common_layers.shape_list(v)
        argmax = tf.argmax(tf.reshape(v, [-1, v_shape[-1]]), axis=-1)
        results[k] = tf.reshape(argmax, v_shape[:-1])
    else:
      # Argmax in TF doesn't handle more than 5 dimensions yet.
      logits_shape = common_layers.shape_list(logits)
      argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=-1)
      results = tf.reshape(argmax, logits_shape[:-1])

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return results.
    return results


@registry.register_hparams
def basic_conv():
  """Basic 2-frame conv model."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 64
  hparams.batch_size = 8
  hparams.num_hidden_layers = 3
  hparams.optimizer = "Adam"
  hparams.learning_rate_constant = 0.0002
  hparams.learning_rate_warmup_steps = 500
  hparams.learning_rate_schedule = "constant * linear_warmup"
  hparams.label_smoothing = 0.05
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.dropout = 0.1
  hparams.add_hparam("num_compress_steps", 5)
  return hparams


@registry.register_hparams
def basic_conv_small():
  """Small conv model."""
  hparams = basic_conv()
  hparams.hidden_size = 32
  return hparams


@registry.register_hparams
def basic_conv_small_per_image_standardization():
  """Small conv model."""
  hparams = common_hparams.basic_params1()
  hparams.kernel_sizes = [(3, 3), (5, 5)]
  hparams.filter_numbers = [32, 3*256]
  hparams.batch_size = 2
  hparams.add_hparam("per_image_standardization", True)
  return hparams


@registry.register_model
class MichiganBasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    def deconv2d(cur, i, kernel_size, output_filters, activation=tf.nn.relu):
      thicker = common_layers.conv(
          cur,
          output_filters * 4,
          kernel_size,
          padding="SAME",
          activation=activation,
          name="deconv2d" + str(i))
      return tf.depth_to_space(thicker, 2)

    cur_frame = common_layers.standardize_images(features["inputs_0"])
    prev_frame = common_layers.standardize_images(features["inputs_1"])

    frames = tf.concat([cur_frame, prev_frame], axis=3)
    frames = tf.reshape(frames, [-1, 210, 160, 6])

    h1 = tf.layers.conv2d(frames, filters=64, strides=2, kernel_size=(8, 8),
                          padding="SAME", activation=tf.nn.relu)
    h2 = tf.layers.conv2d(h1, filters=128, strides=2, kernel_size=(6, 6),
                          padding="SAME", activation=tf.nn.relu)
    h3 = tf.layers.conv2d(h2, filters=128, strides=2, kernel_size=(6, 6),
                          padding="SAME", activation=tf.nn.relu)
    h4 = tf.layers.conv2d(h3, filters=128, strides=2, kernel_size=(4, 4),
                          padding="SAME", activation=tf.nn.relu)
    h45 = tf.reshape(h4, [-1, 14 * 10 * 128])
    h5 = tf.layers.dense(h45, 2048, activation=tf.nn.relu)
    h6 = tf.layers.dense(h5, 2048, activation=tf.nn.relu)
    h7 = tf.layers.dense(h6, 14 * 10 * 128, activation=tf.nn.relu)
    h8 = tf.reshape(h7, [-1, 14, 10, 128])

    h9 = deconv2d(h8, 1, (4, 4), 128, activation=tf.nn.relu)
    h9 = h9[:, :27, :, :]
    h10 = deconv2d(h9, 2, (6, 6), 128, activation=tf.nn.relu)
    h10 = h10[:, :53, :, :]
    h11 = deconv2d(h10, 3, (6, 6), 128, activation=tf.nn.relu)
    h11 = h11[:, :105, :, :]
    h12 = deconv2d(h11, 4, (8, 8), 3 * 256, activation=tf.identity)

    reward = tf.layers.flatten(h12)

    return {"targets": h12, "reward": reward}
