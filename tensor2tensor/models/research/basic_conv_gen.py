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

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class BasicConvGen(t2t_model.T2TModel):
  """Basic convolutional next-frame model."""

  def make_even_size(self, x):
    """Pad x to be even-sized on axis 1 and 2, but only if necessary."""
    shape = [dim if dim is not None else -1 for dim in x.get_shape().as_list()]
    if shape[1] % 2 == 0 and shape[2] % 2 == 0:
      return x
    if shape[1] % 2 == 0:
      x, _ = common_layers.pad_to_same_length(
          x, x, final_length_divisible_by=2, axis=2)
      return x
    if shape[2] % 2 == 0:
      x, _ = common_layers.pad_to_same_length(
          x, x, final_length_divisible_by=2, axis=1)
      return x
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2, axis=1)
    x, _ = common_layers.pad_to_same_length(
        x, x, final_length_divisible_by=2, axis=2)
    return x

  def body(self, features):
    hparams = self.hparams
    filters = hparams.hidden_size
    kernel1, kernel2 = (3, 3), (4, 4)

    # Embed the inputs.
    inputs_shape = common_layers.shape_list(features["inputs"])
    x = tf.layers.dense(features["inputs"], filters, name="inputs_embed")

    # Down-stride.
    layer_inputs = [x]
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("downstride%d" % i):
        layer_inputs.append(x)
        x = self.make_even_size(x)
        if i < hparams.filter_double_steps:
          filters *= 2
        x = tf.layers.conv2d(x, filters, kernel2, activation=common_layers.belu,
                             strides=(2, 2), padding="SAME")
        x = common_layers.layer_norm(x)

    # Add embedded action.
    action = tf.reshape(features["input_action"][:, 1, :],
                        [-1, 1, 1, hparams.hidden_size])
    action_mask = tf.layers.dense(action, filters, name="action_mask")
    zeros_mask = tf.zeros(common_layers.shape_list(x)[:-1] + [filters],
                          dtype=tf.float32)
    x *= action_mask + zeros_mask

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
    layer_inputs = list(reversed(layer_inputs))
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("upstride%d" % i):
        if i >= hparams.num_compress_steps - hparams.filter_double_steps:
          filters //= 2
        x = tf.layers.conv2d_transpose(
            x, filters, kernel2, activation=common_layers.belu,
            strides=(2, 2), padding="SAME")
        y = layer_inputs[i]
        shape = common_layers.shape_list(y)
        x = x[:, :shape[1], :shape[2], :]
        x = common_layers.layer_norm(x + y)
        x = common_attention.add_timing_signal_nd(x)

    # Cut down to original size.
    x = x[:, :inputs_shape[1], :inputs_shape[2], :]

    # Reward prediction.
    reward_pred = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    return {"targets": x, "target_reward": reward_pred}

  def infer(self, features, *args, **kwargs):
    """Produce predictions from the model by running it."""
    # Inputs and features preparation needed to handle edge cases.
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)

    def logits_to_samples(logits):
      """Get samples from logits."""
      # If the last dimension is 1 then we're using L1/L2 loss.
      if common_layers.shape_list(logits)[-1] == 1:
        return tf.to_int32(tf.squeeze(logits, axis=-1))
      # Argmax in TF doesn't handle more than 5 dimensions yet.
      logits_shape = common_layers.shape_list(logits)
      argmax = tf.argmax(tf.reshape(logits, [-1, logits_shape[-1]]), axis=-1)
      return tf.reshape(argmax, logits_shape[:-1])

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
        results[k] = logits_to_samples(v)
    else:
      results = logits_to_samples(logits)

    # Restore inputs to not confuse Estimator in edge cases.
    if inputs_old is not None:
      features["inputs"] = inputs_old

    # Return results.
    return results


@registry.register_hparams
def basic_conv():
  """Basic 2-frame conv model."""
  hparams = common_hparams.basic_params1()
  hparams.hidden_size = 32
  hparams.batch_size = 8
  hparams.num_hidden_layers = 2
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_constant = 1.5
  hparams.learning_rate_warmup_steps = 1500
  hparams.learning_rate_schedule = "linear_warmup * constant * rsqrt_decay"
  hparams.label_smoothing = 0.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.0
  hparams.weight_decay = 0.0
  hparams.dropout = 0.5
  hparams.add_hparam("num_compress_steps", 6)
  hparams.add_hparam("filter_double_steps", 5)
  return hparams


@registry.register_hparams
def basic_conv_small():
  """Small conv model."""
  hparams = basic_conv()
  hparams.hidden_size = 32
  return hparams


@registry.register_hparams
def basic_conv_l1():
  """Basic conv model with L1 modality."""
  hparams = basic_conv()
  hparams.target_modality = "video:l1"
  return hparams


@registry.register_hparams
def basic_conv_l2():
  """Basic conv model with L2 modality."""
  hparams = basic_conv()
  hparams.target_modality = "video:l2"
  return hparams
