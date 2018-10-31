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

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.models.video import base
from tensor2tensor.models.video import basic_deterministic_params  # pylint: disable=unused-import
from tensor2tensor.utils import registry

import tensorflow as tf


tfl = tf.layers
tfcl = tf.contrib.layers


@registry.register_model
class NextFrameBasicDeterministic(base.NextFrameBase):
  """Basic next-frame model, may take actions and predict rewards too."""

  @property
  def is_recurrent_model(self):
    return False

  def inject_latent(self, layer, inputs, target):
    del inputs, target
    return layer, 0.0

  def middle_network(self, layer, internal_states):
    # Run a stack of convolutions.
    x = layer
    kernel1 = (3, 3)
    filters = common_layers.shape_list(x)[-1]
    for i in range(self.hparams.num_hidden_layers):
      with tf.variable_scope("layer%d" % i):
        y = tf.nn.dropout(x, 1.0 - self.hparams.residual_dropout)
        y = tf.layers.conv2d(y, filters, kernel1, activation=common_layers.belu,
                             strides=(1, 1), padding="SAME")
        if i == 0:
          x = y
        else:
          x = common_layers.layer_norm(x + y)
    return x, internal_states

  def next_frame(self, frames, actions, rewards, target_frame,
                 internal_states, video_extra):
    del rewards, video_extra

    hparams = self.hparams
    filters = hparams.hidden_size
    kernel2 = (4, 4)

    # Embed the inputs.
    stacked_frames = tf.concat(frames, axis=-1)
    inputs_shape = common_layers.shape_list(stacked_frames)
    # Using non-zero bias initializer below for edge cases of uniform inputs.
    x = tf.layers.dense(
        stacked_frames, filters, name="inputs_embed",
        bias_initializer=tf.random_normal_initializer(stddev=0.01))
    x = common_attention.add_timing_signal_nd(x)

    # Down-stride.
    layer_inputs = [x]
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("downstride%d" % i):
        layer_inputs.append(x)
        x = tf.nn.dropout(x, 1.0 - self.hparams.dropout)
        x = common_layers.make_even_size(x)
        if i < hparams.filter_double_steps:
          filters *= 2
        x = common_attention.add_timing_signal_nd(x)
        x = tf.layers.conv2d(x, filters, kernel2, activation=common_layers.belu,
                             strides=(2, 2), padding="SAME")
        x = common_layers.layer_norm(x)

    # Add embedded action if present.
    if self.has_actions:
      action = actions[-1]
      x = common_video.inject_additional_input(
          x, action, "action_enc", hparams.action_injection)

    # Inject latent if present. Only for stochastic models.
    x, extra_loss = self.inject_latent(x, frames, target_frame)

    x_mid = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x, internal_states = self.middle_network(x, internal_states)

    # Up-convolve.
    layer_inputs = list(reversed(layer_inputs))
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("upstride%d" % i):
        x = tf.nn.dropout(x, 1.0 - self.hparams.dropout)
        if self.has_actions:
          x = common_video.inject_additional_input(
              x, action, "action_enc", hparams.action_injection)
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
    x_fin = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    if self.is_per_pixel_softmax:
      x = tf.layers.dense(x, hparams.problem.num_channels * 256, name="logits")
    else:
      x = tf.layers.dense(x, hparams.problem.num_channels, name="logits")

    # No reward prediction if not needed.
    if not self.has_rewards:
      return x, None, extra_loss, internal_states

    # Reward prediction based on middle and final logits.
    reward_pred = tf.concat([x_mid, x_fin], axis=-1)
    reward_pred = tf.nn.relu(tf.layers.dense(
        reward_pred, 128, name="reward_pred"))
    reward_pred = tf.expand_dims(reward_pred, axis=3)  # Need fake channels dim.
    return x, reward_pred, extra_loss, internal_states
