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
"""Basic recurrent models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.models.video import basic_deterministic
from tensor2tensor.models.video import basic_deterministic_params
from tensor2tensor.utils import registry

import tensorflow as tf


tfl = tf.layers
tfcl = tf.contrib.layers


@registry.register_model
class NextFrameBasicRecurrent(basic_deterministic.NextFrameBasicDeterministic):
  """Basic next-frame recurrent model."""

  def predict_next_frame(self, frame, action, lstm_states):
    hparams = self.hparams
    filters = hparams.hidden_size
    kernel1, kernel2 = (3, 3), (4, 4)
    lstm_func = common_video.conv_lstm_2d

    # Embed the inputs.
    inputs_shape = common_layers.shape_list(frame)
    # Using non-zero bias initializer below for edge cases of uniform inputs.
    x = tf.layers.dense(
        frame, filters, name="inputs_embed",
        bias_initializer=tf.random_normal_initializer(stddev=0.01))
    x = common_attention.add_timing_signal_nd(x)

    # Down-stride.
    layer_inputs = [x]
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("downstride%d" % i):
        layer_inputs.append(x)
        x = common_layers.make_even_size(x)
        if i < hparams.filter_double_steps:
          filters *= 2
        x = common_attention.add_timing_signal_nd(x)
        x = tf.layers.conv2d(x, filters, kernel2, activation=common_layers.belu,
                             strides=(2, 2), padding="SAME")
        x = common_layers.layer_norm(x)

    # Add embedded action if present.
    if self.has_action:
      x = self.inject_additional_input(
          x, action, "action_enc", hparams.action_injection)

    x, extra_loss = self.inject_latent(x, self.features, filters)

    # LSTM layers
    for j in range(hparams.num_lstm_layers):
      x, lstm_states[j] = lstm_func(x, lstm_states[j], hparams.num_lstm_filters)

    # Run a stack of convolutions.
    for i in range(hparams.num_hidden_layers):
      with tf.variable_scope("layer%d" % i):
        y = tf.nn.dropout(x, 1.0 - hparams.dropout)
        y = tf.layers.conv2d(y, filters, kernel1, activation=common_layers.belu,
                             strides=(1, 1), padding="SAME")
        if i == 0:
          x = y
        else:
          x = common_layers.layer_norm(x + y)

    # Up-convolve.
    layer_inputs = list(reversed(layer_inputs))
    for i in range(hparams.num_compress_steps):
      with tf.variable_scope("upstride%d" % i):
        if self.has_action:
          x = self.inject_additional_input(
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
    if self.is_per_pixel_softmax:
      x = tf.layers.dense(x, hparams.problem.num_channels * 256, name="logits")
    else:
      x = tf.layers.dense(x, hparams.problem.num_channels, name="logits")

    # Reward prediction if needed.
    reward_pred = 0.0
    if self.has_reward:
      reward_pred = tf.expand_dims(  # Add a fake channels dim.
          tf.reduce_mean(x, axis=[1, 2], keepdims=True), axis=3)
    return x, reward_pred, extra_loss, lstm_states

  def body(self, features):
    hparams = self.hparams
    self.has_action = "input_action" in features
    self.has_reward = "target_reward" in features
    # dirty hack to enable the latent tower
    self.features = features

    # Split inputs and targets into lists.
    input_frames = tf.unstack(features["inputs"], axis=1)
    target_frames = tf.unstack(features["targets"], axis=1)
    all_frames = input_frames + target_frames
    if self.has_action:
      input_actions = tf.unstack(features["input_action"], axis=1)
      target_actions = tf.unstack(features["target_action"], axis=1)
      all_actions = input_actions + target_actions

    res_frames, sampled_frames, sampled_frames_raw, res_rewards = [], [], [], []
    lstm_states = [None] * hparams.num_lstm_layers
    extra_loss = 0.0

    num_frames = len(all_frames)
    for i in range(num_frames - 1):
      frame = all_frames[i]
      action = all_actions[i] if self.has_action else None

      # Run model.
      with tf.variable_scope("recurrent_model", reuse=tf.AUTO_REUSE):
        func_out = self.predict_next_frame(frame, action, lstm_states)
        res_frame, res_reward, res_extra_loss, lstm_states = func_out
        res_frames.append(res_frame)
        res_rewards.append(res_reward)
        extra_loss += res_extra_loss

      sampled_frame_raw = self.get_sampled_frame(res_frame)
      sampled_frames_raw.append(sampled_frame_raw)
      # TODO(lukaszkaiser): this should be consistent with modality.bottom()
      sampled_frame = common_layers.standardize_images(sampled_frame_raw)
      sampled_frames.append(sampled_frame)

      # Only for Softmax loss: sample next frame so we can keep iterating.
      if self.is_predicting and i >= hparams.video_num_input_frames:
        all_frames[i+1] = sampled_frame

    # Concatenate results and return them.
    output_frames = res_frames[hparams.video_num_input_frames-1:]
    frames = tf.stack(output_frames, axis=1)

    if not self.has_reward:
      return frames, extra_loss
    rewards = tf.concat(res_rewards[hparams.video_num_input_frames-1:], axis=1)
    return {"targets": frames, "target_reward": rewards}, extra_loss


@registry.register_hparams
def next_frame_basic_recurrent():
  """Basic 2-frame recurrent model with stochastic tower."""
  hparams = basic_deterministic_params.next_frame_basic_deterministic()
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 4
  hparams.add_hparam("num_lstm_layers", 1)
  hparams.add_hparam("num_lstm_filters", 8)
  return hparams
