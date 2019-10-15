# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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
from tensor2tensor.layers import discretization
from tensor2tensor.models.video import base
from tensor2tensor.models.video import basic_deterministic_params  # pylint: disable=unused-import
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_model
class NextFrameBasicDeterministic(base.NextFrameBase):
  """Basic next-frame model, may take actions and predict rewards too."""

  @property
  def is_recurrent_model(self):
    return False

  def inject_latent(self, layer, inputs, target, action):
    del inputs, target, action
    return layer, 0.0

  def middle_network(self, layer, internal_states):
    # Run a stack of convolutions.
    activation_fn = common_layers.belu
    if self.hparams.activation_fn == "relu":
      activation_fn = tf.nn.relu
    x = layer
    kernel1 = (3, 3)
    filters = common_layers.shape_list(x)[-1]
    for i in range(self.hparams.num_hidden_layers):
      with tf.variable_scope("layer%d" % i):
        y = tf.nn.dropout(x, 1.0 - self.hparams.residual_dropout)
        y = tf.layers.conv2d(y, filters, kernel1, activation=activation_fn,
                             strides=(1, 1), padding="SAME")
        if i == 0:
          x = y
        else:
          x = common_layers.layer_norm(x + y)
    return x, internal_states

  def update_internal_states_early(self, internal_states, frames):
    """Update the internal states early in the network if requested."""
    del frames
    return internal_states

  def next_frame(self, frames, actions, rewards, target_frame,
                 internal_states, video_extra):
    del rewards, video_extra

    hparams = self.hparams
    filters = hparams.hidden_size
    kernel2 = (4, 4)
    action = actions[-1]
    activation_fn = common_layers.belu
    if self.hparams.activation_fn == "relu":
      activation_fn = tf.nn.relu

    # Normalize frames.
    frames = [common_layers.standardize_images(f) for f in frames]

    # Stack the inputs.
    if internal_states is not None and hparams.concat_internal_states:
      # Use the first part of the first internal state if asked to concatenate.
      batch_size = common_layers.shape_list(frames[0])[0]
      internal_state = internal_states[0][0][:batch_size, :, :, :]
      stacked_frames = tf.concat(frames + [internal_state], axis=-1)
    else:
      stacked_frames = tf.concat(frames, axis=-1)
    inputs_shape = common_layers.shape_list(stacked_frames)

    # Update internal states early if requested.
    if hparams.concat_internal_states:
      internal_states = self.update_internal_states_early(
          internal_states, frames)

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
        x = tf.layers.conv2d(x, filters, kernel2, activation=activation_fn,
                             strides=(2, 2), padding="SAME")
        x = common_layers.layer_norm(x)

    if self.has_actions:
      with tf.variable_scope("policy"):
        x_flat = tf.layers.flatten(x)
        policy_pred = tf.layers.dense(x_flat, self.hparams.problem.num_actions)
        value_pred = tf.layers.dense(x_flat, 1)
        value_pred = tf.squeeze(value_pred, axis=-1)
    else:
      policy_pred, value_pred = None, None

    # Add embedded action if present.
    if self.has_actions:
      x = common_video.inject_additional_input(
          x, action, "action_enc", hparams.action_injection)

    # Inject latent if present. Only for stochastic models.
    norm_target_frame = common_layers.standardize_images(target_frame)
    x, extra_loss = self.inject_latent(x, frames, norm_target_frame, action)

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
            x, filters, kernel2, activation=activation_fn,
            strides=(2, 2), padding="SAME")
        y = layer_inputs[i]
        shape = common_layers.shape_list(y)
        x = x[:, :shape[1], :shape[2], :]
        x = common_layers.layer_norm(x + y)
        x = common_attention.add_timing_signal_nd(x)

    # Cut down to original size.
    x = x[:, :inputs_shape[1], :inputs_shape[2], :]
    x_fin = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    if hparams.do_autoregressive_rnn:
      # If enabled, we predict the target frame autoregregressively using rnns.
      # To this end, the current prediciton is flattened into one long sequence
      # of sub-pixels, and so is the target frame. Each sub-pixel (RGB value,
      # from 0 to 255) is predicted with an RNN. To avoid doing as many steps
      # as width * height * channels, we only use a number of pixels back,
      # as many as hparams.autoregressive_rnn_lookback.
      with tf.variable_scope("autoregressive_rnn"):
        batch_size = common_layers.shape_list(frames[0])[0]
        # Height, width, channels and lookback are the constants we need.
        h, w = inputs_shape[1], inputs_shape[2]  # 105, 80 on Atari games
        c = hparams.problem.num_channels
        lookback = hparams.autoregressive_rnn_lookback
        assert (h * w) % lookback == 0, "Number of pixels must divide lookback."
        m = (h * w) // lookback  # Batch size multiplier for the RNN.
        # These are logits that will be used as inputs to the RNN.
        rnn_inputs = tf.layers.dense(x, c * 64, name="rnn_inputs")
        # They are of shape [batch_size, h, w, c, 64], reshaping now.
        rnn_inputs = tf.reshape(rnn_inputs, [batch_size * m, lookback * c, 64])
        # Same for the target frame.
        rnn_target = tf.reshape(target_frame, [batch_size * m, lookback * c])
        # Construct rnn starting state: flatten rnn_inputs, apply a relu layer.
        rnn_start_state = tf.nn.relu(tf.layers.dense(tf.nn.relu(
            tf.layers.flatten(rnn_inputs)), 256, name="rnn_start_state"))
        # Our RNN function API is on bits, each subpixel has 8 bits.
        total_num_bits = lookback * c * 8
        # We need to provide RNN targets as bits (due to the API).
        rnn_target_bits = discretization.int_to_bit(rnn_target, 8)
        rnn_target_bits = tf.reshape(
            rnn_target_bits, [batch_size * m, total_num_bits])
        if self.is_training:
          # Run the RNN in training mode, add it's loss to the losses.
          rnn_predict, rnn_loss = discretization.predict_bits_with_lstm(
              rnn_start_state, 128, total_num_bits, target_bits=rnn_target_bits,
              extra_inputs=rnn_inputs)
          extra_loss += rnn_loss
          # We still use non-RNN predictions too in order to guide the network.
          x = tf.layers.dense(x, c * 256, name="logits")
          x = tf.reshape(x, [batch_size, h, w, c, 256])
          rnn_predict = tf.reshape(rnn_predict, [batch_size, h, w, c, 256])
          # Mix non-RNN and RNN predictions so that after warmup the RNN is 90%.
          x = tf.reshape(tf.nn.log_softmax(x), [batch_size, h, w, c * 256])
          rnn_predict = tf.nn.log_softmax(rnn_predict)
          rnn_predict = tf.reshape(rnn_predict, [batch_size, h, w, c * 256])
          alpha = 0.9 * common_layers.inverse_lin_decay(
              hparams.autoregressive_rnn_warmup_steps)
          x = alpha * rnn_predict + (1.0 - alpha) * x
        else:
          # In prediction mode, run the RNN without any targets.
          bits, _ = discretization.predict_bits_with_lstm(
              rnn_start_state, 128, total_num_bits, extra_inputs=rnn_inputs,
              temperature=0.0)  # No sampling from this RNN, just greedy.
          # The output is in bits, get back the predicted pixels.
          bits = tf.reshape(bits, [batch_size * m, lookback * c, 8])
          ints = discretization.bit_to_int(tf.maximum(bits, 0), 8)
          ints = tf.reshape(ints, [batch_size, h, w, c])
          x = tf.reshape(tf.one_hot(ints, 256), [batch_size, h, w, c * 256])
    elif self.is_per_pixel_softmax:
      x = tf.layers.dense(x, hparams.problem.num_channels * 256, name="logits")
    else:
      x = tf.layers.dense(x, hparams.problem.num_channels, name="logits")

    reward_pred = None
    if self.has_rewards:
      # Reward prediction based on middle and final logits.
      reward_pred = tf.concat([x_mid, x_fin], axis=-1)
      reward_pred = tf.nn.relu(tf.layers.dense(
          reward_pred, 128, name="reward_pred"))
      reward_pred = tf.squeeze(reward_pred, axis=1)  # Remove extra dims
      reward_pred = tf.squeeze(reward_pred, axis=1)  # Remove extra dims

    return x, reward_pred, policy_pred, value_pred, extra_loss, internal_states
