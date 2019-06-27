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

"""SV2P: Stochastic Variational Video Prediction.

   based on the following paper:
   https://arxiv.org/abs/1710.11252
   by Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan,
      Roy H. Campbell and Sergey Levine
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.layers import discretization

from tensor2tensor.models.video import base
from tensor2tensor.models.video import base_vae
from tensor2tensor.utils import registry

import tensorflow as tf

tfl = tf.layers
tfcl = tf.contrib.layers


@registry.register_model
class NextFrameSv2p(base.NextFrameBase, base_vae.NextFrameBaseVae):
  """Stochastic Variational Video Prediction From Basic Model!"""

  @property
  def is_recurrent_model(self):
    return True

  def tinyify(self, array):
    return common_video.tinyify(
        array, self.hparams.tiny_mode, self.hparams.small_mode)

  def bottom_part_tower(self, input_image, input_reward, action, latent,
                        lstm_state, lstm_size, conv_size, concat_latent=False):
    """The bottom part of predictive towers.

    With the current (early) design, the main prediction tower and
    the reward prediction tower share the same arcitecture. TF Scope can be
    adjusted as required to either share or not share the weights between
    the two towers.

    Args:
      input_image: the current image.
      input_reward: the current reward.
      action: the action taken by the agent.
      latent: the latent vector.
      lstm_state: the current internal states of conv lstms.
      lstm_size: the size of lstms.
      conv_size: the size of convolutions.
      concat_latent: whether or not to concatenate the latent at every step.

    Returns:
      - the output of the partial network.
      - intermidate outputs for skip connections.
    """
    lstm_func = common_video.conv_lstm_2d
    tile_and_concat = common_video.tile_and_concat

    input_image = common_layers.make_even_size(input_image)
    concat_input_image = tile_and_concat(
        input_image, latent, concat_latent=concat_latent)

    layer_id = 0
    enc0 = tfl.conv2d(
        concat_input_image,
        conv_size[0], [5, 5],
        strides=(2, 2),
        activation=tf.nn.relu,
        padding="SAME",
        name="scale1_conv1")
    enc0 = tfcl.layer_norm(enc0, scope="layer_norm1")

    hidden1, lstm_state[layer_id] = lstm_func(
        enc0, lstm_state[layer_id], lstm_size[layer_id], name="state1")
    hidden1 = tile_and_concat(hidden1, latent, concat_latent=concat_latent)
    hidden1 = tfcl.layer_norm(hidden1, scope="layer_norm2")
    layer_id += 1

    hidden2, lstm_state[layer_id] = lstm_func(
        hidden1, lstm_state[layer_id], lstm_size[layer_id], name="state2")
    hidden2 = tfcl.layer_norm(hidden2, scope="layer_norm3")
    hidden2 = common_layers.make_even_size(hidden2)
    enc1 = tfl.conv2d(hidden2, hidden2.get_shape()[3], [3, 3], strides=(2, 2),
                      padding="SAME", activation=tf.nn.relu, name="conv2")
    enc1 = tile_and_concat(enc1, latent, concat_latent=concat_latent)
    layer_id += 1

    if self.hparams.small_mode:
      hidden4, enc2 = hidden2, enc1
    else:
      hidden3, lstm_state[layer_id] = lstm_func(
          enc1, lstm_state[layer_id], lstm_size[layer_id], name="state3")
      hidden3 = tile_and_concat(hidden3, latent, concat_latent=concat_latent)
      hidden3 = tfcl.layer_norm(hidden3, scope="layer_norm4")
      layer_id += 1

      hidden4, lstm_state[layer_id] = lstm_func(
          hidden3, lstm_state[layer_id], lstm_size[layer_id], name="state4")
      hidden4 = tile_and_concat(hidden4, latent, concat_latent=concat_latent)
      hidden4 = tfcl.layer_norm(hidden4, scope="layer_norm5")
      hidden4 = common_layers.make_even_size(hidden4)
      enc2 = tfl.conv2d(hidden4, hidden4.get_shape()[3], [3, 3], strides=(2, 2),
                        padding="SAME", activation=tf.nn.relu, name="conv3")
      layer_id += 1

    if action is not None:
      enc2 = common_video.inject_additional_input(
          enc2, action, "action_enc", self.hparams.action_injection)
    if input_reward is not None:
      enc2 = common_video.inject_additional_input(
          enc2, input_reward, "reward_enc")
    if latent is not None and not concat_latent:
      with tf.control_dependencies([latent]):
        enc2 = tf.concat([enc2, latent], axis=3)

    enc3 = tfl.conv2d(enc2, hidden4.get_shape()[3], [1, 1], strides=(1, 1),
                      padding="SAME", activation=tf.nn.relu, name="conv4")

    hidden5, lstm_state[layer_id] = lstm_func(
        enc3, lstm_state[layer_id], lstm_size[layer_id], name="state5")
    hidden5 = tfcl.layer_norm(hidden5, scope="layer_norm6")
    hidden5 = tile_and_concat(hidden5, latent, concat_latent=concat_latent)
    layer_id += 1
    return hidden5, (enc0, enc1), layer_id

  def reward_prediction(self, *args, **kwargs):
    model = self.hparams.reward_model
    if model == "basic":
      return self.reward_prediction_basic(*args, **kwargs)
    elif model == "big":
      return self.reward_prediction_big(*args, **kwargs)
    elif model == "mid":
      return self.reward_prediction_mid(*args, **kwargs)
    else:
      raise ValueError("Unknown reward model %s" % model)

  def reward_prediction_basic(
      self, input_images, input_reward, action, latent, mid_outputs):
    del input_reward, action, latent, mid_outputs
    x = input_images
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x = tfl.dense(x, 128, activation=tf.nn.relu, name="reward_pred")
    x = tf.expand_dims(x, axis=3)
    return x

  def reward_prediction_mid(
      self, input_images, input_reward, action, latent, mid_outputs):
    """Builds a reward prediction network from intermediate layers."""
    encoded = []
    for i, output in enumerate(mid_outputs):
      enc = output
      enc = tfl.conv2d(enc, 64, [3, 3], strides=(1, 1), activation=tf.nn.relu)
      enc = tfl.conv2d(enc, 32, [3, 3], strides=(2, 2), activation=tf.nn.relu)
      enc = tfl.conv2d(enc, 16, [3, 3], strides=(2, 2), activation=tf.nn.relu)
      enc = tfl.flatten(enc)
      enc = tfl.dense(enc, 64, activation=tf.nn.relu, name="rew_enc_%d" % i)
      encoded.append(enc)
    x = encoded
    x = tf.stack(x, axis=1)
    x = tfl.flatten(x)
    x = tfl.dense(x, 256, activation=tf.nn.relu, name="rew_dense1")
    x = tfl.dense(x, 128, activation=tf.nn.relu, name="rew_dense2")
    return x

  def reward_prediction_big(
      self, input_images, input_reward, action, latent, mid_outputs):
    """Builds a reward prediction network."""
    del mid_outputs
    conv_size = self.tinyify([32, 32, 16, 8])

    with tf.variable_scope("reward_pred", reuse=tf.AUTO_REUSE):
      x = tf.concat(input_images, axis=3)
      x = tfcl.layer_norm(x)

      if not self.hparams.small_mode:
        x = tfl.conv2d(x, conv_size[1], [3, 3], strides=(2, 2),
                       activation=tf.nn.relu, name="reward_conv1")
        x = tfcl.layer_norm(x)

      # Inject additional inputs
      if action is not None:
        x = common_video.inject_additional_input(
            x, action, "action_enc", self.hparams.action_injection)
      if input_reward is not None:
        x = common_video.inject_additional_input(x, input_reward, "reward_enc")
      if latent is not None:
        latent = tfl.flatten(latent)
        latent = tf.expand_dims(latent, axis=1)
        latent = tf.expand_dims(latent, axis=1)
        x = common_video.inject_additional_input(x, latent, "latent_enc")

      x = tfl.conv2d(x, conv_size[2], [3, 3], strides=(2, 2),
                     activation=tf.nn.relu, name="reward_conv2")
      x = tfcl.layer_norm(x)
      x = tfl.conv2d(x, conv_size[3], [3, 3], strides=(2, 2),
                     activation=tf.nn.relu, name="reward_conv3")

  def get_extra_loss(self,
                     latent_means=None, latent_stds=None,
                     true_frames=None, gen_frames=None):
    """Losses in addition to the default modality losses."""
    del true_frames, gen_frames
    return self.get_kl_loss(latent_means, latent_stds)

  def construct_predictive_tower(
      self, input_image, input_reward, action, lstm_state, latent,
      concat_latent=False):
    # Main tower
    lstm_func = common_video.conv_lstm_2d
    frame_shape = common_layers.shape_list(input_image)
    batch_size, img_height, img_width, color_channels = frame_shape
    # the number of different pixel motion predictions
    # and the number of masks for each of those predictions
    num_masks = self.hparams.num_masks
    upsample_method = self.hparams.upsample_method
    tile_and_concat = common_video.tile_and_concat

    lstm_size = self.tinyify([32, 32, 64, 64, 128, 64, 32])
    conv_size = self.tinyify([32])

    with tf.variable_scope("main", reuse=tf.AUTO_REUSE):
      hidden5, skips, layer_id = self.bottom_part_tower(
          input_image, input_reward, action, latent,
          lstm_state, lstm_size, conv_size, concat_latent=concat_latent)
      enc0, enc1 = skips

      with tf.variable_scope("upsample1", reuse=tf.AUTO_REUSE):
        enc4 = common_layers.cyclegan_upsample(
            hidden5, num_outputs=hidden5.shape.as_list()[-1],
            stride=[2, 2], method=upsample_method)

      enc1_shape = common_layers.shape_list(enc1)
      enc4 = enc4[:, :enc1_shape[1], :enc1_shape[2], :]  # Cut to shape.
      enc4 = tile_and_concat(enc4, latent, concat_latent=concat_latent)

      hidden6, lstm_state[layer_id] = lstm_func(
          enc4, lstm_state[layer_id], lstm_size[5], name="state6",
          spatial_dims=enc1_shape[1:-1])  # 16x16
      hidden6 = tile_and_concat(hidden6, latent, concat_latent=concat_latent)
      hidden6 = tfcl.layer_norm(hidden6, scope="layer_norm7")
      # Skip connection.
      hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16
      layer_id += 1

      with tf.variable_scope("upsample2", reuse=tf.AUTO_REUSE):
        enc5 = common_layers.cyclegan_upsample(
            hidden6, num_outputs=hidden6.shape.as_list()[-1],
            stride=[2, 2], method=upsample_method)

      enc0_shape = common_layers.shape_list(enc0)
      enc5 = enc5[:, :enc0_shape[1], :enc0_shape[2], :]  # Cut to shape.
      enc5 = tile_and_concat(enc5, latent, concat_latent=concat_latent)

      hidden7, lstm_state[layer_id] = lstm_func(
          enc5, lstm_state[layer_id], lstm_size[6], name="state7",
          spatial_dims=enc0_shape[1:-1])  # 32x32
      hidden7 = tfcl.layer_norm(hidden7, scope="layer_norm8")
      layer_id += 1

      # Skip connection.
      hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

      with tf.variable_scope("upsample3", reuse=tf.AUTO_REUSE):
        enc6 = common_layers.cyclegan_upsample(
            hidden7, num_outputs=hidden7.shape.as_list()[-1],
            stride=[2, 2], method=upsample_method)
      enc6 = tfcl.layer_norm(enc6, scope="layer_norm9")
      enc6 = tile_and_concat(enc6, latent, concat_latent=concat_latent)

      if self.hparams.model_options == "DNA":
        # Using largest hidden state for predicting untied conv kernels.
        enc7 = tfl.conv2d_transpose(
            enc6,
            self.hparams.dna_kernel_size**2,
            [1, 1],
            strides=(1, 1),
            padding="SAME",
            name="convt4",
            activation=None)
      else:
        # Using largest hidden state for predicting a new image layer.
        enc7 = tfl.conv2d_transpose(
            enc6,
            color_channels,
            [1, 1],
            strides=(1, 1),
            padding="SAME",
            name="convt4",
            activation=None)
        # This allows the network to also generate one image from scratch,
        # which is useful when regions of the image become unoccluded.
        transformed = [tf.nn.sigmoid(enc7)]

      if self.hparams.model_options == "CDNA":
        # cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
        cdna_input = tfl.flatten(hidden5)
        transformed += common_video.cdna_transformation(
            input_image, cdna_input, num_masks, int(color_channels),
            self.hparams.dna_kernel_size, self.hparams.relu_shift)
      elif self.hparams.model_options == "DNA":
        # Only one mask is supported (more should be unnecessary).
        if num_masks != 1:
          raise ValueError("Only one mask is supported for DNA model.")
        transformed = [
            common_video.dna_transformation(
                input_image, enc7,
                self.hparams.dna_kernel_size, self.hparams.relu_shift)]

      masks = tfl.conv2d(
          enc6, filters=num_masks + 1, kernel_size=[1, 1],
          strides=(1, 1), name="convt7", padding="SAME")
      masks = masks[:, :img_height, :img_width, ...]
      masks = tf.reshape(
          tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
          [batch_size,
           int(img_height),
           int(img_width), num_masks + 1])
      mask_list = tf.split(
          axis=3, num_or_size_splits=num_masks + 1, value=masks)
      output = mask_list[0] * input_image
      for layer, mask in zip(transformed, mask_list[1:]):
        # TODO(mbz): take another look at this logic and verify.
        output = output[:, :img_height, :img_width, :]
        layer = layer[:, :img_height, :img_width, :]
        output += layer * mask

      # Map to softmax digits
      if self.is_per_pixel_softmax:
        output = tf.layers.dense(
            output, self.hparams.problem.num_channels * 256, name="logits")

      mid_outputs = [enc0, enc1, enc4, enc5, enc6]
      return output, lstm_state, mid_outputs

  def video_features(
      self, all_frames, all_actions, all_rewards, all_raw_frames):
    """Video wide latent."""
    del all_actions, all_rewards, all_raw_frames
    if not self.hparams.stochastic_model:
      return None, None, None
    frames = tf.stack(all_frames, axis=1)
    mean, std = self.construct_latent_tower(frames, time_axis=1)
    latent = common_video.get_gaussian_tensor(mean, std)
    return [latent, mean, std]

  def next_frame(self, frames, actions, rewards, target_frame,
                 internal_states, video_features):
    del target_frame

    if self.has_policies or self.has_values:
      raise NotImplementedError("Parameter sharing with policy not supported.")

    latent, latent_mean, latent_std = video_features
    frames, actions, rewards = frames[0], actions[0], rewards[0]

    extra_loss = 0.0
    if internal_states is None:
      internal_states = [None] * (5 if self.hparams.small_mode else 7)
      if latent_mean is not None:
        extra_loss = self.get_extra_loss([latent_mean], [latent_std])

    pred_image, internal_states, mid_outputs = self.construct_predictive_tower(
        frames, None, actions, internal_states, latent)

    if not self.has_rewards:
      return pred_image, None, None, None, extra_loss, internal_states

    pred_reward = self.reward_prediction(
        pred_image, actions, rewards, latent, mid_outputs)
    return pred_image, pred_reward, None, None, extra_loss, internal_states


@registry.register_model
class NextFrameSv2pDiscrete(NextFrameSv2p):
  """SV2P with discrete latent."""

  def video_features(
      self, all_frames, all_actions, all_rewards, all_raw_frames):
    """No video wide latent."""
    del all_frames, all_actions, all_rewards, all_raw_frames
    return None

  def basic_conv_net(self, images, conv_size, scope):
    """Simple multi conv ln relu."""
    conv_size = self.tinyify(conv_size)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      x = images
      for i, c in enumerate(conv_size):
        if i > 0:
          x = tf.nn.relu(x)
        x = common_layers.make_even_size(x)
        x = tfl.conv2d(x, c, [3, 3], strides=(2, 2),
                       activation=None, padding="SAME", name="conv%d" % i)
        x = tfcl.layer_norm(x)
    return x

  def simple_discrete_latent_tower(self, input_image, target_image):
    hparams = self.hparams

    if self.is_predicting:
      batch_size = common_layers.shape_list(input_image)[0]
      rand = tf.random_uniform([batch_size, hparams.bottleneck_bits])
      bits = 2.0 * tf.to_float(tf.less(0.5, rand)) - 1.0
      return bits

    conv_size = self.tinyify([64, 32, 32, 1])
    pair = tf.concat([input_image, target_image], axis=-1)
    posterior_enc = self.basic_conv_net(pair, conv_size, "posterior_enc")
    posterior_enc = tfl.flatten(posterior_enc)
    bits, _ = discretization.tanh_discrete_bottleneck(
        posterior_enc,
        hparams.bottleneck_bits,
        hparams.bottleneck_noise,
        hparams.discretize_warmup_steps,
        hparams.mode)
    return bits

  def next_frame(self, frames, actions, rewards, target_frame,
                 internal_states, video_features):
    del video_features

    if self.has_pred_actions or self.has_values:
      raise NotImplementedError("Parameter sharing with policy not supported.")

    frames, actions, rewards = frames[0], actions[0], rewards[0]

    if internal_states is None:
      internal_states = [None] * (5 if self.hparams.small_mode else 7)

    extra_loss = 0.0
    latent = self.simple_discrete_latent_tower(frames, target_frame)

    pred_image, internal_states, _ = self.construct_predictive_tower(
        frames, None, actions, internal_states, latent, True)

    if not self.has_rewards:
      return pred_image, None, extra_loss, internal_states

    pred_reward = self.reward_prediction(
        pred_image, actions, rewards, latent)
    return pred_image, pred_reward, None, None, extra_loss, internal_states


@registry.register_model
class NextFrameSv2pAtari(NextFrameSv2p):
  """SV2P with specific changes for atari pipeline."""

  def init_internal_states(self):
    # Hardcoded LSTM-CONV shapes.
    # These sizes are calculated based on original atari frames.
    # TODO(mbz): find a cleaner way of doing this maybe?!
    batch_size = self.hparams.batch_size
    shapes = [(batch_size, 53, 40, 8),
              (batch_size, 53, 40, 8),
              (batch_size, 27, 20, 16),
              (batch_size, 27, 20, 16),
              (batch_size, 53, 40, 8)]

    with tf.variable_scope("clean_scope"):
      # Initialize conv-lstm states with zeros
      init = tf.zeros_initializer()
      states = []
      for i, shape in enumerate(shapes):
        # every lstm-conv state has two variables named c and h.
        c = tf.get_variable("c%d" % i, shape, trainable=False, initializer=init)
        h = tf.get_variable("h%d" % i, shape, trainable=False, initializer=init)
        states.append((c, h))
      return states

  def reset_internal_states_ops(self):
    zeros = [(tf.zeros_like(c), tf.zeros_like(h))
             for c, h in self.internal_states]
    return self.save_internal_states_ops(zeros)

  def load_internal_states_ops(self):
    ops = [(c.read_value(), h.read_value()) for c, h in self.internal_states]
    return ops

  def save_internal_states_ops(self, internal_states):
    ops = [[tf.assign(x[0], y[0]), tf.assign(x[1], y[1])]
           for x, y in zip(self.internal_states, internal_states)]
    return ops


@registry.register_model
class NextFrameSv2pLegacy(NextFrameSv2p):
  """Old SV2P code. Only for legacy reasons."""

  def visualize_predictions(self, real_frames, gen_frames):
    def concat_on_y_axis(x):
      x = tf.unstack(x, axis=1)
      x = tf.concat(x, axis=1)
      return x

    frames_gd = common_video.swap_time_and_batch_axes(real_frames)
    frames_pd = common_video.swap_time_and_batch_axes(gen_frames)

    if self.is_per_pixel_softmax:
      frames_pd_shape = common_layers.shape_list(frames_pd)
      frames_pd = tf.reshape(frames_pd, [-1, 256])
      frames_pd = tf.to_float(tf.argmax(frames_pd, axis=-1))
      frames_pd = tf.reshape(frames_pd, frames_pd_shape[:-1] + [3])

    frames_gd = concat_on_y_axis(frames_gd)
    frames_pd = concat_on_y_axis(frames_pd)
    side_by_side_video = tf.concat([frames_gd, frames_pd], axis=2)
    tf.summary.image("full_video", side_by_side_video)

  def get_input_if_exists(self, features, key, batch_size, num_frames):
    if key in features:
      x = features[key]
    else:
      x = tf.zeros((batch_size, num_frames, 1, self.hparams.hidden_size))
    return common_video.swap_time_and_batch_axes(x)

  def construct_model(self,
                      images,
                      actions,
                      rewards):
    """Build convolutional lstm video predictor using CDNA, or DNA.

    Args:
      images: list of tensors of ground truth image sequences
              there should be a 4D image ?xWxHxC for each timestep
      actions: list of action tensors
               each action should be in the shape ?x1xZ
      rewards: list of reward tensors
               each reward should be in the shape ?x1xZ
    Returns:
      gen_images: predicted future image frames
      gen_rewards: predicted future rewards
      latent_mean: mean of approximated posterior
      latent_std: std of approximated posterior

    Raises:
      ValueError: if more than 1 mask specified for DNA model.
    """
    context_frames = self.hparams.video_num_input_frames
    buffer_size = self.hparams.reward_prediction_buffer_size
    if buffer_size == 0:
      buffer_size = context_frames
    if buffer_size > context_frames:
      raise ValueError("Buffer size is bigger than context frames %d %d." %
                       (buffer_size, context_frames))

    batch_size = common_layers.shape_list(images[0])[0]
    ss_func = self.get_scheduled_sample_func(batch_size)

    def process_single_frame(prev_outputs, inputs):
      """Process a single frame of the video."""
      cur_image, input_reward, action = inputs
      time_step, prev_image, prev_reward, frame_buf, lstm_states = prev_outputs

      # sample from softmax (by argmax). this is noop for non-softmax loss.
      prev_image = self.get_sampled_frame(prev_image)

      generated_items = [prev_image]
      groundtruth_items = [cur_image]
      done_warm_start = tf.greater(time_step, context_frames - 1)
      input_image, = self.get_scheduled_sample_inputs(
          done_warm_start, groundtruth_items, generated_items, ss_func)

      # Prediction
      pred_image, lstm_states, _ = self.construct_predictive_tower(
          input_image, None, action, lstm_states, latent)

      if self.hparams.reward_prediction:
        reward_input_image = self.get_sampled_frame(pred_image)
        if self.hparams.reward_prediction_stop_gradient:
          reward_input_image = tf.stop_gradient(reward_input_image)
        with tf.control_dependencies([time_step]):
          frame_buf = [reward_input_image] + frame_buf[:-1]
        pred_reward = self.reward_prediction(frame_buf, None, action, latent)
        pred_reward = common_video.decode_to_shape(
            pred_reward, common_layers.shape_list(input_reward), "reward_dec")
      else:
        pred_reward = prev_reward

      time_step += 1
      outputs = (time_step, pred_image, pred_reward, frame_buf, lstm_states)

      return outputs

    # Latent tower
    latent = None
    if self.hparams.stochastic_model:
      latent_mean, latent_std = self.construct_latent_tower(images, time_axis=0)
      latent = common_video.get_gaussian_tensor(latent_mean, latent_std)

    # HACK: Do first step outside to initialize all the variables

    lstm_states = [None] * (5 if self.hparams.small_mode else 7)
    frame_buffer = [tf.zeros_like(images[0])] * buffer_size
    inputs = images[0], rewards[0], actions[0]
    init_image_shape = common_layers.shape_list(images[0])
    if self.is_per_pixel_softmax:
      init_image_shape[-1] *= 256
    init_image = tf.zeros(init_image_shape, dtype=images.dtype)
    prev_outputs = (tf.constant(0),
                    init_image,
                    tf.zeros_like(rewards[0]),
                    frame_buffer,
                    lstm_states)

    initializers = process_single_frame(prev_outputs, inputs)
    first_gen_images = tf.expand_dims(initializers[1], axis=0)
    first_gen_rewards = tf.expand_dims(initializers[2], axis=0)

    inputs = (images[1:-1], rewards[1:-1], actions[1:-1])

    outputs = tf.scan(process_single_frame, inputs, initializers)
    gen_images, gen_rewards = outputs[1:3]

    gen_images = tf.concat((first_gen_images, gen_images), axis=0)
    gen_rewards = tf.concat((first_gen_rewards, gen_rewards), axis=0)

    if self.hparams.stochastic_model:
      return gen_images, gen_rewards, [latent_mean], [latent_std]
    else:
      return gen_images, gen_rewards, None, None

  def infer(self, features, *args, **kwargs):
    """Produce predictions from the model by running it."""
    del args, kwargs
    if "targets" not in features:
      if "infer_targets" in features:
        targets_shape = common_layers.shape_list(features["infer_targets"])
      elif "inputs" in features:
        targets_shape = common_layers.shape_list(features["inputs"])
        targets_shape[1] = self.hparams.video_num_target_frames
      else:
        raise ValueError("no inputs are given.")
      features["targets"] = tf.zeros(targets_shape, dtype=tf.float32)

    output, _ = self(features)  # pylint: disable=not-callable

    if not isinstance(output, dict):
      output = {"targets": output}

    x = output["targets"]
    if self.is_per_pixel_softmax:
      x_shape = common_layers.shape_list(x)
      x = tf.reshape(x, [-1, x_shape[-1]])
      x = tf.argmax(x, axis=-1)
      x = tf.reshape(x, x_shape[:-1])
    else:
      x = tf.squeeze(x, axis=-1)
      x = tf.to_int64(tf.round(x))
    output["targets"] = x
    if self.hparams.reward_prediction:
      output["target_reward"] = tf.argmax(output["target_reward"], axis=-1)

    # only required for decoding.
    output["outputs"] = output["targets"]
    output["scores"] = output["targets"]
    return output

  def body(self, features):
    hparams = self.hparams
    batch_size = common_layers.shape_list(features["inputs"])[0]

    # Swap time and batch axes.
    input_frames = common_video.swap_time_and_batch_axes(features["inputs"])
    target_frames = common_video.swap_time_and_batch_axes(features["targets"])

    # Get actions if exist otherwise use zeros
    input_actions = self.get_input_if_exists(
        features, "input_action", batch_size, hparams.video_num_input_frames)
    target_actions = self.get_input_if_exists(
        features, "target_action", batch_size, hparams.video_num_target_frames)

    # Get rewards if exist otherwise use zeros
    input_rewards = self.get_input_if_exists(
        features, "input_reward", batch_size, hparams.video_num_input_frames)
    target_rewards = self.get_input_if_exists(
        features, "target_reward", batch_size, hparams.video_num_target_frames)

    all_actions = tf.concat([input_actions, target_actions], axis=0)
    all_rewards = tf.concat([input_rewards, target_rewards], axis=0)
    all_frames = tf.concat([input_frames, target_frames], axis=0)

    # Each image is being used twice, in latent tower and main tower.
    # This is to make sure we are using the *same* image for both, ...
    # ... given how TF queues work.
    # NOT sure if this is required at all. Doesn"t hurt though! :)
    all_frames = tf.identity(all_frames)

    gen_images, gen_rewards, latent_means, latent_stds = self.construct_model(
        images=all_frames,
        actions=all_actions,
        rewards=all_rewards,
    )

    extra_loss = self.get_extra_loss(
        latent_means=latent_means,
        latent_stds=latent_stds,
        true_frames=all_frames,
        gen_frames=gen_images)

    # Visualize predictions in Tensorboard
    if self.is_training:
      self.visualize_predictions(all_frames[1:], gen_images)

    # Ignore the predictions from the input frames.
    # This is NOT the same as original paper/implementation.
    predictions = gen_images[hparams.video_num_input_frames-1:]
    reward_pred = gen_rewards[hparams.video_num_input_frames-1:]
    reward_pred = tf.squeeze(reward_pred, axis=2)  # Remove extra dimension.

    # Swap back time and batch axes.
    predictions = common_video.swap_time_and_batch_axes(predictions)
    reward_pred = common_video.swap_time_and_batch_axes(reward_pred)

    if self.is_training and hparams.internal_loss:
      # add the loss for input frames as well.
      extra_gts = all_frames[1:hparams.video_num_input_frames]
      extra_gts = common_video.swap_time_and_batch_axes(extra_gts)
      extra_pds = gen_images[:hparams.video_num_input_frames-1]
      extra_pds = common_video.swap_time_and_batch_axes(extra_pds)
      extra_raw_gts = features["inputs_raw"][:, 1:]
      recon_loss = self.get_extra_internal_loss(
          extra_raw_gts, extra_gts, extra_pds)
      extra_loss += recon_loss

    return_targets = predictions
    if hparams.reward_prediction:
      return_targets = {"targets": predictions, "target_reward": reward_pred}

    return return_targets, extra_loss


@registry.register_model
class NextFrameSv2pTwoFrames(NextFrameSv2pLegacy):
  """Stochastic next-frame model with 2 frames posterior."""

  def construct_model(self, images, actions, rewards):
    images = tf.unstack(images, axis=0)
    actions = tf.unstack(actions, axis=0)
    rewards = tf.unstack(rewards, axis=0)

    batch_size = common_layers.shape_list(images[0])[0]
    context_frames = self.hparams.video_num_input_frames

    # Predicted images and rewards.
    gen_rewards, gen_images, latent_means, latent_stds = [], [], [], []

    # LSTM states.
    lstm_state = [None] * 7

    # Create scheduled sampling function
    ss_func = self.get_scheduled_sample_func(batch_size)

    pred_image = tf.zeros_like(images[0])
    pred_reward = tf.zeros_like(rewards[0])
    latent = None
    for timestep, image, action, reward in zip(
        range(len(images)-1), images[:-1], actions[:-1], rewards[:-1]):
      # Scheduled Sampling
      done_warm_start = timestep > context_frames - 1
      groundtruth_items = [image, reward]
      generated_items = [pred_image, pred_reward]
      input_image, input_reward = self.get_scheduled_sample_inputs(
          done_warm_start, groundtruth_items, generated_items, ss_func)

      # Latent
      # TODO(mbz): should we use input_image iunstead of image?
      latent_images = tf.stack([image, images[timestep+1]], axis=0)
      latent_mean, latent_std = self.construct_latent_tower(
          latent_images, time_axis=0)
      latent = common_video.get_gaussian_tensor(latent_mean, latent_std)
      latent_means.append(latent_mean)
      latent_stds.append(latent_std)

      # Prediction
      pred_image, lstm_state, _ = self.construct_predictive_tower(
          input_image, input_reward, action, lstm_state, latent)

      if self.hparams.reward_prediction:
        pred_reward = self.reward_prediction(
            pred_image, input_reward, action, latent)
        pred_reward = common_video.decode_to_shape(
            pred_reward, common_layers.shape_list(input_reward), "reward_dec")
      else:
        pred_reward = input_reward

      gen_images.append(pred_image)
      gen_rewards.append(pred_reward)

    gen_images = tf.stack(gen_images, axis=0)
    gen_rewards = tf.stack(gen_rewards, axis=0)

    return gen_images, gen_rewards, latent_means, latent_stds
