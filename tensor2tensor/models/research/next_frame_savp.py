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
"""Stochastic Adversarial Video Prediction model.

Reference: https://arxiv.org/abs/1804.01523
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.models.research import next_frame
from tensor2tensor.models.research import next_frame_params  # pylint: disable=unused-import
from tensor2tensor.utils import registry
import tensorflow as tf


@registry.register_model
class NextFrameSAVP(next_frame.NextFrameStochastic):
  """Stochastic Adversarial Video Prediction."""

  def encoder(self, inputs, n_layers=3):
    """COnvnet that encodes inputs into mean and std of a gaussian.

    Args:
     inputs: 5-D Tensor, shape (batch_size, num_frames, width, height, channels)
     n_layers: Number of layers.

    Returns:
     z_mu: Mean of the latent gaussians.
     z_log_var: log(var) of the latent gaussians.

    Raises:
      ValueError: If inputs is not a 5-D tensor or not float32.
    """
    latent_dims = self.hparams.z_dim

    shape_as_list = inputs.shape.as_list()
    if len(shape_as_list) != 5:
      raise ValueError("Expected inputs to be a 5-D, got %d" %
                       len(shape_as_list))
    if inputs.dtype != tf.float32:
      raise ValueError("Expected dtype tf.float32, got %s" % inputs.dtype)

    # Flatten (N,T,W,H,C) into (NT,W,H,C)
    batch_size, _ = shape_as_list[:2]
    inputs = tf.reshape(inputs, [-1] + list(inputs.shape)[2:])
    n_filters = 64
    rectified = None

    # Applies 3 layer conv-net with padding, instance normalization
    # and leaky relu as per the encoder in
    # https://github.com/alexlee-gk/video_prediction
    padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
    for i in range(n_layers):
      with tf.variable_scope("layer_%d" % (i + 1)):
        n_filters *= 2**i
        if i:
          padded = tf.pad(rectified, padding)
        else:
          padded = tf.pad(inputs, padding)
        convolved = tf.layers.conv2d(padded, filters=n_filters, kernel_size=4,
                                     strides=2, padding="VALID")
        normalized = tf.contrib.layers.instance_norm(convolved)
        rectified = tf.nn.leaky_relu(normalized, alpha=0.2)

    # Mean pooling across all spatial dimensions.
    pooled = tf.nn.avg_pool(
        rectified, [1] + rectified.shape[1:3].as_list() + [1],
        strides=[1, 1, 1, 1], padding="VALID")
    squeezed = tf.squeeze(pooled, [1, 2])

    # Down-project and output the mean and log of the standard deviation of
    # the latents.
    with tf.variable_scope("z_mu"):
      z_mu = tf.layers.dense(squeezed, latent_dims)
    with tf.variable_scope("z_log_sigma_sq"):
      z_log_var = tf.layers.dense(squeezed, latent_dims)
      z_log_var = tf.clip_by_value(z_log_var, -10, 10)

    # Reshape to (batch_size X num_frames X latent_dims)
    z_mu = tf.reshape(z_mu, (batch_size, -1, latent_dims))
    z_log_var = tf.reshape(
        z_log_var, (batch_size, -1, latent_dims))
    return z_mu, z_log_var

  def construct_model(self, images, actions, rewards):
    """Model that takes in images and returns predictions.

    Args:
      images: list of 4-D Tensors indexed by time.
              (batch_size, width, height, channels)
      actions: list of action tensors
               each action should be in the shape ?x1xZ
      rewards: list of reward tensors
               each reward should be in the shape ?x1xZ

    Returns:
      video: list of 4-D predicted frames.
      all_rewards: predicted rewards.
      latent_means: list of gaussian means conditioned on the input at
                    every frame.
      latent_stds: list of gaussian stds conditioned on the input at
                   every frame.
    """
    images = tf.unstack(images, axis=0)
    actions = tf.unstack(actions, axis=0)
    rewards = tf.unstack(rewards, axis=0)

    latent_dims = self.hparams.z_dim
    context_frames = self.hparams.video_num_input_frames
    seq_len = len(images)
    input_shape = common_layers.shape_list(images[0])
    batch_size = input_shape[0]

    # Model does not support reward-conditioned frame generation.
    fake_rewards = rewards[:-1]

    # Concatenate x_{t-1} and x_{t} along depth and encode it to
    # produce the mean and standard deviation of z_{t-1}
    image_pairs = tf.concat([images[:seq_len - 1],
                             images[1:seq_len]], axis=-1)

    z_mu, z_log_sigma_sq = self.encoder(image_pairs)
    # Unstack z_mu and z_log_sigma_sq along the time dimension.
    z_mu = tf.unstack(z_mu, axis=0)
    z_log_sigma_sq = tf.unstack(z_log_sigma_sq, axis=0)
    iterable = zip(images[:-1], actions[:-1], fake_rewards,
                   z_mu, z_log_sigma_sq)

    # Initialize LSTM State
    lstm_state = [None] * 7
    gen_cond_video, gen_prior_video, all_rewards, latent_means, latent_stds = \
      [], [], [], [], []
    pred_image = tf.zeros_like(images[0])
    prior_latent_state, cond_latent_state = None, None
    train_mode = self.hparams.mode == tf.estimator.ModeKeys.TRAIN

    # Create scheduled sampling function
    ss_func = self.get_scheduled_sample_func(batch_size)

    with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):

      for step, (image, action, reward, mu, log_sigma_sq) in enumerate(iterable):  # pylint:disable=line-too-long
        # Sample latents using a gaussian centered at conditional mu and std.
        latent = self.get_gaussian_latent(mu, log_sigma_sq)

        # Sample prior latents from isotropic normal distribution.
        prior_latent = tf.random_normal(tf.shape(latent), dtype=tf.float32)

        # LSTM that encodes correlations between conditional latents.
        # Pg 22 in https://arxiv.org/pdf/1804.01523.pdf
        enc_cond_latent, cond_latent_state = common_video.basic_lstm(
            latent, cond_latent_state, latent_dims, name="cond_latent")

        # LSTM that encodes correlations between prior latents.
        enc_prior_latent, prior_latent_state = common_video.basic_lstm(
            prior_latent, prior_latent_state, latent_dims, name="prior_latent")

        # Scheduled Sampling
        done_warm_start = step > context_frames - 1
        groundtruth_items = [image]
        generated_items = [pred_image]
        input_image = self.get_scheduled_sample_inputs(
            done_warm_start, groundtruth_items, generated_items, ss_func)

        all_latents = tf.concat([enc_cond_latent, enc_prior_latent], axis=0)
        all_image = tf.concat([input_image, input_image], axis=0)
        all_action = tf.concat([action, action], axis=0)
        all_rewards = tf.concat([reward, reward], axis=0)

        all_pred_images, lstm_state = self.construct_predictive_tower(
            all_image, all_rewards, all_action, lstm_state, all_latents,
            concat_latent=True)

        cond_pred_images, prior_pred_images = \
          all_pred_images[:batch_size], all_pred_images[batch_size:]

        if train_mode:
          pred_image = cond_pred_images
        else:
          pred_image = prior_pred_images

        gen_cond_video.append(cond_pred_images)
        gen_prior_video.append(prior_pred_images)
        latent_means.append(mu)
        latent_stds.append(log_sigma_sq)

    gen_cond_video = tf.stack(gen_cond_video, axis=0)
    gen_prior_video = tf.stack(gen_prior_video, axis=0)
    fake_rewards = tf.stack(fake_rewards, axis=0)

    if train_mode:
      return gen_cond_video, fake_rewards, latent_means, latent_stds
    else:
      return gen_prior_video, fake_rewards, latent_means, latent_stds
