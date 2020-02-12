# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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
import numbers
import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
from tensor2tensor.models.video import savp_params  # pylint: disable=unused-import
from tensor2tensor.models.video import sv2p
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry
from tensor2tensor.utils import update_ops_hook

import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan

gan_losses = tfgan.losses.wargs


class NextFrameSavpBase(object):
  """Main function for Stochastic Adversarial Video Prediction."""

  def encoder(self, inputs, n_layers=3):
    """Convnet that encodes inputs into mean and std of a gaussian.

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
        normalized = contrib.layers().instance_norm(convolved)
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

  def expected_output_shape(self, input_shape, stride, padding, kernel_size):
    return (input_shape + 2*padding - kernel_size) // stride + 1

  def get_fc_dimensions(self, strides, kernel_sizes):
    """Get expected fully connected shape after a series of convolutions."""
    output_height, output_width, _ = self.hparams.problem.frame_shape
    output_steps = self.hparams.video_num_target_frames
    output_shape = np.array([output_steps, output_height, output_width])
    for curr_stride, kernel_size in zip(strides, kernel_sizes):
      output_shape = self.expected_output_shape(
          output_shape, np.array(curr_stride), 1, kernel_size)
    return np.prod(output_shape) * self.hparams.num_discriminator_filters * 8

  def discriminator(self, frames):
    """3-D SNGAN discriminator.

    Args:
      frames: a list of batch-major tensors indexed by time.

    Returns:
      logits: 1-D Tensor with shape=batch_size.
              Positive logits imply that the discriminator thinks that it
              belongs to the true class.
    """
    ndf = self.hparams.num_discriminator_filters
    frames = tf.stack(frames)

    # Switch from time-major axis to batch-major axis.
    frames = common_video.swap_time_and_batch_axes(frames)

    # 3-D Conv-net mapping inputs to activations.
    num_outputs = [ndf, ndf*2, ndf*2, ndf*4, ndf*4, ndf*8, ndf*8]
    kernel_sizes = [3, 4, 3, 4, 3, 4, 3]
    strides = [[1, 1, 1], [1, 2, 2], [1, 1, 1], [1, 2, 2], [1, 1, 1],
               [2, 2, 2], [1, 1, 1]]

    names = ["video_sn_conv0_0", "video_sn_conv0_1", "video_sn_conv1_0",
             "video_sn_conv1_1", "video_sn_conv2_0", "video_sn_conv2_1",
             "video_sn_conv3_0"]
    iterable = zip(num_outputs, kernel_sizes, strides, names)
    activations = frames
    for num_filters, kernel_size, stride, name in iterable:
      activations = self.pad_conv3d_lrelu(activations, num_filters, kernel_size,
                                          stride, name)
    num_fc_dimensions = self.get_fc_dimensions(strides, kernel_sizes)
    activations = tf.reshape(activations, (-1, num_fc_dimensions))
    return tf.squeeze(tf.layers.dense(activations, 1))

  def d_step(self, true_frames, gen_frames):
    """Performs the discriminator step in computing the GAN loss.

    Applies stop-gradient to the generated frames while computing the
    discriminator loss to make sure that the gradients are not back-propagated
    to the generator. This makes sure that only the discriminator is updated.

    Args:
      true_frames: True outputs
      gen_frames: Generated frames.
    Returns:
      d_loss: Loss component due to the discriminator.
    """
    hparam_to_disc_loss = {
        "least_squares": gan_losses.least_squares_discriminator_loss,
        "cross_entropy": gan_losses.modified_discriminator_loss,
        "wasserstein": gan_losses.wasserstein_discriminator_loss}

    # Concat across batch-axis.
    _, batch_size, _, _, _ = common_layers.shape_list(true_frames)
    all_frames = tf.concat(
        [true_frames, tf.stop_gradient(gen_frames)], axis=1)

    all_logits = self.discriminator(all_frames)
    true_logits, fake_logits_stop = \
      all_logits[:batch_size], all_logits[batch_size:]
    mean_true_logits = tf.reduce_mean(true_logits)
    tf.summary.scalar("mean_true_logits", mean_true_logits)

    mean_fake_logits_stop = tf.reduce_mean(fake_logits_stop)
    tf.summary.scalar("mean_fake_logits_stop", mean_fake_logits_stop)

    discriminator_loss_func = hparam_to_disc_loss[self.hparams.gan_loss]
    gan_d_loss = discriminator_loss_func(
        discriminator_real_outputs=true_logits,
        discriminator_gen_outputs=fake_logits_stop,
        add_summaries=True)
    return gan_d_loss, true_logits, fake_logits_stop

  def g_step(self, gen_frames, fake_logits_stop):
    """Performs the generator step in computing the GAN loss.

    Args:
      gen_frames: Generated frames
      fake_logits_stop: Logits corresponding to the generated frames as per
                        the discriminator. Assumed to have a stop-gradient term.
    Returns:
      gan_g_loss_pos_d: Loss.
      gan_g_loss_neg_d: -gan_g_loss_pos_d but with a stop gradient on generator.
    """
    hparam_to_gen_loss = {
        "least_squares": gan_losses.least_squares_generator_loss,
        "cross_entropy": gan_losses.modified_generator_loss,
        "wasserstein": gan_losses.wasserstein_generator_loss
    }

    fake_logits = self.discriminator(gen_frames)
    mean_fake_logits = tf.reduce_mean(fake_logits)
    tf.summary.scalar("mean_fake_logits", mean_fake_logits)

    # Generator loss.
    # Using gan_g_loss_pos_d updates the discriminator as well.
    # To avoid this add gan_g_loss_neg_d = -gan_g_loss_pos_d
    # but with stop gradient on the generator.
    # This makes sure that the net gradient on the discriminator is zero and
    # net-gradient on the generator is just due to the gan_g_loss_pos_d.
    generator_loss_func = hparam_to_gen_loss[self.hparams.gan_loss]
    gan_g_loss_pos_d = generator_loss_func(
        discriminator_gen_outputs=fake_logits, add_summaries=True)
    gan_g_loss_neg_d = -generator_loss_func(
        discriminator_gen_outputs=fake_logits_stop, add_summaries=True)
    return gan_g_loss_pos_d, gan_g_loss_neg_d

  def get_gan_loss(self, true_frames, gen_frames, name):
    """Get the discriminator + generator loss at every step.

    This performs an 1:1 update of the discriminator and generator at every
    step.

    Args:
      true_frames: 5-D Tensor of shape (num_steps, batch_size, H, W, C)
                   Assumed to be ground truth.
      gen_frames: 5-D Tensor of shape (num_steps, batch_size, H, W, C)
                  Assumed to be fake.
      name: discriminator scope.
    Returns:
      loss: 0-D Tensor, with d_loss + g_loss
    """
    # D - STEP
    with tf.variable_scope("%s_discriminator" % name, reuse=tf.AUTO_REUSE):
      gan_d_loss, _, fake_logits_stop = self.d_step(
          true_frames, gen_frames)

    # G - STEP
    with tf.variable_scope("%s_discriminator" % name, reuse=True):
      gan_g_loss_pos_d, gan_g_loss_neg_d = self.g_step(
          gen_frames, fake_logits_stop)
    gan_g_loss = gan_g_loss_pos_d + gan_g_loss_neg_d
    tf.summary.scalar("gan_loss_%s" % name, gan_g_loss_pos_d + gan_d_loss)

    if self.hparams.gan_optimization == "joint":
      gan_loss = gan_g_loss + gan_d_loss
    else:
      curr_step = self.get_iteration_num()
      gan_loss = tf.cond(
          tf.logical_not(curr_step % 2 == 0), lambda: gan_g_loss,
          lambda: gan_d_loss)
    return gan_loss

  def get_extra_loss(self, latent_means=None, latent_stds=None,
                     true_frames=None, gen_frames=None):
    """Gets extra loss from VAE and GAN."""
    if not self.is_training:
      return 0.0

    vae_loss, d_vae_loss, d_gan_loss = 0.0, 0.0, 0.0
    # Use sv2p's KL divergence computation.
    if self.hparams.use_vae:
      vae_loss = super(NextFrameSavpBase, self).get_extra_loss(
          latent_means=latent_means, latent_stds=latent_stds)

    if self.hparams.use_gan:
      # Strip out the first context_frames for the true_frames
      # Strip out the first context_frames - 1 for the gen_frames
      context_frames = self.hparams.video_num_input_frames
      true_frames = tf.stack(
          tf.unstack(true_frames, axis=0)[context_frames:])

      # discriminator for VAE.
      if self.hparams.use_vae:
        gen_enc_frames = tf.stack(
            tf.unstack(gen_frames, axis=0)[context_frames-1:])
        d_vae_loss = self.get_gan_loss(true_frames, gen_enc_frames, name="vae")

      # discriminator for GAN.
      gen_prior_frames = tf.stack(
          tf.unstack(self.gen_prior_video, axis=0)[context_frames-1:])
      d_gan_loss = self.get_gan_loss(true_frames, gen_prior_frames, name="gan")

    return (
        vae_loss + self.hparams.gan_loss_multiplier * d_gan_loss +
        self.hparams.gan_vae_loss_multiplier * d_vae_loss)

  def pad_conv3d_lrelu(self, activations, n_filters, kernel_size, strides,
                       scope):
    """Pad, apply 3-D convolution and leaky relu."""
    padding = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]

    # tf.nn.conv3d accepts a list of 5 values for strides
    # with first and last value equal to 1
    if isinstance(strides, numbers.Integral):
      strides = [strides] * 3
    strides = [1] + strides + [1]

    # Filter_shape = [K, K, K, num_input, num_output]
    filter_shape = (
        [kernel_size]*3 + activations.shape[-1:].as_list() + [n_filters])

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      conv_filter = tf.get_variable(
          "conv_filter", shape=filter_shape,
          initializer=tf.truncated_normal_initializer(stddev=0.02))

      if self.hparams.use_spectral_norm:
        conv_filter, assign_op = common_layers.apply_spectral_norm(conv_filter)
        if self.is_training:
          tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign_op)

      padded = tf.pad(activations, padding)
      convolved = tf.nn.conv3d(
          padded, conv_filter, strides=strides, padding="VALID")
      rectified = tf.nn.leaky_relu(convolved, alpha=0.2)
    return rectified

  @staticmethod
  def train_hooks(hook_context):
    del hook_context
    return [update_ops_hook.UpdateOpsHook()]


@registry.register_model
class NextFrameSAVP(NextFrameSavpBase, sv2p.NextFrameSv2pLegacy):
  """Stochastic Adversarial Video Prediction."""

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

    Raises:
      ValueError: If not exactly one of self.hparams.vae or self.hparams.gan
                  is set to True.
    """
    if not self.hparams.use_vae and not self.hparams.use_gan:
      raise ValueError("Set at least one of use_vae or use_gan to be True")
    if self.hparams.gan_optimization not in ["joint", "sequential"]:
      raise ValueError("self.hparams.gan_optimization should be either joint "
                       "or sequential got %s" % self.hparams.gan_optimization)

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
        latent = common_video.get_gaussian_tensor(mu, log_sigma_sq)

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
        input_image, = self.get_scheduled_sample_inputs(
            done_warm_start, groundtruth_items, generated_items, ss_func)

        all_latents = tf.concat([enc_cond_latent, enc_prior_latent], axis=0)
        all_image = tf.concat([input_image, input_image], axis=0)
        all_action = tf.concat([action, action], axis=0)
        all_rewards = tf.concat([reward, reward], axis=0)

        all_pred_images, lstm_state, _ = self.construct_predictive_tower(
            all_image, all_rewards, all_action, lstm_state, all_latents,
            concat_latent=True)

        cond_pred_images, prior_pred_images = \
          all_pred_images[:batch_size], all_pred_images[batch_size:]

        if train_mode and self.hparams.use_vae:
          pred_image = cond_pred_images
        else:
          pred_image = prior_pred_images

        gen_cond_video.append(cond_pred_images)
        gen_prior_video.append(prior_pred_images)
        latent_means.append(mu)
        latent_stds.append(log_sigma_sq)

    gen_cond_video = tf.stack(gen_cond_video, axis=0)
    self.gen_prior_video = tf.stack(gen_prior_video, axis=0)
    fake_rewards = tf.stack(fake_rewards, axis=0)

    if train_mode and self.hparams.use_vae:
      return gen_cond_video, fake_rewards, latent_means, latent_stds
    else:
      return self.gen_prior_video, fake_rewards, latent_means, latent_stds


@registry.register_model
class NextFrameSavpRl(NextFrameSavpBase, sv2p.NextFrameSv2p):
  """Stochastic Adversarial Video Prediction for RL pipeline."""

  def video_features(
      self, all_frames, all_actions, all_rewards, all_raw_frames):
    """No video wide feature."""
    del all_actions, all_rewards, all_raw_frames
    # Concatenate x_{t-1} and x_{t} along depth and encode it to
    # produce the mean and standard deviation of z_{t-1}
    seq_len = len(all_frames)
    image_pairs = tf.concat([all_frames[:seq_len-1],
                             all_frames[1:seq_len]], axis=-1)
    z_mu, z_log_sigma_sq = self.encoder(image_pairs)
    # Unstack z_mu and z_log_sigma_sq along the time dimension.
    z_mu = tf.unstack(z_mu, axis=0)
    z_log_sigma_sq = tf.unstack(z_log_sigma_sq, axis=0)
    return [z_mu, z_log_sigma_sq]

  def video_extra_loss(self, frames_predicted, frames_target,
                       internal_states, video_features):

    if not self.is_training:
      return 0.0

    latent_means, latent_stds = video_features
    true_frames, gen_frames = frames_target, frames_predicted

    loss = super(NextFrameSavpRl, self).get_extra_loss(
        latent_means=latent_means, latent_stds=latent_stds,
        true_frames=true_frames, gen_frames=gen_frames)
    return loss

  def next_frame(self, frames, actions, rewards, target_frame,
                 internal_states, video_features):
    del target_frame

    if not self.hparams.use_vae or self.hparams.use_gan:
      raise NotImplementedError("Only supporting VAE for now.")

    if self.has_pred_actions or self.has_values:
      raise NotImplementedError("Parameter sharing with policy not supported.")

    image, action, reward = frames[0], actions[0], rewards[0]
    latent_dims = self.hparams.z_dim
    batch_size = common_layers.shape_list(image)[0]

    if internal_states is None:
      # Initialize LSTM State
      frame_index = 0
      lstm_state = [None] * 7
      cond_latent_state, prior_latent_state = None, None
      gen_prior_video = []
    else:
      (frame_index, lstm_state, cond_latent_state,
       prior_latent_state, gen_prior_video) = internal_states

    z_mu, log_sigma_sq = video_features
    z_mu, log_sigma_sq = z_mu[frame_index], log_sigma_sq[frame_index]

    # Sample latents using a gaussian centered at conditional mu and std.
    latent = common_video.get_gaussian_tensor(z_mu, log_sigma_sq)

    # Sample prior latents from isotropic normal distribution.
    prior_latent = tf.random_normal(tf.shape(latent), dtype=tf.float32)

    # # LSTM that encodes correlations between conditional latents.
    # # Pg 22 in https://arxiv.org/pdf/1804.01523.pdf
    enc_cond_latent, cond_latent_state = common_video.basic_lstm(
        latent, cond_latent_state, latent_dims, name="cond_latent")

    # LSTM that encodes correlations between prior latents.
    enc_prior_latent, prior_latent_state = common_video.basic_lstm(
        prior_latent, prior_latent_state, latent_dims, name="prior_latent")

    all_latents = tf.concat([enc_cond_latent, enc_prior_latent], axis=0)
    all_image = tf.concat([image, image], 0)
    all_action = tf.concat([action, action], 0) if self.has_actions else None

    all_pred_images, lstm_state = self.construct_predictive_tower(
        all_image, None, all_action, lstm_state, all_latents,
        concat_latent=True)

    cond_pred_images, prior_pred_images = \
      all_pred_images[:batch_size], all_pred_images[batch_size:]

    if self.is_training and self.hparams.use_vae:
      pred_image = cond_pred_images
    else:
      pred_image = prior_pred_images

    gen_prior_video.append(prior_pred_images)
    internal_states = (frame_index + 1, lstm_state, cond_latent_state,
                       prior_latent_state, gen_prior_video)

    if not self.has_rewards:
      return pred_image, None, 0.0, internal_states

    pred_reward = self.reward_prediction(
        pred_image, action, reward, latent)
    return pred_image, pred_reward, None, None, 0.0, internal_states
