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

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video

import tensorflow as tf


class NextFrameBaseVae(object):
  """Basic function for stochastic variational video prediction."""

  def __init__(self, hparams):
    self.hparams = hparams

  def get_iteration_num(self):
    step_num = tf.train.get_global_step()
    # TODO(lukaszkaiser): what should it be if it's undefined?
    if step_num is None:
      step_num = 1000000
    return step_num

  def get_beta(self, kl_loss=0.0):
    """Get the KL multiplier, either dynamically or schedule based.

    if hparams.latent_loss_multiplier_dynamic is set to true, then beta
    is being adjusted to keep KL under hparams.latent_loss_multiplier_epsilon.
    In order to do so, the beta is being updated at each iteration
    by taking steps of size hparams.latent_loss_multiplier_alpha.
    The same formulation can be retrieved by solving the Lagrangian
    with KL < epsilon as a constraint.

    Args:
      kl_loss: KL loss. Only used for dynamic adjustment.

    Returns:
      beta: the final value of beta.

    """
    if self.hparams.latent_loss_multiplier_dynamic:
      beta = tf.Variable(self.hparams.latent_loss_multiplier,
                         trainable=False, dtype=tf.float32)
      alpha = self.hparams.latent_loss_multiplier_alpha
      epsilon = self.hparams.latent_loss_multiplier_epsilon
      shadow_beta = beta + alpha * (kl_loss - epsilon)
      # Caping the beta between 0 and 1. May need to change this later on.
      shadow_beta = tf.maximum(shadow_beta, 0.0)
      shadow_beta = tf.minimum(shadow_beta, 1.0)
      update_op = tf.assign(beta, shadow_beta)
    else:
      beta = common_video.beta_schedule(
          schedule=self.hparams.latent_loss_multiplier_schedule,
          global_step=self.get_iteration_num(),
          final_beta=self.hparams.latent_loss_multiplier,
          decay_start=(self.hparams.num_iterations_1st_stage +
                       self.hparams.num_iterations_2nd_stage),
          decay_end=self.hparams.anneal_end)
      update_op = tf.identity(beta)  # fake update for regular beta.
    with tf.control_dependencies([update_op]):
      tf.summary.scalar("beta", beta)
      return beta

  def get_extra_loss(self, mean, std):
    """Losses in addition to the default modality losses."""
    kl_loss = common_layers.kl_divergence(mean, std)
    beta = self.get_beta(kl_loss)
    tf.summary.histogram("posterior_mean", mean)
    tf.summary.histogram("posterior_std", std)
    tf.summary.scalar("kl_raw", tf.reduce_mean(kl_loss))
    # information capacity from "Understanding disentangling in beta-VAE"
    if self.hparams.information_capacity > 0.0:
      kl_loss = tf.abs(kl_loss - self.hparams.information_capacity)
    return beta * kl_loss

  def construct_latent_tower(self, images, time_axis):
    """Create the latent tower."""
    # No latent in the first phase
    first_phase = tf.less(
        self.get_iteration_num(), self.hparams.num_iterations_1st_stage)

    # use all frames by default but this allows more
    # predicted frames at inference time
    latent_num_frames = self.hparams.latent_num_frames
    tf.logging.info("Creating latent tower with %d frames." % latent_num_frames)
    if latent_num_frames > 0:
      images = images[:latent_num_frames]

    return common_video.conv_latent_tower(
        images=images,
        time_axis=time_axis,
        latent_channels=self.hparams.latent_channels,
        min_logvar=self.hparams.latent_std_min,
        is_training=self.is_training,
        random_latent=first_phase,
        tiny_mode=self.hparams.tiny_mode,
        small_mode=self.hparams.small_mode)



