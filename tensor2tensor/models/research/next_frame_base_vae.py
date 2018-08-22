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

  def get_beta(self):
    beta = common_video.beta_schedule(
        schedule=self.hparams.latent_loss_multiplier_schedule,
        global_step=self.get_iteration_num(),
        final_beta=self.hparams.latent_loss_multiplier,
        decay_start=(self.hparams.num_iterations_1st_stage +
                     self.hparams.num_iterations_2nd_stage),
        decay_end=self.hparams.anneal_end)
    tf.summary.scalar("beta", beta)
    return beta

  def get_extra_loss(self, mean, std):
    """Losses in addition to the default modality losses."""
    if self.is_training:
      beta = self.get_beta()
      kl_loss = common_layers.kl_divergence(mean, std)
      tf.summary.histogram("posterior_mean", mean)
      tf.summary.histogram("posterior_std", std)
      tf.summary.scalar("kl_raw", tf.reduce_mean(kl_loss))
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
        tiny_mode=self.hparams.tiny_mode)



