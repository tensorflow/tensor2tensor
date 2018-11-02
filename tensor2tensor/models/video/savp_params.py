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

"""Param sets for SAVP model."""

from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import modalities
from tensor2tensor.models.video import sv2p_params
from tensor2tensor.utils import registry


@registry.register_hparams
def next_frame_savp():
  """SAVP model hparams."""
  hparams = sv2p_params.next_frame_sv2p()
  hparams.add_hparam("z_dim", 8)
  hparams.add_hparam("num_discriminator_filters", 32)
  hparams.add_hparam("use_vae", True)
  hparams.add_hparam("use_gan", False)
  hparams.add_hparam("use_spectral_norm", True)
  hparams.add_hparam("gan_loss", "cross_entropy")
  hparams.add_hparam("gan_loss_multiplier", 0.01)
  hparams.add_hparam("gan_vae_loss_multiplier", 0.01)
  hparams.add_hparam("gan_optimization", "joint")
  hparams.modality = {
      "inputs": modalities.VideoModalityL1Raw,
      "targets": modalities.VideoModalityL1Raw,
  }
  hparams.latent_loss_multiplier_schedule = "linear"
  hparams.upsample_method = "bilinear_upsample_conv"
  hparams.internal_loss = False
  hparams.reward_prediction = False
  return hparams


@registry.register_hparams
def next_frame_savp_vae():
  """SAVP - VAE only model."""
  hparams = next_frame_savp()
  hparams.use_vae = True
  hparams.use_gan = False
  hparams.latent_loss_multiplier = 1e-3
  hparams.latent_loss_multiplier_schedule = "linear_anneal"
  hparams.anneal_end = 100000
  return hparams


@registry.register_hparams
def next_frame_savp_gan():
  """SAVP - GAN only model."""
  hparams = next_frame_savp()
  hparams.use_gan = True
  hparams.use_vae = False
  hparams.gan_loss_multiplier = 0.001
  hparams.optimizer_adam_beta1 = 0.5
  hparams.learning_rate_constant = 2e-4
  hparams.gan_loss = "cross_entropy"
  hparams.learning_rate_decay_steps = 100000
  hparams.learning_rate_schedule = "constant*linear_decay"
  return hparams
