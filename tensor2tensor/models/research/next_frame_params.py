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
"""Param sets for next frame prediction models."""

from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry


@registry.register_hparams
def next_frame():
  """Basic 2-frame conv model."""
  hparams = common_hparams.basic_params1()
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 1
  hparams.hidden_size = 64
  hparams.batch_size = 4
  hparams.num_hidden_layers = 2
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_constant = 1.5
  hparams.learning_rate_warmup_steps = 1500
  hparams.learning_rate_schedule = "linear_warmup * constant * rsqrt_decay"
  hparams.label_smoothing = 0.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.3
  hparams.weight_decay = 0.0
  hparams.clip_grad_norm = 1.0
  hparams.dropout = 0.5
  hparams.add_hparam("num_compress_steps", 6)
  hparams.add_hparam("filter_double_steps", 2)
  hparams.add_hparam("video_modality_loss_cutoff", 0.02)
  hparams.add_hparam("preprocess_resize_frames", None)
  hparams.add_hparam("concatenate_actions", True)
  hparams.add_hparam("tiny_mode", False)
  hparams.add_hparam("shuffle_buffer_size", 128)
  return hparams


@registry.register_hparams
def next_frame_pixel_noise():
  """Basic 2-frame conv model with pixel noise."""
  hparams = next_frame()
  hparams.add_hparam("video_modality_input_noise", 0.05)
  hparams.input_modalities = "inputs:video:pixel_noise"
  return hparams


@registry.register_hparams
def next_frame_stochastic():
  """SV2P model."""
  hparams = next_frame()
  hparams.optimizer = "TrueAdam"
  hparams.learning_rate_schedule = "constant"
  hparams.learning_rate_constant = 1e-3
  hparams.video_num_input_frames = 1
  hparams.video_num_target_frames = 3
  hparams.batch_size = 16
  hparams.target_modality = "video:l2raw"
  hparams.input_modalities = "inputs:video:l2raw"
  hparams.video_modality_loss_cutoff = 0.0
  hparams.add_hparam("stochastic_model", True)
  hparams.add_hparam("reward_prediction", True)
  hparams.add_hparam("reward_prediction_stop_gradient", True)
  hparams.add_hparam("model_options", "CDNA")
  hparams.add_hparam("num_masks", 10)
  hparams.add_hparam("latent_channels", 1)
  hparams.add_hparam("latent_std_min", -5.0)
  hparams.add_hparam("num_iterations_1st_stage", 10000)
  hparams.add_hparam("num_iterations_2nd_stage", 10000)
  hparams.add_hparam("latent_loss_multiplier", 1e-3)
  hparams.add_hparam("latent_loss_multiplier_schedule", "constant")
  hparams.add_hparam("multi_latent", False)
  hparams.add_hparam("relu_shift", 1e-12)
  hparams.add_hparam("dna_kernel_size", 5)
  # Scheduled sampling method. Choose between prob or count.
  hparams.add_hparam("scheduled_sampling_mode", "count")
  hparams.add_hparam("scheduled_sampling_decay_steps", 10000)
  hparams.add_hparam("scheduled_sampling_k", 900.0)
  hparams.add_hparam("latent_num_frames", 0)  # 0 means use all frames.
  hparams.add_hparam("anneal_end", 100000)
  hparams.add_hparam("upsample_method", "conv2d_transpose")
  hparams.add_hparam("internal_loss", False)
  return hparams


@registry.register_hparams
def next_frame_stochastic_emily():
  """Emily's model."""
  hparams = next_frame_stochastic()
  hparams.latent_loss_multiplier = 1e-4
  hparams.learning_rate_constant = 0.002
  hparams.add_hparam("z_dim", 10)
  hparams.add_hparam("g_dim", 128)
  hparams.add_hparam("rnn_size", 256)
  hparams.add_hparam("posterior_rnn_layers", 1)
  hparams.add_hparam("predictor_rnn_layers", 2)
  return hparams


@registry.register_hparams
def next_frame_savp():
  """SAVP model."""
  hparams = next_frame_stochastic()
  hparams.add_hparam("z_dim", 8)
  hparams.add_hparam("num_discriminator_filters", 32)
  hparams.add_hparam("use_vae", True)
  hparams.add_hparam("use_gan", False)
  hparams.add_hparam("use_spectral_norm", True)
  hparams.add_hparam("gan_loss", "cross_entropy")
  hparams.add_hparam("gan_loss_multiplier", 0.01)
  hparams.add_hparam("gan_vae_loss_multiplier", 0.01)
  hparams.add_hparam("gan_optimization", "joint")
  hparams.target_modality = "video:l1raw"
  hparams.input_modalities = "inputs:video:l1raw"
  hparams.latent_loss_multiplier_schedule = "linear_anneal"
  hparams.anneal_end = 100000
  hparams.upsample_method = "bilinear_upsample_conv"
  return hparams


@registry.register_hparams
def next_frame_stochastic_cutoff():
  """SV2P model with additional cutoff in L2 loss for environments like pong."""
  hparams = next_frame_stochastic()
  hparams.video_modality_loss_cutoff = 0.4
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 1
  return hparams


@registry.register_hparams
def next_frame_stochastic_tiny():
  """SV2P model with additional cutoff in L2 loss for environments like pong."""
  hparams = next_frame_stochastic()
  hparams.batch_size = 2
  hparams.tiny_mode = True
  hparams.num_masks = 1
  hparams.video_modality_loss_cutoff = 0.4
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 1
  return hparams


@registry.register_hparams
def next_frame_tpu():
  hparams = next_frame()
  hparams.batch_size = 1
  return hparams


@registry.register_hparams
def next_frame_ae():
  """Conv autoencoder."""
  hparams = next_frame()
  hparams.input_modalities = "inputs:video:bitwise"
  hparams.hidden_size = 256
  hparams.batch_size = 8
  hparams.num_hidden_layers = 4
  hparams.num_compress_steps = 4
  hparams.dropout = 0.4
  return hparams


@registry.register_hparams
def next_frame_small():
  """Small conv model."""
  hparams = next_frame()
  hparams.hidden_size = 32
  return hparams


@registry.register_hparams
def next_frame_tiny():
  """Tiny for testing."""
  hparams = next_frame()
  hparams.hidden_size = 32
  hparams.num_hidden_layers = 1
  hparams.num_compress_steps = 2
  hparams.filter_double_steps = 1
  return hparams


@registry.register_hparams
def next_frame_l1():
  """Basic conv model with L1 modality."""
  hparams = next_frame()
  hparams.target_modality = "video:l1"
  hparams.video_modality_loss_cutoff = 2.4
  return hparams


@registry.register_hparams
def next_frame_l2():
  """Basic conv model with L2 modality."""
  hparams = next_frame()
  hparams.target_modality = "video:l2"
  hparams.video_modality_loss_cutoff = 2.4
  return hparams


@registry.register_ranged_hparams
def next_frame_base_range(rhp):
  """Basic tuning grid."""
  rhp.set_float("dropout", 0.2, 0.6)
  rhp.set_discrete("hidden_size", [64, 128, 256])
  rhp.set_int("num_compress_steps", 5, 8)
  rhp.set_discrete("batch_size", [4, 8, 16, 32])
  rhp.set_int("num_hidden_layers", 1, 3)
  rhp.set_int("filter_double_steps", 1, 6)
  rhp.set_float("learning_rate_constant", 1., 4.)
  rhp.set_int("learning_rate_warmup_steps", 500, 3000)
  rhp.set_float("initializer_gain", 0.8, 1.8)


@registry.register_ranged_hparams
def next_frame_doubling_range(rhp):
  """Filter doubling and dropout tuning grid."""
  rhp.set_float("dropout", 0.2, 0.6)
  rhp.set_int("filter_double_steps", 2, 5)


@registry.register_ranged_hparams
def next_frame_clipgrad_range(rhp):
  """Filter doubling and dropout tuning grid."""
  rhp.set_float("dropout", 0.3, 0.4)
  rhp.set_float("clip_grad_norm", 0.5, 10.0)


@registry.register_ranged_hparams
def next_frame_xent_cutoff_range(rhp):
  """Cross-entropy tuning grid."""
  rhp.set_float("video_modality_loss_cutoff", 0.005, 0.05)


@registry.register_ranged_hparams
def next_frame_ae_range(rhp):
  """Autoencoder world model tuning grid."""
  rhp.set_float("dropout", 0.3, 0.5)
  rhp.set_int("num_compress_steps", 1, 3)
  rhp.set_int("num_hidden_layers", 2, 6)
  rhp.set_float("learning_rate_constant", 1., 2.)
  rhp.set_float("initializer_gain", 0.8, 1.5)
  rhp.set_int("filter_double_steps", 2, 3)
