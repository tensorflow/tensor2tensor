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

"""Param sets for SV2P model."""

from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import modalities
from tensor2tensor.models.video import basic_stochastic
from tensor2tensor.utils import registry


@registry.register_hparams
def next_frame_sv2p():
  """SV2P model hparams."""
  hparams = basic_stochastic.next_frame_basic_stochastic()
  hparams.optimizer = "true_adam"
  hparams.learning_rate_schedule = "constant"
  hparams.learning_rate_constant = 1e-3
  hparams.video_num_input_frames = 1
  hparams.video_num_target_frames = 3
  hparams.batch_size = 16
  hparams.bottom = {
      "inputs": modalities.video_raw_bottom,
      "targets": modalities.video_raw_targets_bottom,
  }
  hparams.loss = {
      "targets": modalities.video_l2_raw_loss,
  }
  hparams.top = {
      "targets": modalities.video_raw_top,
  }
  hparams.video_modality_loss_cutoff = 0.0
  hparams.scheduled_sampling_mode = "count"
  hparams.scheduled_sampling_k = 900.0
  hparams.add_hparam("reward_prediction", True)
  hparams.add_hparam("reward_prediction_stop_gradient", False)
  hparams.add_hparam("reward_prediction_buffer_size", 0)
  hparams.add_hparam("model_options", "CDNA")
  hparams.add_hparam("num_masks", 10)
  hparams.add_hparam("multi_latent", False)
  hparams.add_hparam("relu_shift", 1e-12)
  hparams.add_hparam("dna_kernel_size", 5)
  hparams.add_hparam("upsample_method", "conv2d_transpose")
  hparams.add_hparam("reward_model", "basic")
  hparams.add_hparam("visualize_logits_histogram", True)
  return hparams


@registry.register_hparams
def next_frame_sv2p_discrete():
  """SV2P discrete model hparams."""
  hparams = next_frame_sv2p()
  hparams.action_injection = "multiplicative"
  hparams.small_mode = True
  hparams.add_hparam("bottleneck_bits", 128)
  hparams.add_hparam("bottleneck_noise", 0.02)
  hparams.add_hparam("discrete_warmup_steps", 40000)
  hparams.add_hparam("full_latent_tower", False)
  hparams.add_hparam("latent_predictor_state_size", 128)
  hparams.add_hparam("latent_predictor_temperature", 0.5)
  hparams.add_hparam("discretize_warmup_steps", 40000)
  return hparams


@registry.register_hparams
def next_frame_sv2p_atari():
  """SV2P model for atari."""
  hparams = next_frame_sv2p()
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 4
  hparams.action_injection = "multiplicative"
  hparams.num_iterations_1st_stage = 12000
  hparams.num_iterations_2nd_stage = 12000
  hparams.anneal_end = 40000
  hparams.latent_loss_multiplier_schedule = "noisy_linear_cosine_decay"
  hparams.latent_loss_multiplier = 1e-3
  hparams.information_capacity = 0.0
  hparams.small_mode = True
  return hparams


@registry.register_hparams
def next_frame_sv2p_atari_softmax():
  """SV2P model for atari with softmax."""
  hparams = next_frame_sv2p_atari()
  hparams.bottom = {}
  hparams.loss = {}
  hparams.top = {}
  hparams.internal_loss = True
  return hparams


@registry.register_hparams
def next_frame_sv2p_atari_deterministic():
  """Deterministic for atari."""
  hparams = next_frame_sv2p_atari()
  hparams.stochastic_model = False
  return hparams


@registry.register_hparams
def next_frame_sv2p_atari_softmax_deterministic():
  """Deterministic for atari."""
  hparams = next_frame_sv2p_atari_softmax()
  hparams.stochastic_model = False
  return hparams


@registry.register_hparams
def next_frame_sv2p_tiny():
  """Tiny SV2P model."""
  hparams = next_frame_sv2p_atari_softmax()
  hparams.batch_size = 2
  hparams.tiny_mode = True
  hparams.num_masks = 1
  hparams.video_modality_loss_cutoff = 0.4
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 4
  return hparams


@registry.register_hparams
def next_frame_sv2p_tiny_external():
  """Tiny SV2P model with external loss."""
  hparams = next_frame_sv2p_tiny()
  hparams.internal_loss = False
  return hparams


@registry.register_hparams
def next_frame_sv2p_cutoff():
  """SV2P model with additional cutoff in L2 loss for environments like pong."""
  hparams = next_frame_sv2p()
  hparams.video_modality_loss_cutoff = 0.4
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 1
  return hparams
