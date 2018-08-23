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
"""Param sets for SV2P model."""

from __future__ import division
from __future__ import print_function

from tensor2tensor.models.research import next_frame_basic_stochastic
from tensor2tensor.utils import registry


@registry.register_hparams
def next_frame_sv2p():
  """SV2P model hparams."""
  hparams = next_frame_basic_stochastic.next_frame_basic_stochastic()
  hparams.optimizer = "TrueAdam"
  hparams.learning_rate_schedule = "constant"
  hparams.learning_rate_constant = 1e-3
  hparams.video_num_input_frames = 1
  hparams.video_num_target_frames = 3
  hparams.batch_size = 16
  hparams.target_modality = "video:l2raw"
  hparams.input_modalities = "inputs:video:l2raw"
  hparams.video_modality_loss_cutoff = 0.0
  hparams.add_hparam("reward_prediction", True)
  hparams.add_hparam("reward_prediction_stop_gradient", True)
  hparams.add_hparam("reward_prediction_buffer_size", 0)
  hparams.add_hparam("model_options", "CDNA")
  hparams.add_hparam("num_masks", 10)
  hparams.add_hparam("multi_latent", False)
  hparams.add_hparam("relu_shift", 1e-12)
  hparams.add_hparam("dna_kernel_size", 5)
  # Scheduled sampling method. Choose between prob or count.
  hparams.add_hparam("scheduled_sampling_mode", "count")
  hparams.add_hparam("scheduled_sampling_decay_steps", 10000)
  hparams.add_hparam("scheduled_sampling_k", 900.0)
  hparams.add_hparam("upsample_method", "conv2d_transpose")
  hparams.add_hparam("internal_loss", False)
  return hparams


@registry.register_hparams
def next_frame_sv2p_tiny():
  """Tiny SV2P model."""
  hparams = next_frame_sv2p()
  hparams.batch_size = 2
  hparams.tiny_mode = True
  hparams.num_masks = 1
  hparams.video_modality_loss_cutoff = 0.4
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 1
  return hparams


@registry.register_hparams
def next_frame_sv2p_cutoff():
  """SV2P model with additional cutoff in L2 loss for environments like pong."""
  hparams = next_frame_sv2p()
  hparams.video_modality_loss_cutoff = 0.4
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 1
  return hparams

