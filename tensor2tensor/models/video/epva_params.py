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

"""Param sets for EPVA model."""

from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import modalities
from tensor2tensor.models.video import basic_deterministic_params
from tensor2tensor.utils import registry


@registry.register_hparams
def next_frame_epva():
  """EPVA hparams."""
  hparams = basic_deterministic_params.next_frame_basic_deterministic()
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 4
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
  hparams.learning_rate_schedule = "constant"
  hparams.learning_rate_constant = 1e-05
  hparams.batch_size = 2
  hparams.clip_grad_norm = 0.01
  # TODO(msaffar): disentangle EPVA from SV2P
  hparams.add_hparam("reward_prediction", False)
  hparams.add_hparam("clip_pixel_values", True)
  hparams.add_hparam("context_frames", 5)
  hparams.add_hparam("enc_learning_rate", 1e-5)
  hparams.add_hparam("enc_pred_loss_scale", 0.1)
  hparams.add_hparam("enc_pred_loss_scale_delay", 6e5)
  hparams.add_hparam("enc_size", 64)
  hparams.add_hparam("enc_keep_prob", .65)
  hparams.add_hparam("enc_pred_use_l1_loss", False)
  hparams.add_hparam("enc_pred_use_l2norm", False)
  hparams.add_hparam("van_learning_rate", 3e-5)
  hparams.add_hparam("van_keep_prob", .9)
  hparams.add_hparam("sequence_length ", 64)
  hparams.add_hparam("skip_num", 2)
  hparams.add_hparam("pred_noise_std", 0)
  hparams.add_hparam("lstm_state_noise_stddev", 0)
  return hparams
