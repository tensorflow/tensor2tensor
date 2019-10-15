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

"""Param sets for deterministic basic next frame prediction model."""

from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import modalities
from tensor2tensor.models.video import base
from tensor2tensor.utils import registry


@registry.register_hparams
def next_frame_basic_deterministic():
  """Basic 2-frame conv model."""
  hparams = base.next_frame_base()
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 1
  hparams.hidden_size = 64
  hparams.batch_size = 4
  hparams.num_hidden_layers = 2
  hparams.optimizer = "Adafactor"
  hparams.learning_rate_constant = 1.5
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate_schedule = "linear_warmup * constant * rsqrt_decay"
  hparams.label_smoothing = 0.0
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 1.3
  hparams.weight_decay = 0.0
  hparams.clip_grad_norm = 1.0
  hparams.dropout = 0.1
  hparams.add_hparam("residual_dropout", 0.5)
  hparams.add_hparam("num_compress_steps", 6)
  hparams.add_hparam("filter_double_steps", 2)
  hparams.add_hparam("pixel_sampling_temperature", 0.0)
  hparams.add_hparam("concat_internal_states", False)
  hparams.add_hparam("do_autoregressive_rnn", False)
  hparams.add_hparam("autoregressive_rnn_lookback", 8)
  hparams.add_hparam("autoregressive_rnn_warmup_steps", 8000)
  hparams.add_hparam("activation_fn", "relu")
  hparams.bottom["inputs"] = modalities.video_identity_bottom
  hparams.bottom["targets"] = modalities.video_identity_bottom
  return hparams


@registry.register_hparams
def next_frame_pixel_noise():
  """Basic 2-frame conv model with pixel noise."""
  hparams = next_frame_basic_deterministic()
  hparams.add_hparam("video_modality_input_noise", 0.05)
  hparams.bottom["inputs"] = modalities.video_pixel_noise_bottom
  hparams.top["inputs"] = modalities.video_top
  return hparams


@registry.register_hparams
def next_frame_pixel_noise_long():
  """Long scheduled sampling setting."""
  hparams = next_frame_pixel_noise()
  hparams.batch_size = 2
  hparams.video_num_target_frames = 16
  return hparams


@registry.register_hparams
def next_frame_sampling():
  """Basic conv model with scheduled sampling."""
  hparams = next_frame_basic_deterministic()
  hparams.scheduled_sampling_mode = "prob_inverse_exp"
  hparams.scheduled_sampling_max_prob = 1.0
  hparams.scheduled_sampling_decay_steps = 10000
  return hparams


@registry.register_hparams
def next_frame_tpu():
  hparams = next_frame_basic_deterministic()
  hparams.batch_size = 1
  return hparams


@registry.register_hparams
def next_frame_ae():
  """Conv autoencoder."""
  hparams = next_frame_basic_deterministic()
  hparams.bottom["inputs"] = modalities.video_bitwise_bottom
  hparams.top["inputs"] = modalities.video_top
  hparams.hidden_size = 256
  hparams.batch_size = 8
  hparams.num_hidden_layers = 4
  hparams.num_compress_steps = 4
  hparams.dropout = 0.4
  return hparams


@registry.register_hparams
def next_frame_ae_tiny():
  """Conv autoencoder, tiny set for testing."""
  hparams = next_frame_tiny()
  hparams.bottom["inputs"] = modalities.video_bitwise_bottom
  hparams.top["inputs"] = modalities.video_top
  hparams.batch_size = 8
  hparams.dropout = 0.4
  return hparams


@registry.register_hparams
def next_frame_small():
  """Small conv model."""
  hparams = next_frame_basic_deterministic()
  hparams.hidden_size = 32
  return hparams


@registry.register_hparams
def next_frame_tiny():
  """Tiny for testing."""
  hparams = next_frame_basic_deterministic()
  hparams.hidden_size = 32
  hparams.num_hidden_layers = 1
  hparams.num_compress_steps = 2
  hparams.filter_double_steps = 1
  return hparams


@registry.register_hparams
def next_frame_l1():
  """Basic conv model with L1 modality."""
  hparams = next_frame_basic_deterministic()
  hparams.loss["targets"] = modalities.video_l1_loss
  hparams.top["targets"] = modalities.video_l1_top
  hparams.video_modality_loss_cutoff = 2.4
  return hparams


@registry.register_hparams
def next_frame_l2():
  """Basic conv model with L2 modality."""
  hparams = next_frame_basic_deterministic()
  hparams.loss["targets"] = modalities.video_l2_loss
  hparams.top["targets"] = modalities.video_l1_top
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
