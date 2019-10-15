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

"""Basic recurrent models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_video
from tensor2tensor.models.video import basic_stochastic
from tensor2tensor.utils import registry


@registry.register_model
class NextFrameBasicRecurrent(
    basic_stochastic.NextFrameBasicStochasticDiscrete):
  """Basic next-frame recurrent model."""

  @property
  def is_recurrent_model(self):
    return True

  def middle_network(self, layer, internal_states):
    lstm_func = common_video.conv_lstm_2d
    hp = self.hparams

    lstm_states = internal_states
    if lstm_states is None:
      lstm_states = [None] * hp.num_lstm_layers

    # LSTM layers
    x = layer
    for j in range(hp.num_lstm_layers):
      x, lstm_states[j] = lstm_func(x, lstm_states[j], hp.num_lstm_filters)
    return x, lstm_states


@registry.register_hparams
def next_frame_basic_recurrent():
  """Basic 2-frame recurrent model with stochastic tower."""
  hparams = basic_stochastic.next_frame_basic_stochastic_discrete()
  hparams.filter_double_steps = 2
  hparams.hidden_size = 64
  hparams.video_num_input_frames = 4
  hparams.video_num_target_frames = 4
  hparams.concat_internal_states = False
  hparams.add_hparam("num_lstm_layers", 2)
  hparams.add_hparam("num_lstm_filters", 256)
  return hparams
