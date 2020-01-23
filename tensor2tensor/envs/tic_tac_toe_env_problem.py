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

"""TicTacToeEnvProblem wraps the TicTacToeEnv in an EnvProblem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.envs import gym_env_problem
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry


@registry.register_env_problem
class TicTacToeEnvProblem(gym_env_problem.GymEnvProblem):
  """Plays `batch_size` games of tic-tac-toe."""

  def __init__(self):
    super(TicTacToeEnvProblem, self).__init__(
        base_env_name="T2TEnv-TicTacToeEnv-v0",
        reward_range=(-1, 1))

  @property
  def input_modality(self):
    return modalities.ModalityType.IDENTITY_SYMBOL

  @property
  def input_vocab_size(self):
    # Since a box can be either x or o or empty.
    return 3

  @property
  def target_modality(self):
    return modalities.ModalityType.IDENTITY_SYMBOL

  @property
  def target_vocab_size(self):
    # Since reward is either -1 or 0 or +1.
    return 3

  @property
  def action_modality(self):
    return modalities.ModalityType.SYMBOL_WEIGHTS_ALL
