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

"""Mujoco Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from tensor2tensor.envs import rendered_env_problem
from tensor2tensor.layers import modalities
from tensor2tensor.rl import gym_utils
from tensor2tensor.utils import registry



@registry.register_env_problem
class ReacherEnvProblem(rendered_env_problem.RenderedEnvProblem):
  """Mujoco's reacher environment."""

  def __init__(self):
    base_env_name = "Reacher-v2"
    wrapper_fn = functools.partial(
        gym_utils.gym_env_wrapper, **{
            "rl_env_max_episode_steps": -1,
            "maxskip_env": False,
            "rendered_env": True,
            "rendered_env_resize_to": None,  # Do not resize frames
            "sticky_actions": False,
            "output_dtype": None,
            "num_actions": None,
        })
    super(ReacherEnvProblem, self).__init__(
        base_env_name=base_env_name, env_wrapper_fn=wrapper_fn)

  @property
  def input_modality(self):
    return modalities.ModalityType.VIDEO

  @property
  def target_modality(self):
    return modalities.ModalityType.VIDEO

  @property
  def action_modality(self):
    return modalities.ModalityType.IDENTITY

  @property
  def reward_modality(self):
    return modalities.ModalityType.IDENTITY

  @property
  def input_vocab_size(self):
    return 256

  @property
  def target_vocab_size(self):
    return 256
