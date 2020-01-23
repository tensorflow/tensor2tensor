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

"""Unified interface for different RL algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class PolicyLearner(object):
  """API for policy learners."""

  def __init__(
      self, frame_stack_size, base_event_dir, agent_model_dir, total_num_epochs
  ):
    self.frame_stack_size = frame_stack_size
    self.base_event_dir = base_event_dir
    self.agent_model_dir = agent_model_dir
    self.total_num_epochs = total_num_epochs

  def train(
      self,
      env_fn,
      hparams,
      simulated,
      save_continuously,
      epoch,
      sampling_temp=1.0,
      num_env_steps=None,
      env_step_multiplier=1,
      eval_env_fn=None,
      report_fn=None
  ):
    """Train."""
    raise NotImplementedError()

  def evaluate(self, env_fn, hparams, sampling_temp):
    raise NotImplementedError()
