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

"""Unified interface for different RL algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.rl import rl_trainer_lib


class PolicyLearner(object):
  """API for policy learners."""

  def __init__(self, frame_stack_size, base_event_dir, agent_model_dir):
    self.frame_stack_size = frame_stack_size
    self.base_event_dir = base_event_dir
    self.agent_model_dir = agent_model_dir

  def train(
      self, env_fn, hparams, num_env_steps, simulated, save_continuously,
      epoch, eval_env_fn=None
  ):
    # TODO(konradczechowski): move 'simulated' to  batch_env
    raise NotImplementedError()

  def evaluate(self, env_fn, hparams, stochastic):
    raise NotImplementedError()


class PPOLearner(PolicyLearner):
  """PPO for policy learning."""

  def __init__(self, *args, **kwargs):
    super(PPOLearner, self).__init__(*args, **kwargs)
    self._num_completed_iterations = 0

  def train(
      self, env_fn, hparams, num_env_steps, simulated, save_continuously,
      epoch, eval_env_fn=None
  ):
    self._num_completed_iterations += num_env_steps // (
        hparams.num_agents * hparams.epoch_length
    )

    if not save_continuously:
      # We do not save model, as that resets frames that we need at restarts.
      # But we need to save at the last step, so we set it very high.
      hparams.save_models_every_epochs = 1000000

    if simulated:
      simulated_str = "sim"
    else:
      simulated_str = "real"
    name_scope = "ppo_{}{}".format(simulated_str, epoch + 1)
    event_dir = os.path.join(
        self.base_event_dir, "ppo_summaries", str(epoch) + simulated_str
    )

    rl_trainer_lib.train(
        env_fn, hparams, event_dir, self.agent_model_dir,
        self._num_completed_iterations, name_scope=name_scope,
        frame_stack_size=self.frame_stack_size, force_beginning_resets=simulated
    )

  def evaluate(self, env_fn, hparams, stochastic):
    if stochastic:
      policy_to_actions_lambda = lambda policy: policy.sample()
    else:
      policy_to_actions_lambda = lambda policy: policy.mode()

    rl_trainer_lib.evaluate(
        env_fn, hparams, self.agent_model_dir,
        frame_stack_size=self.frame_stack_size, force_beginning_resets=False,
        policy_to_actions_lambda=policy_to_actions_lambda
    )
