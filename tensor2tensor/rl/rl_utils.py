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

r"""Utilities for RL training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.models.research import rl
from tensor2tensor.rl.dopamine_connector import DQNLearner
from tensor2tensor.rl.ppo_learner import PPOLearner
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


def compute_mean_reward(rollouts, clipped):
  """Calculate mean rewards from given epoch."""
  reward_name = "reward" if clipped else "unclipped_reward"
  rewards = []
  for rollout in rollouts:
    if rollout[-1].done:
      rollout_reward = sum(getattr(frame, reward_name) for frame in rollout)
      rewards.append(rollout_reward)
  if rewards:
    mean_rewards = np.mean(rewards)
  else:
    mean_rewards = 0
  return mean_rewards


def get_metric_name(stochastic, max_num_noops, clipped):
  return "mean_reward/eval/stochastic_{}_max_noops_{}_{}".format(
      stochastic, max_num_noops, "clipped" if clipped else "unclipped")


def evaluate_single_config(hparams, stochastic, max_num_noops,
                           agent_model_dir):
  """Evaluate the PPO agent in the real environment."""
  eval_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  env = setup_env(
      hparams, batch_size=hparams.eval_batch_size, max_num_noops=max_num_noops
  )
  env.start_new_epoch(0)
  env_fn = rl.make_real_env_fn(env)
  learner = LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, base_event_dir=None,
      agent_model_dir=agent_model_dir
  )
  learner.evaluate(env_fn, eval_hparams, stochastic)
  rollouts = env.current_epoch_rollouts()
  env.close()

  return tuple(
      compute_mean_reward(rollouts, clipped) for clipped in (True, False)
  )


def evaluate_all_configs(hparams, agent_model_dir):
  """Evaluate the agent with multiple eval configurations."""
  metrics = {}
  # Iterate over all combinations of picking actions by sampling/mode and
  # whether to do initial no-ops.
  for stochastic in (True, False):
    for max_num_noops in (hparams.eval_max_num_noops, 0):
      scores = evaluate_single_config(
          hparams, stochastic, max_num_noops, agent_model_dir
      )
      for (score, clipped) in zip(scores, (True, False)):
        metric_name = get_metric_name(stochastic, max_num_noops, clipped)
        metrics[metric_name] = score

  return metrics


LEARNERS = {
    "ppo": PPOLearner,
    "dqn": DQNLearner,
}


def setup_env(hparams, batch_size, max_num_noops):
  """Setup."""
  game_mode = "Deterministic-v4"
  camel_game_name = "".join(
      [w[0].upper() + w[1:] for w in hparams.game.split("_")])
  camel_game_name += game_mode
  env_name = camel_game_name

  env = T2TGymEnv(base_env_name=env_name,
                  batch_size=batch_size,
                  grayscale=hparams.grayscale,
                  resize_width_factor=hparams.resize_width_factor,
                  resize_height_factor=hparams.resize_height_factor,
                  base_env_timesteps_limit=hparams.env_timesteps_limit,
                  max_num_noops=max_num_noops)
  return env

def update_hparams_from_hparams(target_hparams, source_hparams, prefix):
  """Copy a subset of hparams to target_hparams."""
  for (param_name, param_value) in six.iteritems(source_hparams.values()):
    if param_name.startswith(prefix):
      target_hparams.set_hparam(param_name[len(prefix):], param_value)
