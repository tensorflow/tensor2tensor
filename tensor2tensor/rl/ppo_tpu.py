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

"""PPO algorithm implementation.

Based on: https://arxiv.org/abs/1707.06347
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.rl import tf_new_collect
from tensor2tensor.utils import learning_rate
from tensor2tensor.utils import optimize

import tensorflow as tf
import tensorflow_probability as tfp


def define_ppo_step(data_points, hparams, action_space, observation_spec):
  """Define ppo step."""
  # Most PPO code has been removed to get a minimal reproducible example.
  # What's left is restoring the observation to the original shape, feeding
  # it to the policy network and computing a "loss" dependent on the result
  # to ensure that it gets computed in the graph.

  observation = data_points

  obs_shape = observation_spec.shape
  # Join the two batch dimensions, so
  # observation_shape = (BS, HISTORY, H, W, C).
  # This operation seems to be the problem.
  observation = tf_new_collect.restore_tensors(
      observation, observation_spec._replace(
          shape=((obs_shape[0] * obs_shape[1],) + obs_shape[2:]),
      )
  )

  # Run the network on the observation. This takes about 9 minutes, which is
  # really slow, especially because we need to run this several times in 1 PPO
  # epoch. For comparison, running just the collect takes about 2 min on TPU V3,
  # TF 1.13.
  (logits, new_value) = get_policy(observation, hparams, action_space)
  x = tf.math.reduce_mean(new_value)
  losses = [x] * 3

  # This variant is fast (takes about as much time as just the collect),
  # possibly because the reshape above gets optimized out because it doesn't
  # affect the result (we take a mean over all dimensions).
  #x = tf.to_float(tf.math.reduce_mean(observation))
  #losses = [obs_mean] * 3

  # This takes about 7 minutes - still very slow, which indicates that the
  # network itself is not the cause - BTW it's interesting, because we're again
  # taking a mean over all dimensions, just in 2 steps and with a cast in
  # between, it seems that the optimizer doesn't handle this.
  #x = tf.to_float(tf.math.reduce_mean(observation, axis=(1, 2, 3, 4)))
  #losses = [tf.math.reduce_mean(new_value)] * 3

  return losses


def define_ppo_epoch(memory, hparams, action_space, observation_spec):
  """PPO epoch."""
  # Most PPO code has been removed to get a minimal reproducible example.

  # This is to avoid propagating gradients through simulated environment.
  observation = tf.stop_gradient(memory.observation)
  # Observations are batched on 2 levels: collect batches (parallel rollouts
  # from the world model) and PPO optimization batches, so
  # observation_shape = (PPO_BS, COLLECT_BS, HISTORY, H, W, C)
  observation_spec = observation_spec._replace(
      shape=((hparams.optimization_batch_size,) + observation_spec.shape)
  )
  # Normally we would shuffle the memory and go over it in batches, but even
  # this small example hangs up (just taking the first batch once, without
  # shuffling).
  ppo_step_rets = define_ppo_step(observation[:hparams.optimization_batch_size, :],
                             hparams, action_space, observation_spec
                            )
  ppo_summaries = [tf.reduce_mean(ret) for ret in ppo_step_rets]
  return ppo_summaries
