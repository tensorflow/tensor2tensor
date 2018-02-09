# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Library for training of RL agent with PPO algorithm."""

from __future__ import absolute_import

import functools

# Dependency imports

import gym
from tensor2tensor.rl import collect
from tensor2tensor.rl import ppo
from tensor2tensor.rl.envs import utils

import tensorflow as tf


def define_train(hparams):
  """Define the training setup."""
  env_lambda = lambda: gym.make(hparams.environment)
  policy_lambda = hparams.network
  env = env_lambda()
  action_space = env.action_space

  batch_env = utils.define_batch_env(env_lambda, hparams.num_agents)

  policy_factory = tf.make_template(
      "network",
      functools.partial(policy_lambda, action_space, hparams))

  memory, collect_summary = collect.define_collect(policy_factory,
                                                   batch_env, hparams)
  ppo_summary = ppo.define_ppo_epoch(memory, policy_factory, hparams)
  summary = tf.summary.merge([collect_summary, ppo_summary])

  return summary


def train(hparams, event_dir=None):
  summary_op = define_train(hparams)

  if event_dir:
      summary_writer = tf.summary.FileWriter(event_dir, graph=tf.get_default_graph(), flush_secs=60)
  else:
      summary_writer = None

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_index in range(hparams.epochs_num):
      summary = sess.run(summary_op)
      if summary_writer:
        summary_writer.add_summary(summary, epoch_index)
