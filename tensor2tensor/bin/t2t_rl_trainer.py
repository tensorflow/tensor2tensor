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

"""Training of RL agent with PPO algorithm."""

from __future__ import absolute_import

import functools
from munch import Munch
import tensorflow as tf

from tensor2tensor.rl.collect import define_collect
from tensor2tensor.rl.envs.utils import define_batch_env
from tensor2tensor.rl.ppo import define_ppo_epoch

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("event_dir", "/tmp",
                    "Where to store the event file.")

def define_train(policy_lambda, env_lambda, config):
  env = env_lambda()
  action_space = env.action_space
  observation_space = env.observation_space

  batch_env = define_batch_env(env_lambda, config["num_agents"])

  policy_factory = tf.make_template(
      'network',
      functools.partial(policy_lambda, observation_space,
                        action_space, config))

  memory, collect_summary = define_collect(policy_factory, batch_env, config)
  ppo_summary = define_ppo_epoch(memory, policy_factory, config)
  summary = tf.summary.merge([collect_summary, ppo_summary])

  return summary


def main():
  train(example_params())


def train(params):
  policy_lambda, env_lambda, config = params
  summary_op = define_train(policy_lambda, env_lambda, config)

  summary_writer = tf.summary.FileWriter(
      FLAGS.event_dir, graph=tf.get_default_graph(), flush_secs=60)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_index in range(config.epochs_num):
      summary = sess.run(summary_op)
      summary_writer.add_summary(summary, epoch_index)


def example_params():
  from tensor2tensor.rl import networks
  config = {}
  config['init_mean_factor'] = 0.1
  config['init_logstd'] = 0.1
  config['policy_layers'] = 100, 100
  config['value_layers'] = 100, 100
  config['num_agents'] = 30
  config['clipping_coef'] = 0.2
  config['gae_gamma'] = 0.99
  config['gae_lambda'] = 0.95
  config['entropy_loss_coef'] = 0.01
  config['value_loss_coef'] = 1
  config['optimizer'] = tf.train.AdamOptimizer
  config['learning_rate'] = 1e-4
  config['optimization_epochs'] = 15
  config['epoch_length'] = 200
  config['epochs_num'] = 2000

  config = Munch(config)
  return networks.feed_forward_gaussian_fun, pendulum_lambda, config


def pendulum_lambda():
  import gym
  return gym.make("Pendulum-v0")


if __name__ == '__main__':
  main()
