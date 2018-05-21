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
"""Library for training of RL agent with PPO algorithm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports

import gym

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor.models.research import rl  # pylint: disable=unused-import
from tensor2tensor.rl import collect
from tensor2tensor.rl import ppo
from tensor2tensor.rl.envs import tf_atari_wrappers
from tensor2tensor.rl.envs import utils
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


def define_train(hparams, environment_spec, event_dir):
  """Define the training setup."""
  policy_lambda = hparams.network

  if environment_spec == "stacked_pong":
    environment_spec = lambda: gym.make("PongNoFrameskip-v4")
    wrappers = hparams.in_graph_wrappers if hasattr(
        hparams, "in_graph_wrappers") else []
    wrappers.append((tf_atari_wrappers.MaxAndSkipWrapper, {"skip": 4}))
    hparams.in_graph_wrappers = wrappers
  if isinstance(environment_spec, str):
    env_lambda = lambda: gym.make(environment_spec)
  else:
    env_lambda = environment_spec

  batch_env = utils.batch_env_factory(
      env_lambda, hparams, num_agents=hparams.num_agents)

  policy_factory = functools.partial(
      policy_lambda, batch_env.action_space, hparams)

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    memory, collect_summary = collect.define_collect(
        policy_factory, batch_env, hparams, eval_phase=False,
        on_simulated=hparams.simulated_environment)
    ppo_summary = ppo.define_ppo_epoch(memory, policy_factory, hparams)
    summary = tf.summary.merge([collect_summary, ppo_summary])

  with tf.variable_scope("eval", reuse=tf.AUTO_REUSE):
    eval_env_lambda = env_lambda
    if event_dir and hparams.video_during_eval:
      # Some environments reset environments automatically, when reached done
      # state. For them we shall record only every second episode.
      d = 2 if env_lambda().metadata.get("semantics.autoreset") else 1
      eval_env_lambda = lambda: gym.wrappers.Monitor(  # pylint: disable=g-long-lambda
          env_lambda(), event_dir, video_callable=lambda i: i % d == 0)
      eval_env_lambda = (
          lambda: utils.EvalVideoWrapper(eval_env_lambda()))
    eval_batch_env = utils.batch_env_factory(
        eval_env_lambda, hparams,
        num_agents=hparams.num_eval_agents, xvfb=hparams.video_during_eval)

    _, eval_summary = collect.define_collect(
        policy_factory, eval_batch_env, hparams, eval_phase=True)
  return summary, eval_summary


def train(hparams, environment_spec, event_dir=None, model_dir=None,
          restore_agent=True, epoch=0):
  """Train."""
  with tf.name_scope("rl_train"):
    train_summary_op, eval_summary_op = define_train(hparams, environment_spec,
                                                     event_dir)
    if event_dir:
      summary_writer = tf.summary.FileWriter(
          event_dir, graph=tf.get_default_graph(), flush_secs=60)
    if model_dir:
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*"))
    else:
      summary_writer = None
      model_saver = None

    if hparams.simulated_environment:
      env_model_loader = tf.train.Saver(tf.global_variables("basic_conv_gen.*"))
    else:
      env_model_loader = None

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      if env_model_loader:
        trainer_lib.restore_checkpoint(
            hparams.world_model_dir, env_model_loader, sess, must_restore=True)
      start_step = 0
      if model_saver and restore_agent:
        start_step = trainer_lib.restore_checkpoint(
            model_dir, model_saver, sess)

      # Fail-friendly, don't train if already trained for this epoch
      if start_step >= ((hparams.epochs_num * (epoch+1)) - 5):
        tf.logging.info("Skipping PPO training for epoch %d as train steps "
                        "(%d) already reached", epoch, start_step)
        return

      for epoch_index in range(hparams.epochs_num):
        summary = sess.run(train_summary_op)
        if summary_writer:
          summary_writer.add_summary(summary, epoch_index)
        if (hparams.eval_every_epochs and
            epoch_index % hparams.eval_every_epochs == 0):
          summary = sess.run(eval_summary_op)
          if summary_writer and summary:
            summary_writer.add_summary(summary, epoch_index)
          else:
            tf.logging.info("Eval summary not saved")
        if (model_saver and hparams.save_models_every_epochs and
            (epoch_index % hparams.save_models_every_epochs == 0 or
             (epoch_index + 1) == hparams.epochs_num)):
          ckpt_path = os.path.join(
              model_dir, "model.ckpt-{}".format(epoch_index + 1 + start_step))
          model_saver.save(sess, ckpt_path)
