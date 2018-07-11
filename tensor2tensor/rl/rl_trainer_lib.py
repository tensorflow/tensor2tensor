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

import os

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor.models.research import rl  # pylint: disable=unused-import
from tensor2tensor.rl import collect
from tensor2tensor.rl import ppo
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


def define_train(hparams, event_dir):
  """Define the training setup."""
  del event_dir
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    memory, collect_summary, initialization\
      = collect.define_collect(
          hparams, "ppo_train", eval_phase=False)
    ppo_summary = ppo.define_ppo_epoch(memory, hparams)
    summary = tf.summary.merge([collect_summary, ppo_summary])

  return summary, None, initialization


def train(hparams, event_dir=None, model_dir=None,
          restore_agent=True, epoch=0):
  """Train."""
  with tf.name_scope("rl_train"):
    train_summary_op, _, initialization = define_train(hparams, event_dir)
    if event_dir:
      summary_writer = tf.summary.FileWriter(
          event_dir, graph=tf.get_default_graph(), flush_secs=60)
    if model_dir:
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*"))
    else:
      summary_writer = None
      model_saver = None

    # TODO(piotrmilos): This should be refactored, possibly with
    # handlers for each type of env
    if hparams.environment_spec.simulated_env:
      env_model_loader = tf.train.Saver(
          tf.global_variables("next_frame*"))
    else:
      env_model_loader = None

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      initialization(sess)
      if env_model_loader:
        trainer_lib.restore_checkpoint(
            hparams.world_model_dir, env_model_loader, sess, must_restore=True)
      start_step = 0
      if model_saver and restore_agent:
        start_step = trainer_lib.restore_checkpoint(
            model_dir, model_saver, sess)

      # Fail-friendly, don't train if already trained for this epoch
      if start_step >= ((hparams.epochs_num * (epoch + 1))):
        tf.logging.info("Skipping PPO training for epoch %d as train steps "
                        "(%d) already reached", epoch, start_step)
        return

      for epoch_index in range(hparams.epochs_num):
        summary = sess.run(train_summary_op)
        if summary_writer:
          summary_writer.add_summary(summary, epoch_index)
        if (hparams.eval_every_epochs and
            epoch_index % hparams.eval_every_epochs == 0):
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
