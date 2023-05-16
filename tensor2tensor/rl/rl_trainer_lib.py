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


def define_train(hparams):
  """Define the training setup."""
  with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
    memory, collect_summary, train_initialization\
      = collect.define_collect(
          hparams, "ppo_train", eval_phase=False)
    ppo_summary = ppo.define_ppo_epoch(memory, hparams)
    train_summary = tf.compat.v1.summary.merge([collect_summary, ppo_summary])

    if hparams.eval_every_epochs:
      _, eval_collect_summary, eval_initialization\
        = collect.define_collect(
            hparams, "ppo_eval", eval_phase=True)
      return train_summary, eval_collect_summary, \
             (train_initialization, eval_initialization)
    else:
      return train_summary, None, (train_initialization,)


def train(hparams, event_dir=None, model_dir=None,
          restore_agent=True, name_scope="rl_train"):
  """Train."""
  with tf.Graph().as_default():
    with tf.compat.v1.name_scope(name_scope):
      train_summary_op, eval_summary_op, intializers = define_train(hparams)
      if event_dir:
        summary_writer = tf.compat.v1.summary.FileWriter(
            event_dir, graph=tf.compat.v1.get_default_graph(), flush_secs=60)
      else:
        summary_writer = None

      if model_dir:
        model_saver = tf.compat.v1.train.Saver(
            tf.compat.v1.global_variables(".*network_parameters.*"))
      else:
        model_saver = None

      # TODO(piotrmilos): This should be refactored, possibly with
      # handlers for each type of env
      if hparams.environment_spec.simulated_env:
        env_model_loader = tf.compat.v1.train.Saver(
            tf.compat.v1.global_variables("next_frame*"))
      else:
        env_model_loader = None

      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for initializer in intializers:
          initializer(sess)
        if env_model_loader:
          trainer_lib.restore_checkpoint(
              hparams.world_model_dir, env_model_loader, sess,
              must_restore=True)
        start_step = 0
        if model_saver and restore_agent:
          start_step = trainer_lib.restore_checkpoint(
              model_dir, model_saver, sess)

        # Fail-friendly, complete only unfinished epoch
        steps_to_go = hparams.epochs_num - start_step

        if steps_to_go <= 0:
          tf.compat.v1.logging.info("Skipping PPO training. Requested %d steps while "
                          "%d train steps already reached",
                          hparams.epochs_num, start_step)
          return

        for epoch_index in range(steps_to_go):
          summary = sess.run(train_summary_op)
          if summary_writer:
            summary_writer.add_summary(summary, epoch_index)

          if (hparams.eval_every_epochs and
              epoch_index % hparams.eval_every_epochs == 0):
            eval_summary = sess.run(eval_summary_op)
            if summary_writer:
              summary_writer.add_summary(eval_summary, epoch_index)

          epoch_index_and_start = epoch_index + start_step
          if (model_saver and hparams.save_models_every_epochs and
              (epoch_index_and_start % hparams.save_models_every_epochs == 0 or
               (epoch_index + 1) == steps_to_go)):
            ckpt_path = os.path.join(
                model_dir, "model.ckpt-{}".format(epoch_index + 1 + start_step))
            model_saver.save(sess, ckpt_path)
