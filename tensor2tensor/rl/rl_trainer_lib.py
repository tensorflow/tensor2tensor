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

import copy
import os

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor.models.research import rl  # pylint: disable=unused-import
from tensor2tensor.rl import collect
from tensor2tensor.rl import ppo
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


def define_train(hparams):
  """Define the training setup."""
  train_hparams = copy.copy(hparams)
  train_hparams.add_hparam("eval_phase", False)
  train_hparams.add_hparam(
      "policy_to_actions_lambda", lambda policy: policy.sample()
  )

  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
    memory, collect_summary, train_initialization = (
        collect.define_collect(train_hparams, "ppo_train")
    )
    ppo_summary = ppo.define_ppo_epoch(memory, hparams)
    train_summary = tf.summary.merge([collect_summary, ppo_summary])

    if hparams.eval_every_epochs:
      eval_hparams = copy.copy(hparams)
      eval_hparams.add_hparam("eval_phase", True)
      eval_hparams.add_hparam(
          "policy_to_actions_lambda", lambda policy: policy.mode()
      )
      eval_hparams.environment_spec = hparams.environment_eval_spec
      eval_hparams.num_agents = hparams.num_eval_agents

      _, eval_collect_summary, eval_initialization = (
          collect.define_collect(eval_hparams, "ppo_eval")
      )
      return train_summary, eval_collect_summary, (train_initialization,
                                                   eval_initialization)
    else:
      return train_summary, None, (train_initialization,)


def train(hparams, event_dir=None, model_dir=None,
          restore_agent=True, name_scope="rl_train", report_fn=None):
  """Train."""
  with tf.Graph().as_default():
    with tf.name_scope(name_scope):
      train_summary_op, eval_summary_op, intializers = define_train(hparams)
      if event_dir:
        summary_writer = tf.summary.FileWriter(
            event_dir, graph=tf.get_default_graph(), flush_secs=60)
      else:
        summary_writer = None

      if model_dir:
        model_saver = tf.train.Saver(
            tf.global_variables(".*network_parameters.*"))
      else:
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
          tf.logging.info("Skipping PPO training. Requested %d steps while "
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
            if report_fn:
              summary_proto = tf.Summary()
              summary_proto.ParseFromString(eval_summary)
              for elem in summary_proto.value:
                if "mean_score" in elem.tag:
                  report_fn(elem.simple_value, epoch_index)
                  break

          epoch_index_and_start = epoch_index + start_step
          if (model_saver and hparams.save_models_every_epochs and
              (epoch_index_and_start % hparams.save_models_every_epochs == 0 or
               (epoch_index + 1) == steps_to_go)):
            ckpt_path = os.path.join(
                model_dir, "model.ckpt-{}".format(epoch_index + 1 + start_step))
            model_saver.save(sess, ckpt_path)


def evaluate(hparams, model_dir, name_scope="rl_eval"):
  """Evaluate."""
  hparams = copy.copy(hparams)
  hparams.add_hparam("eval_phase", True)
  with tf.Graph().as_default():
    with tf.name_scope(name_scope):
      (collect_memory, _, collect_init) = collect.define_collect(
          hparams, "ppo_eval"
      )
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*")
      )

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        collect_init(sess)
        trainer_lib.restore_checkpoint(model_dir, model_saver, sess)
        sess.run(collect_memory)
