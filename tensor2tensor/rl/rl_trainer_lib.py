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


def define_train(train_env, ppo_hparams, eval_env_fn=None, **collect_kwargs):
  """Define the training setup."""
  memory, collect_summary, train_initialization = (
      collect.define_collect(
          train_env, ppo_hparams, "ppo_train", eval_phase=False,
          policy_to_actions_lambda=(lambda policy: policy.sample()),
          **collect_kwargs
      )
  )
  ppo_summary = ppo.define_ppo_epoch(
      memory, ppo_hparams, train_env.action_space, train_env.batch_size
  )
  train_summary = tf.summary.merge([collect_summary, ppo_summary])

  if ppo_hparams.eval_every_epochs:
    assert eval_env_fn is not None
    eval_env = eval_env_fn(in_graph=True)
    _, eval_collect_summary, eval_initialization = (
        collect.define_collect(
            eval_env, ppo_hparams, "ppo_eval", eval_phase=True,
            policy_to_actions_lambda=(lambda policy: policy.mode()),
            **collect_kwargs
        )
    )
    return train_summary, eval_collect_summary, (train_initialization,
                                                 eval_initialization)
  else:
    return train_summary, None, (train_initialization,)


def train(
    ppo_hparams, event_dir, model_dir, num_target_iterations,
    train_summary_op, eval_summary_op, initializers, report_fn=None
):
  """Train."""
  summary_writer = tf.summary.FileWriter(
      event_dir, graph=tf.get_default_graph(), flush_secs=60)

  model_saver = tf.train.Saver(
      tf.global_variables(".*network_parameters.*"))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for initializer in initializers:
      initializer(sess)
    num_completed_iterations = trainer_lib.restore_checkpoint(
        model_dir, model_saver, sess)

    # Fail-friendly, complete only unfinished epoch
    num_iterations_to_go = num_target_iterations - num_completed_iterations

    if num_iterations_to_go <= 0:
      tf.logging.info(
          "Skipping PPO training. Requested %d iterations while %d train "
          "iterations already reached", num_target_iterations,
          num_completed_iterations
      )
      return

    for epoch_index in range(num_iterations_to_go):
      summary = sess.run(train_summary_op)
      if summary_writer:
        summary_writer.add_summary(summary, epoch_index)

      if (ppo_hparams.eval_every_epochs and
          epoch_index % ppo_hparams.eval_every_epochs == 0):
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

      epoch_index_and_start = epoch_index + num_completed_iterations
      if (model_saver and ppo_hparams.save_models_every_epochs and
          (epoch_index_and_start %
           ppo_hparams.save_models_every_epochs == 0 or
           (epoch_index + 1) == num_iterations_to_go)):
        ckpt_path = os.path.join(
            model_dir, "model.ckpt-{}".format(
                epoch_index + 1 + num_completed_iterations
            )
        )
        model_saver.save(sess, ckpt_path)


def evaluate(env_fn, ppo_hparams, model_dir, **collect_kwargs):
  """Evaluate."""
  with tf.Graph().as_default():
    with tf.name_scope("rl_eval"):
      eval_env = env_fn(in_graph=True)
      (collect_memory, _, collect_init) = collect.define_collect(
          eval_env, ppo_hparams, "ppo_eval", eval_phase=True, **collect_kwargs
      )
      model_saver = tf.train.Saver(
          tf.global_variables(".*network_parameters.*")
      )

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        collect_init(sess)
        trainer_lib.restore_checkpoint(model_dir, model_saver, sess)
        sess.run(collect_memory)
