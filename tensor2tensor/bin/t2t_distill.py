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
r"""Perform distillation for a teacher to student.

This script is intended to be used with --model=distillation. See the model for
example hyperparameters and usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import cloud_mlengine
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()

  if FLAGS.cloud_mlengine:
    cloud_mlengine.launch()
    return

  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])

  with t2t_trainer.maybe_cloud_tpu():
    root_output_dir = FLAGS.output_dir

    # Train Teacher ============
    hparams = t2t_trainer.create_hparams()
    hparams.distill_phase = "train"
    teacher_dir = os.path.join(root_output_dir, "teacher")
    FLAGS.output_dir = teacher_dir

    exp_fn = t2t_trainer.create_experiment_fn()
    run_config = t2t_trainer.create_run_config(hparams)
    exp = exp_fn(run_config, hparams)
    if t2t_trainer.is_chief():
      t2t_trainer.save_metadata(hparams)
    t2t_trainer.execute_schedule(exp)
    # ==========================
    # Train Student ============
    hparams = t2t_trainer.create_hparams()
    hparams.add_hparam("teacher_dir", teacher_dir)
    hparams.distill_phase = "distill"
    student_dir = os.path.join(root_output_dir, "student")
    FLAGS.output_dir = student_dir

    exp_fn = t2t_trainer.create_experiment_fn()
    run_config = t2t_trainer.create_run_config(hparams)
    exp = exp_fn(run_config, hparams)

    if t2t_trainer.is_chief():
      t2t_trainer.save_metadata(hparams)
    t2t_trainer.execute_schedule(exp)
    # ==========================


def create_teacher_experiment(run_config, hparams, argv):
  """Creates experiment function."""
  tf.logging.info("training teacher")
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()

  if FLAGS.cloud_mlengine:
    return cloud_mlengine.launch()

  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])

  with t2t_trainer.maybe_cloud_tpu():
    hparams.distill_phase = "train"
    exp_fn = t2t_trainer.create_experiment_fn()
    exp = exp_fn(run_config, hparams)
    return exp


def create_student_experiment(run_config, hparams, argv):
  """Creates experiment function."""
  tf.logging.info("training student")
  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  t2t_trainer.maybe_log_registry_and_exit()

  if FLAGS.cloud_mlengine:
    return cloud_mlengine.launch()

  if FLAGS.generate_data:
    t2t_trainer.generate_data()

  if cloud_mlengine.job_dir():
    FLAGS.output_dir = cloud_mlengine.job_dir()

  if argv:
    t2t_trainer.set_hparams_from_args(argv[1:])

  with t2t_trainer.maybe_cloud_tpu():
    hparams.add_hparam("teacher_dir", FLAGS.teacher_dir)
    hparams.distill_phase = "distill"
    exp_fn = t2t_trainer.create_experiment_fn()
    exp = exp_fn(run_config, hparams)
    return exp


def create_experiment_fn(argv, train_teacher):

  def teacher_experiment_fn(run_config, hparams):
    return create_teacher_experiment(run_config, hparams, argv)

  def student_experiment_fn(run_config, hparams):
    return create_student_experiment(run_config, hparams, argv)

  return teacher_experiment_fn if train_teacher else student_experiment_fn


if __name__ == "__main__":
  tf.app.run()
