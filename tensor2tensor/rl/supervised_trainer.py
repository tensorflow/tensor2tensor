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

"""Supervised trainer for use in the RL pipeline.

Based on bin.t2t_trainer. Exports a function train() that allows passing
parameters via arguments instead of flags. Takes problem and hparams as
instances instead of names in registry.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


def create_experiment(
    run_config,
    hparams,
    model_name,
    problem,
    data_dir,
    train_steps,
    eval_steps,
    min_eval_frequency=2000):
  """Create Experiment.

  Based on t2t_trainer_lib.create_experiment. Gets problem instance instead of
  name as an argument.
  """
  # HParams
  hparams.add_hparam("model_dir", run_config.model_dir)
  hparams.add_hparam("data_dir", data_dir)
  hparams.add_hparam("train_steps", train_steps)
  hparams.add_hparam("eval_steps", eval_steps)
  hparams.add_hparam("schedule", "train_and_evaluate")
  hparams.add_hparam("warm_start_from", None)
  hparams.add_hparam("std_server_protocol", None)

  problem_hparams = problem.get_hparams(hparams)
  hparams.problem = problem
  hparams.problem_hparams = problem_hparams

  # Estimator
  estimator = trainer_lib.create_estimator(
      model_name,
      hparams,
      run_config
  )

  # Input fns from Problem
  (train_input_fn, eval_input_fn) = (
      problem.make_estimator_input_fn(mode_key, hparams)
      for mode_key in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL)
  )

  # Hooks
  train_hooks = []
  eval_hooks = []

  # In-process eval
  use_validation_monitor = bool(min_eval_frequency)
  if use_validation_monitor:
    tf.logging.info("Using ValidationMonitor")
    train_hooks.append(
        tf.contrib.learn.monitors.ValidationMonitor(
            hooks=eval_hooks, input_fn=eval_input_fn, eval_steps=eval_steps,
            every_n_steps=min_eval_frequency
        )
    )

  train_hooks += t2t_model.T2TModel.get_train_hooks(model_name)
  eval_hooks += t2t_model.T2TModel.get_eval_hooks(model_name)

  train_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      train_hooks, estimator
  )
  eval_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      eval_hooks, estimator
  )

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=train_steps, hooks=train_hooks
  )
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=eval_steps,
      hooks=eval_hooks,
      start_delay_secs=120
  )

  return trainer_lib.T2TExperiment(
      estimator, hparams, train_spec, eval_spec, use_validation_monitor
  )


def create_experiment_fn(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn


def create_run_config(output_dir, local_eval_frequency):
  """Create a run config.

  Args:
    output_dir: model's output directory
    local_eval_frequency

  Returns:
    a run config
  """
  save_ckpt_steps = max(
      FLAGS.iterations_per_loop, local_eval_frequency
  )
  save_ckpt_secs = FLAGS.save_checkpoints_secs or None
  if save_ckpt_secs:
    save_ckpt_steps = None
  assert output_dir

  return trainer_lib.create_run_config(
      model_dir=output_dir,
      random_seed=FLAGS.random_seed,
      save_checkpoints_steps=save_ckpt_steps,
      save_checkpoints_secs=save_ckpt_secs
  )


def train(problem, model_name, hparams, data_dir, output_dir, train_steps,  # pylint: disable=redefined-outer-name
          eval_steps, local_eval_frequency=None):
  if local_eval_frequency is None:
    local_eval_frequency = getattr(FLAGS, "local_eval_frequency")

  tf.logging.set_verbosity(tf.logging.INFO)
  trainer_lib.set_random_seed(FLAGS.random_seed)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  exp_fn = create_experiment_fn(
      problem=problem, model_name=model_name, data_dir=data_dir,
      train_steps=train_steps, eval_steps=eval_steps,
      min_eval_frequency=local_eval_frequency
  )
  exp = exp_fn(
      create_run_config(output_dir, local_eval_frequency), hparams
  )
  t2t_trainer.execute_schedule(exp)
