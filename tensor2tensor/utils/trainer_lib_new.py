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

"""Library for training. See t2t_trainer_new.py.

Overrides some of the trainer_lib declarations and reexports the rest.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils.trainer_lib import (  # pylint: disable=unused-import
    next_checkpoint, create_session_config, is_cloud_async_distributed,
    create_run_config, create_estimator, create_hooks, T2TExperiment,
    set_random_seed, restore_checkpoint
)

import tensorflow as tf


def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem=None):
  """Create HParams with data_dir and problem hparams, if kwargs provided."""
  hparams = registry.hparams(hparams_set)
  if data_dir:
    hparams.add_hparam("data_dir", data_dir)
  if problem:
    add_problem_hparams(hparams, problem)
  if hparams_overrides_str:
    tf.logging.info("Overriding hparams in %s with %s", hparams_set,
                    hparams_overrides_str)
    hparams = hparams.parse(hparams_overrides_str)
  return hparams


def create_experiment(
    run_config,
    hparams,
    model_name,
    problem,
    data_dir,
    train_steps,
    eval_steps,
    min_eval_frequency=2000,
    eval_throttle_seconds=600,
    schedule="train_and_evaluate",
    export=False,
    decode_hparams=None,
    use_tfdbg=False,
    use_dbgprofile=False,
    eval_early_stopping_steps=None,
    eval_early_stopping_metric=None,
    eval_early_stopping_metric_delta=None,
    eval_early_stopping_metric_minimize=True,
    use_tpu=False,
    use_tpu_estimator=False,
    use_xla=False,
    additional_train_hooks=None,
    additional_eval_hooks=None,
    warm_start_from=None,
    decode_from_file=None,
    decode_to_file=None,
    decode_reference=None,
    std_server_protocol=None):
  """Create Experiment."""
  # HParams
  hparams.add_hparam("model_dir", run_config.model_dir)
  hparams.add_hparam("data_dir", data_dir)
  hparams.add_hparam("train_steps", train_steps)
  hparams.add_hparam("eval_steps", eval_steps)
  hparams.add_hparam("schedule", schedule)
  hparams.add_hparam("warm_start_from", warm_start_from)
  hparams.add_hparam("std_server_protocol", std_server_protocol)
  if decode_hparams is not None:
    decode_hparams.add_hparam("decode_from_file", decode_from_file)
    decode_hparams.add_hparam("decode_to_file", decode_to_file)
    decode_hparams.add_hparam("decode_reference", decode_reference)
  add_problem_hparams(hparams, problem)

  # Estimator
  estimator = create_estimator(
      model_name,
      hparams,
      run_config,
      schedule=schedule,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu,
      use_tpu_estimator=use_tpu_estimator,
      use_xla=use_xla)

  # Input fns from Problem
  train_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.TRAIN,
                                                   hparams)
  eval_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.EVAL,
                                                  hparams)

  # Export
  exporter = None
  if export:
    def compare_fn(best_eval_result, current_eval_result):
      metric = eval_early_stopping_metric or "loss"
      return current_eval_result[metric] < best_eval_result[metric]

    exporter = tf.estimator.BestExporter(
        name="best",
        serving_input_receiver_fn=lambda: problem.serving_input_fn(hparams),
        compare_fn=compare_fn,
        assets_extra=problem.export_assets)

  # Hooks
  validation_monitor_kwargs = dict(
      input_fn=eval_input_fn,
      eval_steps=eval_steps,
      every_n_steps=min_eval_frequency,
      early_stopping_rounds=eval_early_stopping_steps,
      early_stopping_metric=eval_early_stopping_metric,
      early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
  dbgprofile_kwargs = {"output_dir": run_config.model_dir}
  early_stopping_kwargs = dict(
      events_dir=os.path.join(run_config.model_dir, "eval_continuous"),
      tag=eval_early_stopping_metric,
      num_plateau_steps=eval_early_stopping_steps,
      plateau_decrease=eval_early_stopping_metric_minimize,
      plateau_delta=eval_early_stopping_metric_delta,
      every_n_steps=min_eval_frequency)

  # Eval on TPU Pods is not supported yet
  if use_tpu and run_config.tpu_config.num_shards > 8 and "eval" in schedule:
    raise ValueError("Eval is not currently supported on a TPU Pod")

  # In-process eval (and possible early stopping)
  if schedule == "continuous_train_and_eval" and min_eval_frequency:
    tf.logging.warn("ValidationMonitor only works with "
                    "--schedule=train_and_evaluate")
  use_validation_monitor = (
      schedule == "train_and_evaluate" and min_eval_frequency)
  # Distributed early stopping
  local_schedules = ["train_and_evaluate", "continuous_train_and_eval"]
  use_early_stopping = (
      schedule not in local_schedules and eval_early_stopping_steps)
  train_hooks, eval_hooks = create_hooks(
      use_tfdbg=use_tfdbg,
      use_dbgprofile=use_dbgprofile,
      dbgprofile_kwargs=dbgprofile_kwargs,
      use_validation_monitor=use_validation_monitor,
      validation_monitor_kwargs=validation_monitor_kwargs,
      use_early_stopping=use_early_stopping,
      early_stopping_kwargs=early_stopping_kwargs)
  train_hooks += t2t_model.T2TModel.get_train_hooks(model_name)
  eval_hooks += t2t_model.T2TModel.get_eval_hooks(model_name)
  if additional_train_hooks:
    train_hooks += additional_train_hooks
  if additional_eval_hooks:
    eval_hooks += additional_eval_hooks

  train_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      train_hooks, estimator)
  eval_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      eval_hooks, estimator)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=train_steps, hooks=train_hooks)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=eval_steps,
      hooks=eval_hooks,
      start_delay_secs=0 if hparams.schedule == "evaluate" else 120,
      throttle_secs=eval_throttle_seconds,
      exporters=exporter)

  return T2TExperiment(estimator, hparams, train_spec, eval_spec,
                       use_validation_monitor, decode_hparams)


def create_experiment_fn(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn


def add_problem_hparams(hparams, problem):
  """Add problem hparams for the problems."""
  p_hparams = problem.get_hparams(hparams)

  hparams.problem = problem
  hparams.problem_hparams = p_hparams
