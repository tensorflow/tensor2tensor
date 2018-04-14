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

"""Library for training. See t2t_trainer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

# Dependency imports

import numpy as np

from tensor2tensor.utils import devices
from tensor2tensor.utils import metrics_hook
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug


def create_session_config(log_device_placement=False,
                          enable_graph_rewriter=False,
                          gpu_mem_fraction=0.95,
                          use_tpu=False):
  """The TensorFlow Session config to use."""
  if use_tpu:
    graph_options = tf.GraphOptions()
  else:
    if enable_graph_rewriter:
      rewrite_options = rewriter_config_pb2.RewriterConfig()
      rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.ON
      graph_options = tf.GraphOptions(rewrite_options=rewrite_options)
    else:
      graph_options = tf.GraphOptions(
          optimizer_options=tf.OptimizerOptions(
              opt_level=tf.OptimizerOptions.L1, do_function_inlining=False))

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)

  config = tf.ConfigProto(
      allow_soft_placement=True,
      graph_options=graph_options,
      gpu_options=gpu_options,
      log_device_placement=log_device_placement)
  return config


def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem_name=None):
  """Create HParams with data_dir and problem hparams, if kwargs provided."""
  hparams = registry.hparams(hparams_set)()
  if data_dir:
    hparams.add_hparam("data_dir", data_dir)
  if problem_name:
    add_problem_hparams(hparams, problem_name)
  if hparams_overrides_str:
    tf.logging.info("Overriding hparams in %s with %s",
                    hparams_set,
                    hparams_overrides_str)
    hparams = hparams.parse(hparams_overrides_str)
  return hparams


def create_run_config(master="",
                      model_dir=None,
                      iterations_per_loop=1000,
                      num_shards=8,
                      log_device_placement=False,
                      save_checkpoints_steps=1000,
                      save_checkpoints_secs=None,
                      keep_checkpoint_max=20,
                      keep_checkpoint_every_n_hours=10000,
                      num_gpus=1,
                      gpu_order="",
                      shard_to_cpu=False,
                      num_async_replicas=1,
                      enable_graph_rewriter=False,
                      gpu_mem_fraction=0.95,
                      no_data_parallelism=False,
                      daisy_chain_variables=True,
                      schedule="continuous_train_and_eval",
                      worker_job="/job:localhost",
                      worker_id=0,
                      ps_replicas=0,
                      ps_job="/job:ps",
                      ps_gpu=0,
                      random_seed=None,
                      sync=False,
                      tpu_infeed_sleep_secs=None,
                      use_tpu=False):
  """Create RunConfig, TPUConfig, and Parallelism object."""
  session_config = create_session_config(
      log_device_placement=log_device_placement,
      enable_graph_rewriter=enable_graph_rewriter,
      gpu_mem_fraction=gpu_mem_fraction,
      use_tpu=use_tpu)
  run_config_args = {
      "master": master,
      "evaluation_master": master,
      "model_dir": model_dir,
      "session_config": session_config,
      "save_summary_steps": 100,
      "save_checkpoints_steps": save_checkpoints_steps,
      "keep_checkpoint_max": keep_checkpoint_max,
      "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
      "tf_random_seed": random_seed,
  }
  if save_checkpoints_secs:
    del run_config_args["save_checkpoints_steps"]
    run_config_args["save_checkpoints_secs"] = save_checkpoints_secs
  run_config_cls = tf.contrib.learn.RunConfig

  # If using TPU, use TPU RunConfig, add TPUConfig, and add additional args
  if use_tpu:
    run_config_cls = tf.contrib.tpu.RunConfig
    tpu_config = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=iterations_per_loop,
        num_shards=num_shards,
        per_host_input_for_training=True,
        initial_infeed_sleep_secs=tpu_infeed_sleep_secs)
    run_config_args["tpu_config"] = tpu_config

  config = run_config_cls(**run_config_args)

  # If not using TPU, add device info for data_parallelism
  config.use_tpu = use_tpu
  if not use_tpu:
    config.t2t_device_info = {
        "num_async_replicas": num_async_replicas,
    }
    config.data_parallelism = devices.data_parallelism(
        daisy_chain_variables=daisy_chain_variables,
        ps_replicas=ps_replicas,
        ps_job=ps_job,
        ps_gpu=ps_gpu,
        schedule=schedule,
        sync=sync,
        worker_gpu=num_gpus,
        worker_replicas=num_async_replicas,
        worker_id=worker_id,
        gpu_order=gpu_order,
        locally_shard_to_cpu=shard_to_cpu,
        worker_job=worker_job,
        no_data_parallelism=no_data_parallelism)

  return config


def create_estimator(model_name,
                     hparams,
                     run_config,
                     schedule="train_and_evaluate",
                     decode_hparams=None,
                     use_tpu=False):
  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      model_name, hparams, decode_hparams=decode_hparams, use_tpu=use_tpu)

  if use_tpu:
    problem = hparams.problem_instances[0]
    batch_size = (problem.tpu_batch_size_per_shard(hparams) *
                  run_config.tpu_config.num_shards)
    return tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size if "eval" in schedule else None)
  else:
    return tf.estimator.Estimator(
        model_fn=model_fn, model_dir=run_config.model_dir, config=run_config)


def create_hooks(use_tfdbg=False, use_dbgprofile=False, dbgprofile_kwargs=None,
                 use_validation_monitor=False, validation_monitor_kwargs=None,
                 use_early_stopping=False, early_stopping_kwargs=None):
  """Create train and eval hooks for Experiment."""
  train_monitors = []
  eval_hooks = []

  if use_tfdbg:
    hook = debug.LocalCLIDebugHook()
    train_monitors.append(hook)
    eval_hooks.append(hook)

  if use_dbgprofile:
    # Recorded traces can be visualized with chrome://tracing/
    # The memory/tensor lifetime is also profiled
    tf.logging.info("Using ProfilerHook")
    defaults = dict(save_steps=10, show_dataflow=True, show_memory=True)
    defaults.update(dbgprofile_kwargs)
    # To handle different versions of TF
    if hasattr(tf.train, "ProfilerHook"):
      hook_mod = tf.train
    else:
      hook_mod = tf.contrib.hooks
    train_monitors.append(hook_mod.ProfilerHook(**defaults))

  if use_validation_monitor:
    tf.logging.info("Using ValidationMonitor")
    train_monitors.append(
        tf.contrib.learn.monitors.ValidationMonitor(
            hooks=eval_hooks, **validation_monitor_kwargs))

  if use_early_stopping:
    tf.logging.info("Using EarlyStoppingHook")
    hook = metrics_hook.EarlyStoppingHook(**early_stopping_kwargs)
    # Adding to both training and eval so that eval aborts as well
    train_monitors.append(hook)
    eval_hooks.append(hook)

  return train_monitors, eval_hooks


def create_experiment(run_config,
                      hparams,
                      model_name,
                      problem_name,
                      data_dir,
                      train_steps,
                      eval_steps,
                      min_eval_frequency=2000,
                      schedule="train_and_evaluate",
                      export=False,
                      decode_hparams=None,
                      use_tfdbg=False,
                      use_dbgprofile=False,
                      eval_early_stopping_steps=None,
                      eval_early_stopping_metric=None,
                      eval_early_stopping_metric_delta=None,
                      eval_early_stopping_metric_minimize=True,
                      use_tpu=False):
  """Create Experiment."""
  # HParams
  hparams.add_hparam("model_dir", run_config.model_dir)
  hparams.add_hparam("data_dir", data_dir)
  hparams.add_hparam("train_steps", train_steps)
  hparams.add_hparam("eval_steps", eval_steps)
  add_problem_hparams(hparams, problem_name)

  # Estimator
  estimator = create_estimator(
      model_name,
      hparams,
      run_config,
      schedule=schedule,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu)

  # Input fns from Problem
  problem = hparams.problem_instances[0]
  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams)
  eval_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL, hparams)

  # Export
  if export:
    tf.logging.warn("Exporting from the trainer is deprecated. "
                    "See serving/export.py.")

  # Hooks
  hooks_kwargs = {}
  if not use_tpu:
    dbgprofile_kwargs = {"output_dir": run_config.model_dir}
    validation_monitor_kwargs = dict(
        input_fn=eval_input_fn,
        eval_steps=eval_steps,
        every_n_steps=min_eval_frequency,
        early_stopping_rounds=eval_early_stopping_steps,
        early_stopping_metric=eval_early_stopping_metric,
        early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
    early_stopping_kwargs = dict(
        events_dir=os.path.join(run_config.model_dir, "eval_continuous"),
        tag=eval_early_stopping_metric,
        num_plateau_steps=eval_early_stopping_steps,
        plateau_decrease=eval_early_stopping_metric_minimize,
        plateau_delta=eval_early_stopping_metric_delta,
        every_n_steps=min_eval_frequency)

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
    train_monitors, eval_hooks = create_hooks(
        use_tfdbg=use_tfdbg,
        use_dbgprofile=use_dbgprofile,
        dbgprofile_kwargs=dbgprofile_kwargs,
        use_validation_monitor=use_validation_monitor,
        use_early_stopping=use_early_stopping,
        validation_monitor_kwargs=validation_monitor_kwargs,
        early_stopping_kwargs=early_stopping_kwargs)
    hooks_kwargs = {"train_monitors": train_monitors, "eval_hooks": eval_hooks}

  # Experiment
  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps,
      min_eval_frequency=min_eval_frequency,
      train_steps_per_iteration=min(min_eval_frequency, train_steps),
      eval_delay_secs=0 if schedule == "evaluate" else 120,
      **hooks_kwargs)


def create_experiment_fn(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn


def create_export_strategy(problem, hparams):
  return tf.contrib.learn.make_export_strategy(
      lambda: problem.serving_input_fn(hparams), as_text=True)


def add_problem_hparams(hparams, problem_name):
  """Add problem hparams for the problems."""
  problem = registry.problem(problem_name)
  p_hparams = problem.get_hparams(hparams)

  hparams.problem_instances = [problem]
  hparams.problems = [p_hparams]


def set_random_seed(seed):
  tf.set_random_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
