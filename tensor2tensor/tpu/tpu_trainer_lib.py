# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Library for training on TPU. See tpu_trainer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import trainer_utils

import tensorflow as tf


def create_run_config(master="",
                      model_dir=None,
                      iterations_per_loop=1000,
                      num_shards=8,
                      log_device_placement=False,
                      save_checkpoints_steps=1000,
                      num_gpus=1,
                      gpu_order="",
                      shard_to_cpu=False,
                      use_tpu=True):
  """Create TPUConfig and tpu.RunConfig."""
  session_config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=log_device_placement)
  run_config_args = {
      "model_dir": model_dir,
      "session_config": session_config,
      "save_summary_steps": 0,
      "save_checkpoints_steps": save_checkpoints_steps,
  }
  run_config_cls = tf.estimator.RunConfig

  # If using TPU, use TPU RunConfig, add TPUConfig, and add additional args
  if use_tpu:
    run_config_cls = tf.contrib.tpu.RunConfig
    tpu_config = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=iterations_per_loop,
        num_shards=num_shards,
        per_host_input_for_training=(num_shards <= 8))
    run_config_args["master"] = master
    run_config_args["tpu_config"] = tpu_config

  config = run_config_cls(**run_config_args)

  # If not using TPU, add device info for data_parallelism
  if not use_tpu:
    config.t2t_device_info = {
        "num_gpus": num_gpus,
        "gpu_order": gpu_order,
        "shard_to_cpu": shard_to_cpu,
        "num_shards": max(1, num_gpus + int(shard_to_cpu))
    }

  return config


def create_estimator(model_name, hparams, run_config, use_tpu=True):
  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      model_name, hparams, use_tpu=use_tpu)

  if use_tpu:
    batch_size = hparams.tpu_batch_size_per_shard
    batch_size *= run_config.tpu_config.num_shards
    return tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size * 2)
  else:
    return tf.estimator.Estimator(
        model_fn=model_fn, model_dir=run_config.model_dir, config=run_config)


def create_experiment(run_config,
                      hparams,
                      model_name,
                      problem_name,
                      data_dir,
                      train_steps,
                      eval_steps,
                      min_eval_frequency,
                      use_tpu=True):
  """Create Experiment."""
  # HParams
  hparams.add_hparam("data_dir", data_dir)
  trainer_utils.add_problem_hparams(hparams, problem_name)

  # Estimator
  estimator = create_estimator(model_name, hparams, run_config, use_tpu=use_tpu)

  # Input fns from Problem
  problem = hparams.problem_instances[0]
  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams)
  eval_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL, hparams)

  # Experiment
  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps,
      min_eval_frequency=min_eval_frequency,
      train_steps_per_iteration=min_eval_frequency)


def create_experiment_fn(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn
