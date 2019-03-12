# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

import collections
import json
import os
import random
import numpy as np

from tensor2tensor.utils import decoding
from tensor2tensor.utils import devices
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import metrics_hook
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug


create_hparams = hparams_lib.create_hparams
add_problem_hparams = hparams_lib.add_problem_hparams


def next_checkpoint(model_dir, timeout_mins=240):
  """Yields successive checkpoints from model_dir.

  Args:
    model_dir: The directory in which checkpoints are saved.
    timeout_mins: The maximum amount of time in minutes to wait
                  between checkpoints. Set this to -1 to wait indefinitely.
  Yields:
    last_ckpt: a new checkpoint path, or None if the timeout was reached.
  """
  last_ckpt = None
  timeout_secs = None
  if timeout_mins != -1:
    timeout_secs = timeout_mins * 60
  while True:
    last_ckpt = tf.contrib.training.wait_for_new_checkpoint(
        model_dir, last_ckpt, seconds_to_sleep=60, timeout=timeout_secs)

    if last_ckpt is None:
      tf.logging.info(
          "Eval timeout: no new checkpoints within %dm" % timeout_mins)
      break

    yield last_ckpt


def next_undecoded_checkpoint(model_dir, timeout_mins=240):
  """Yields successive checkpoints from model_dir."""
  last_ckpt = None
  last_step = 0
  while True:
    # Get the latest checkpoint.
    last_ckpt = tf.contrib.training.wait_for_new_checkpoint(
        model_dir, last_ckpt, seconds_to_sleep=60, timeout=60 * timeout_mins)
    # Get all the checkpoint from the model dir.
    ckpt_path = tf.train.get_checkpoint_state(model_dir)
    all_model_checkpoint_paths = ckpt_path.all_model_checkpoint_paths
    ckpt_step = np.inf
    next_ckpt = None
    # Find the next checkpoint to eval based on last_step.
    for ckpt in all_model_checkpoint_paths:
      step = int(os.path.basename(ckpt).split("-")[1])
      if step > last_step and step < ckpt_step:
        ckpt_step = step
        next_ckpt = ckpt

    # If all the checkpoints have been evaluated.
    if last_ckpt is None and next_ckpt is None:
      tf.logging.info(
          "Eval timeout: no new checkpoints within %dm" % timeout_mins)
      break

    if next_ckpt is not None:
      last_step = ckpt_step
      last_ckpt = next_ckpt

    yield last_ckpt


def create_session_config(log_device_placement=False,
                          enable_graph_rewriter=False,
                          gpu_mem_fraction=0.95,
                          use_tpu=False,
                          xla_jit_level=tf.OptimizerOptions.OFF,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0):
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
              opt_level=tf.OptimizerOptions.L1,
              do_function_inlining=False,
              global_jit_level=xla_jit_level))

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)

  config = tf.ConfigProto(
      allow_soft_placement=True,
      graph_options=graph_options,
      gpu_options=gpu_options,
      log_device_placement=log_device_placement,
      inter_op_parallelism_threads=inter_op_parallelism_threads,
      intra_op_parallelism_threads=intra_op_parallelism_threads,
      isolate_session_state=True)
  return config


def is_cloud_async_distributed():
  return ("chief" in
          json.loads(os.environ.get("TF_CONFIG", "{}")).get("cluster", {}))


def create_run_config(model_name,
                      master="",
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
                      num_async_replicas=1,
                      enable_graph_rewriter=False,
                      gpu_mem_fraction=0.95,
                      no_data_parallelism=False,
                      optionally_use_dist_strat=False,
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
                      use_tpu=False,
                      use_tpu_estimator=False,
                      xla_jit_level=tf.OptimizerOptions.OFF,
                      inter_op_parallelism_threads=0,
                      log_step_count_steps=100,
                      intra_op_parallelism_threads=0,
                      tpu_config_extra_kwargs=None,
                      cloud_tpu_name=""):
  """Create RunConfig, TPUConfig, and Parallelism object."""
  session_config = create_session_config(
      log_device_placement=log_device_placement,
      enable_graph_rewriter=enable_graph_rewriter,
      gpu_mem_fraction=gpu_mem_fraction,
      use_tpu=use_tpu,
      xla_jit_level=xla_jit_level,
      inter_op_parallelism_threads=inter_op_parallelism_threads,
      intra_op_parallelism_threads=intra_op_parallelism_threads)
  run_config_args = {
      "master": master,
      "evaluation_master": master,
      "model_dir": model_dir,
      "session_config": session_config,
      "save_summary_steps": 100,
      "save_checkpoints_steps": save_checkpoints_steps,
      "save_checkpoints_secs": save_checkpoints_secs,
      "keep_checkpoint_max": keep_checkpoint_max,
      "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
      "tf_random_seed": random_seed,
      "log_step_count_steps": log_step_count_steps
  }
  if save_checkpoints_secs:
    del run_config_args["save_checkpoints_steps"]
  run_config_cls = tf.contrib.learn.RunConfig

  if use_tpu or use_tpu_estimator:
    # If using TPUEstimator, use TPU RunConfig, add TPUConfig, and add
    # additional args.
    tpu_config_kwargs = {
        "iterations_per_loop": iterations_per_loop,
        "num_shards": num_shards,
        "per_host_input_for_training": True,
        "initial_infeed_sleep_secs": tpu_infeed_sleep_secs,
    }
    if tpu_config_extra_kwargs is not None:
      tpu_config_kwargs.update(tpu_config_extra_kwargs)
    run_config_cls = tf.contrib.tpu.RunConfig
    tpu_config = tf.contrib.tpu.TPUConfig(
        **tpu_config_kwargs)
    run_config_args["tpu_config"] = tpu_config
    if not master and "KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS" in os.environ:
      # If running on TPU but no master is set and the KUBE env var is present
      # then we're running on ML Engine. Set the master.
      run_config_args["master"] = os.environ[
          "KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS"]
      run_config_args["evaluation_master"] = run_config_args["master"]
    elif not master and cloud_tpu_name:
      # Update run_config to use cluster instead of master/evaluation_master
      # as we need the cluster spec to use Cloud Pods
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          cloud_tpu_name)
      run_config_args["cluster"] = tpu_cluster_resolver
      del run_config_args["master"]
      del run_config_args["evaluation_master"]
  elif is_cloud_async_distributed():
    run_config_cls = tf.estimator.RunConfig
    del run_config_args["master"]
    del run_config_args["evaluation_master"]

  config = run_config_cls(**run_config_args)

  # If not using TPU, add device info for data_parallelism
  config.use_tpu = use_tpu
  if not use_tpu:
    config.t2t_device_info = {
        "num_async_replicas": num_async_replicas,
    }
    use_distribution_strategy = (
        optionally_use_dist_strat and
        t2t_model.T2TModel.has_symmetric_shards(model_name) and
        not no_data_parallelism and ps_replicas == 0 and ps_gpu == 0 and
        num_async_replicas == 1)

    if use_distribution_strategy:
      tf.logging.info(
          "Configuring MirroredStrategy DistributionStrategy to replicate the "
          "model."
      )
      distribution = tf.contrib.distribute.MirroredStrategy()
      config = config.replace(train_distribute=distribution)
      config.data_parallelism = None
    else:
      tf.logging.info("Configuring DataParallelism to replicate the model.")
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
          worker_job=worker_job,
          no_data_parallelism=no_data_parallelism)

  return config


def create_estimator(model_name,
                     hparams,
                     run_config,
                     schedule="train_and_evaluate",
                     decode_hparams=None,
                     use_tpu=False,
                     use_tpu_estimator=False,
                     use_xla=False):
  """Create a T2T Estimator."""
  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      model_name, hparams, decode_hparams=decode_hparams, use_tpu=use_tpu)


  del use_xla
  if use_tpu or use_tpu_estimator:
    problem = hparams.problem
    batch_size = (
        problem.tpu_batch_size_per_shard(hparams) *
        run_config.tpu_config.num_shards)
    mlperf_log.transformer_print(
        key=mlperf_log.INPUT_BATCH_SIZE, value=batch_size)
    if getattr(hparams, "mtf_mode", False):
      batch_size = problem.tpu_batch_size_per_shard(hparams)
    predict_batch_size = batch_size
    if decode_hparams and decode_hparams.batch_size:
      predict_batch_size = decode_hparams.batch_size
    if decode_hparams and run_config.tpu_config:
      decode_hparams.add_hparam("iterations_per_loop",
                                run_config.tpu_config.iterations_per_loop)
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
        use_tpu=use_tpu,
        train_batch_size=batch_size,
        eval_batch_size=batch_size if "eval" in schedule else None,
        predict_batch_size=predict_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
    )
  return estimator


def create_hooks(use_tfdbg=False,
                 use_dbgprofile=False,
                 dbgprofile_kwargs=None,
                 use_validation_monitor=False,
                 validation_monitor_kwargs=None,
                 use_early_stopping=False,
                 early_stopping_kwargs=None):
  """Create train and eval hooks for Experiment."""
  train_hooks = []
  eval_hooks = []

  if use_tfdbg:
    hook = debug.LocalCLIDebugHook()
    train_hooks.append(hook)
    eval_hooks.append(hook)

  if use_dbgprofile:
    # Recorded traces can be visualized with chrome://tracing/
    # The memory/tensor lifetime is also profiled
    tf.logging.info("Using ProfilerHook")
    defaults = dict(save_steps=10, show_dataflow=True, show_memory=True)
    defaults.update(dbgprofile_kwargs)
    train_hooks.append(tf.train.ProfilerHook(**defaults))

  if use_validation_monitor:
    tf.logging.info("Using ValidationMonitor")
    train_hooks.append(
        tf.contrib.learn.monitors.ValidationMonitor(
            hooks=eval_hooks, **validation_monitor_kwargs))

  if use_early_stopping:
    tf.logging.info("Using EarlyStoppingHook")
    hook = metrics_hook.EarlyStoppingHook(**early_stopping_kwargs)
    # Adding to both training and eval so that eval aborts as well
    train_hooks.append(hook)
    eval_hooks.append(hook)

  return train_hooks, eval_hooks


class HookContext(collections.namedtuple(
    "HookContext",
    ["estimator", "problem", "hparams"])):
  pass


class T2TExperiment(object):
  """Custom Experiment class for running distributed experiments."""

  def __init__(self, estimator, hparams, train_spec, eval_spec,
               use_validation_monitor, decode_hparams=None):
    self._train_spec = train_spec
    self._eval_spec = eval_spec
    self._hparams = hparams
    self._decode_hparams = decode_hparams
    self._estimator = estimator
    self._use_validation_monitor = use_validation_monitor

  @property
  def estimator(self):
    return self._estimator

  @property
  def train_steps(self):
    return self._train_spec.max_steps

  @property
  def eval_steps(self):
    return self._eval_spec.steps

  def continuous_train_and_eval(self, continuous_eval_predicate_fn=None):
    del continuous_eval_predicate_fn
    tf.estimator.train_and_evaluate(self._estimator, self._train_spec,
                                    self._eval_spec)
    return self.evaluate()

  def train_and_evaluate(self):
    if self._use_validation_monitor:
      tf.logging.warning("EvalSpec not provided. Estimator will not manage "
                         "model evaluation. Assuming ValidationMonitor present "
                         "in train_hooks.")
      self.train()

  def train(self, max_steps=None):
    mlperf_log.transformer_print(key=mlperf_log.TRAIN_LOOP)
    mlperf_log.transformer_print(key=mlperf_log.TRAIN_EPOCH, value=0)
    self._estimator.train(
        self._train_spec.input_fn,
        hooks=self._train_spec.hooks,
        max_steps=max_steps or self._train_spec.max_steps)

  def train_eval_and_decode(self):
    """Does eval and decode after training every eval_freq_in_steps."""
    eval_steps = self._hparams.eval_freq_in_steps
    packed_dataset = "_packed" in self._hparams.problem.name
    mlperf_log.transformer_print(key=mlperf_log.TRAIN_LOOP)
    for i in range(0, self._train_spec.max_steps, eval_steps):
      mlperf_log.transformer_print(
          key=mlperf_log.TRAIN_EPOCH, value=i // eval_steps)
      if packed_dataset and i > 0:
        problem = registry.problem(self._hparams.problem.name + "_packed")
        p_hparams = problem.get_hparams(self._hparams)
        self._hparams.problem = problem
        self._hparams.problem_hparams = p_hparams
      self._estimator.train(
          self._train_spec.input_fn,
          steps=eval_steps,
          hooks=self._train_spec.hooks)
      self._set_eval_dir_name("eval")
      self._estimator.evaluate(
          self._eval_spec.input_fn,
          steps=self._eval_spec.steps,
          hooks=self._eval_spec.hooks,
          name="eval")
      if packed_dataset:
        problem = registry.problem(
            self._hparams.problem.name.replace("_packed", ""))
        p_hparams = problem.get_hparams(self._hparams)
        self._hparams.problem = problem
        self._hparams.problem_hparams = p_hparams
      mlperf_log.transformer_print(key=mlperf_log.EVAL_START)
      if self._hparams.mlperf_mode:
        self._decode_hparams.mlperf_decode_step = i + eval_steps
      self.decode(dataset_split=tf.estimator.ModeKeys.EVAL)
      d_hparams = self._decode_hparams
      if self._hparams.mlperf_mode and d_hparams.mlperf_success:
        mlperf_log.transformer_print(
            key=mlperf_log.RUN_STOP, value={"success": "true"})
        break

    d_hparams = self._decode_hparams
    if self._hparams.mlperf_mode and not d_hparams.mlperf_success:
      mlperf_log.transformer_print(
          key=mlperf_log.RUN_STOP, value={"success": "false"})

  def _set_eval_dir_name(self, eval_dir_name):
    attr = "eval_dir_name"
    hp = self._hparams
    if attr not in hp:
      hp.add_hparam(attr, "")
    hp.eval_dir_name = eval_dir_name

  def evaluate(self):
    name = "eval"
    self._set_eval_dir_name("eval")
    return self._estimator.evaluate(
        self._eval_spec.input_fn,
        steps=self._eval_spec.steps,
        hooks=self._eval_spec.hooks,
        name=name)

  def evaluate_on_train_data(self):
    name = "eval_train"
    self._set_eval_dir_name(name)
    self._estimator.evaluate(
        self._train_spec.input_fn,
        steps=self._eval_spec.steps,
        hooks=self._eval_spec.hooks,
        name=name)

  def continuous_eval(self):
    """Evaluate until checkpoints stop being produced."""
    for ckpt_path in next_checkpoint(self._hparams.model_dir,
                                     self._hparams.eval_timeout_mins):
      # Skip zero'th step.
      train_step = decoding.get_step_from_ckpt_path(ckpt_path)
      if train_step == 0:
        tf.logging.info("Skipping evaluation at step 0")
        continue
      self.evaluate()

  def continuous_eval_on_train_data(self):
    """Evaluate on train data until checkpoints stop being produced."""
    for ckpt_path in next_checkpoint(self._hparams.model_dir,
                                     self._hparams.eval_timeout_mins):
      # Skip zero'th step.
      train_step = decoding.get_step_from_ckpt_path(ckpt_path)
      if train_step == 0:
        tf.logging.info("Skipping evaluation at step 0")
        continue
      self.evaluate_on_train_data()

  def test(self):
    """Perform 1 train step and 1 eval step."""
    if self._use_validation_monitor:
      return self.train_and_evaluate()

    self._estimator.train(
        self._train_spec.input_fn, hooks=self._train_spec.hooks, max_steps=1)

    self._estimator.evaluate(
        self._eval_spec.input_fn, steps=1, hooks=self._eval_spec.hooks)

  def run_std_server(self):
    """Starts a TensorFlow server and joins the serving thread.

    Typically used for parameter servers.

    Raises:
      ValueError: if not enough information is available in the estimator's
        config to create a server.
    """
    config = tf.estimator.RunConfig()
    server = tf.train.Server(
        config.cluster_spec,
        job_name=config.task_type,
        task_index=config.task_id,
        protocol=self._hparams.std_server_protocol)
    server.join()

  def decode(self,
             dataset_split=None,
             decode_from_file=False,
             checkpoint_path=None):
    """Decodes from dataset or file."""
    if decode_from_file:
      decoding.decode_from_file(self._estimator,
                                self._decode_hparams.decode_from_file,
                                self._hparams,
                                self._decode_hparams,
                                self._decode_hparams.decode_to_file)
    else:
      decoding.decode_from_dataset(
          self._estimator,
          self._hparams.problem.name,
          self._hparams,
          self._decode_hparams,
          dataset_split=dataset_split,
          checkpoint_path=checkpoint_path)

  def continuous_decode(self):
    """Decode from dataset on new checkpoint."""
    for _ in next_checkpoint(self._hparams.model_dir,
                             self._decode_hparams.decode_timeout_mins):
      self.decode()

  def continuous_decode_on_train_data(self):
    """Decode from dataset on new checkpoint."""
    for _ in next_checkpoint(self._hparams.model_dir,
                             self._decode_hparams.decode_timeout_mins):
      self.decode(dataset_split=tf.estimator.ModeKeys.TRAIN)

  def continuous_decode_on_eval_data(self):
    """Decode from dataset on new checkpoint."""
    if self._hparams.mlperf_mode:
      ckpt_generator = next_undecoded_checkpoint(
          self._hparams.model_dir, self._decode_hparams.decode_timeout_mins)
    else:
      ckpt_generator = next_checkpoint(self._hparams.model_dir,
                                       self._decode_hparams.decode_timeout_mins)

    for ckpt in ckpt_generator:
      current_step = decoding.get_step_from_ckpt_path(ckpt)
      tf.logging.info("Decoding step %d" % current_step)
      # Skip checkpoint 0.
      if current_step == 0:
        continue
      # Decode the latest checkpoint by default.
      checkpoint_path = None
      if self._hparams.mlperf_mode:
        self._decode_hparams.mlperf_decode_step = current_step
        checkpoint_path = ckpt

      mlperf_log.transformer_print(key=mlperf_log.EVAL_START)
      self.decode(
          dataset_split=tf.estimator.ModeKeys.EVAL,
          checkpoint_path=checkpoint_path)
      d_hparams = self._decode_hparams
      if self._hparams.mlperf_mode and d_hparams.mlperf_success:
        mlperf_log.transformer_print(
            key=mlperf_log.RUN_STOP, value={"success": "true"})
        break

    d_hparams = self._decode_hparams
    if self._hparams.mlperf_mode and not d_hparams.mlperf_success:
      mlperf_log.transformer_print(
          key=mlperf_log.RUN_STOP, value={"success": "false"})

  def continuous_decode_from_file(self):
    """Decode from file on new checkpoint."""
    for _ in next_checkpoint(self._hparams.model_dir,
                             self._decode_hparams.decode_timeout_mins):
      self.decode(decode_from_file=True)


def create_experiment(
    run_config,
    hparams,
    model_name,
    problem_name,
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
    eval_timeout_mins=240,
    eval_use_test_set=False,
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
  hparams.add_hparam("eval_freq_in_steps", min_eval_frequency)
  hparams.add_hparam("eval_timeout_mins", eval_timeout_mins)
  if decode_hparams is not None:
    decode_hparams.add_hparam("decode_from_file", decode_from_file)
    decode_hparams.add_hparam("decode_to_file", decode_to_file)
    decode_hparams.add_hparam("decode_reference", decode_reference)
  add_problem_hparams(hparams, problem_name)

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
  problem = hparams.problem
  train_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.TRAIN,
                                                   hparams)

  dataset_split = "test" if eval_use_test_set else None
  dataset_kwargs = {"dataset_split": dataset_split}
  eval_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.EVAL,
                                                  hparams,
                                                  dataset_kwargs=dataset_kwargs)

  # Export
  exporter = None
  if export:
    def compare_fn(best_eval_result, current_eval_result):
      metric = eval_early_stopping_metric or "loss"
      return current_eval_result[metric] < best_eval_result[metric]

    def serving_input_receiver_fn(hparams, decode_hparams, use_tpu):
      return problem.serving_input_fn(hparams, decode_hparams, use_tpu)

    exporter = tf.estimator.BestExporter(
        name="best",
        serving_input_receiver_fn=serving_input_receiver_fn,
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

  hook_context = HookContext(
      estimator=estimator, problem=problem, hparams=hparams)

  train_hooks += t2t_model.T2TModel.get_train_hooks(model_name, hook_context)
  eval_hooks += t2t_model.T2TModel.get_eval_hooks(model_name, hook_context)
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


def set_random_seed(seed):
  tf.set_random_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def restore_checkpoint(ckpt_dir, saver, sess, must_restore=False):
  """Restore from a checkpoint."""
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if must_restore and not ckpt:
    raise ValueError("No checkpoint found in %s" % ckpt_dir)
  if not ckpt:
    return 0

  path = ckpt.model_checkpoint_path
  tf.logging.info("Restoring checkpoint %s", path)
  saver.restore(sess, path)
  step = int(path.split("-")[-1])
  return step
