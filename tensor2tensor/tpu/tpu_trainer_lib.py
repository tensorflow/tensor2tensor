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

import copy

# Dependency imports

import six

from tensor2tensor.utils import data_reader
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import metrics
from tensor2tensor.utils import optimize
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils

import tensorflow as tf


def _create_dummy_vars():
  """Dummy vars for restore to work when not using TPU codepath."""
  with tf.variable_scope("losses_avg"):
    with tf.variable_scope("problem_0"):
      for var_name in ["total", "extra", "training"]:
        tf.get_variable(
            "%s_loss" % var_name, initializer=100.0, trainable=False)
  with tf.variable_scope("train_stats"):
    tf.get_variable("problem_0_steps", initializer=0, trainable=False)


def _get_batch_size(params, hparams, config):
  """Batch size determined by params dict, HParams, and RunConfig."""
  # If params specifies batch size, use that. TPUEstimator passes batch size in
  # params.
  batch_size = params and params.get("batch_size")

  # If not set, then we're running on CPU/GPU, so use the batch size from the
  # hparams, and multiply by the number of data shards.
  if not batch_size:
    batch_size = hparams.tpu_batch_size_per_shard
    if config:
      batch_size *= config.t2t_device_info["num_shards"]

  return batch_size


def t2t_input_fn(problem, mode, hparams, params=None, config=None):
  """Builds input pipeline for problem.

  Args:
    problem: Problem to build input pipeline for
    mode: tf.estimator.ModeKeys
    hparams: HParams
    params: dict, may include "batch_size"
    config: RunConfig

  Returns:
    (features_dict<str name, Tensor feature>, Tensor targets)
  """
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  num_threads = 4 if is_training else 1

  batch_size = _get_batch_size(params, hparams, config)

  def valid_size(example):
    return data_reader.example_valid_size(example, hparams.min_length,
                                          hparams.max_length)

  def define_shapes(example):
    """Set the right shapes for the features."""
    inputs = example["inputs"]
    targets = example["targets"]

    # Ensure inputs and targets are proper rank.
    while len(inputs.get_shape()) < 4:
      inputs = tf.expand_dims(inputs, axis=-1)
    while len(targets.get_shape()) < 4:
      targets = tf.expand_dims(targets, axis=-1)

    example["inputs"] = inputs
    example["targets"] = targets

    # Ensure batch size is set on all features
    for _, t in six.iteritems(example):
      shape = t.get_shape().as_list()
      shape[0] = batch_size
      t.set_shape(t.get_shape().merge_with(shape))
      # Assert shapes are fully known
      t.get_shape().assert_is_fully_defined()

    return example

  # Read and preprocess
  data_dir = hparams.data_dir
  dataset = problem.dataset(
      mode=mode, data_dir=data_dir, num_threads=num_threads, hparams=hparams)
  dataset = dataset.map(
      data_reader.cast_int64_to_int32, num_threads=num_threads)
  if is_training:
    dataset = dataset.repeat(None)

  # Batch (and pad)
  if _are_shapes_fully_defined(dataset.output_shapes):
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
  else:
    # If shapes are not fully defined, filter out long ones and pad to
    # hparams.max_length
    dataset = dataset.filter(valid_size)
    padded_shapes = _fill_shape_nones(
        dataset.output_shapes, none_filler=hparams.max_length)
    dataset = dataset.apply(
        tf.contrib.data.padded_batch_and_drop_remainder(batch_size,
                                                        padded_shapes))

  dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)
  dataset = dataset.prefetch(1)
  features = dataset.make_one_shot_iterator().get_next()

  return features, features["targets"]


def get_input_fn(mode, hparams):
  """Get input fn for Estimator. See input_fn."""

  def wrapped_input_fn(params, config):
    return t2t_input_fn(
        hparams.problem_instances[0],
        mode,
        hparams,
        params=params,
        config=config)

  return wrapped_input_fn


def _are_shapes_fully_defined(shapes_dict):
  for shape in shapes_dict.values():
    if not shape.is_fully_defined():
      return False
  return True


def _fill_shape_nones(shapes_dict, none_filler=None):
  padded_shapes = {}
  for key, shape in six.iteritems(shapes_dict):
    padded_shapes[key] = [
        (dim if dim is not None else none_filler) for dim in shape.as_list()
    ]
  return padded_shapes


def create_data_parallelism(num_gpus=1,
                            gpu_order="",
                            shard_to_cpu=False,
                            num_shards=1):
  """Create Parallelism object."""
  gpus = list(range(num_gpus))
  if gpu_order:
    gpus = [int(s) for s in gpu_order.split(" ")]
    assert len(gpus) == num_gpus
  data_shard_devices = ["gpu:%d" % i for i in gpus]
  if shard_to_cpu or num_gpus < 1:
    data_shard_devices += ["cpu:0"]
  assert len(data_shard_devices) == num_shards
  tf.logging.info("Data parallel devices: %s", data_shard_devices)
  return expert_utils.Parallelism(data_shard_devices, reuse=True)


def t2t_model_fn(model_name,
                 hparams,
                 features,
                 labels,
                 mode,
                 config=None,
                 params=None,
                 use_tpu=True):
  """Model fn.

  Args:
    model_name: str, registered model name.
    hparams: HParams
    features: dict<str name, Tensor feature>
    labels: Tensor
    mode: tf.estimator.ModeKeys
    config: RunConfig
    params: dict, may include batch_size
    use_tpu: bool, whether using TPU

  Returns:
    EstimatorSpec or TPUEstimatorSpec
  """
  _create_dummy_vars()
  hparams = copy.deepcopy(hparams)
  problem = hparams.problem_instances[0]
  problem_hp = hparams.problems[0]
  hparams.use_tpu = use_tpu

  features["problem_choice"] = tf.constant(0)
  features["input_space_id"] = tf.constant(problem_hp.input_space_id)
  features["target_space_id"] = tf.constant(problem_hp.target_space_id)

  # Build and call model
  data_parallelism = (
      expert_utils.Parallelism([""])
      if use_tpu else create_data_parallelism(**config.t2t_device_info))
  model = registry.model(model_name)(
      hparams, mode, problem_hp, data_parallelism=data_parallelism)
  logits, losses_dict = model(features)

  # Set known shapes
  shape = logits.get_shape().as_list()
  if shape[0] is None:
    shape[0] = _get_batch_size(params, hparams, config)
  if shape[1] is None:
    shape[1] = hparams.max_length
  logits.set_shape(shape)

  # Accumulate losses
  assert "training" in losses_dict
  loss = sum(losses_dict.values())

  if mode == tf.estimator.ModeKeys.EVAL:
    if use_tpu:
      eval_metrics_fn = create_eval_metrics_fn(problem, hparams)
      _remove_summaries()
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode, eval_metrics=(eval_metrics_fn, [logits, labels]), loss=loss)
    else:
      eval_metrics_fns = metrics.create_evaluation_metrics([problem], hparams)
      eval_metrics = {}
      for metric_name, metric_fn in six.iteritems(eval_metrics_fns):
        eval_metrics[metric_name] = metric_fn(logits, features)

      return tf.estimator.EstimatorSpec(
          mode,
          predictions={"predictions": logits},
          eval_metric_ops=eval_metrics,
          loss=loss)

  assert mode == tf.estimator.ModeKeys.TRAIN

  lr = hparams.learning_rate * optimize.learning_rate_decay(hparams)
  train_op = optimize.optimize(loss, lr, hparams, use_tpu=use_tpu)

  if use_tpu:
    _remove_summaries()  # summaries not currently working on TPU
    return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
  else:
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def get_model_fn(model_name, hparams, use_tpu=True):
  """Model fn for Estimator. See model_fn."""

  def wrapping_model_fn(features, labels, mode, params, config):
    return t2t_model_fn(
        model_name,
        hparams,
        features,
        labels,
        mode,
        config=config,
        params=params,
        use_tpu=use_tpu)

  return wrapping_model_fn


# These metrics are implemented with py_funcs and therefore do no work with TPU
TPU_METRIC_BLACKLIST = set([
    metrics.Metrics.APPROX_BLEU,
    metrics.Metrics.ROUGE_2_F,
    metrics.Metrics.ROUGE_L_F,
])


def create_eval_metrics_fn(problem, hparams):
  """Create the metrics_fn that TPUEstimatorSpec expects."""

  tm = problem.get_hparams().target_modality
  if isinstance(tm, tuple):
    tm = registry.create_modality(tm, hparams)
  weights_fn = tm.targets_weights_fn

  def make_metric_fn(metric_fn):

    def wrapped_metric_fn(logits, labels):
      num, den = metric_fn(logits, labels, weights_fn=weights_fn)
      return tf.metrics.mean(num, den)

    return wrapped_metric_fn

  metric_fns = []
  eval_metrics = problem.eval_metrics()

  for metric in eval_metrics:
    if metric in TPU_METRIC_BLACKLIST:
      tf.logging.warn("Skipping eval metric %s in TPU_METRIC_BLACKLIST", metric)
      continue
    name = "metrics-%s/%s" % (problem.name, metric)
    metric_fns.append((name, make_metric_fn(metrics.METRICS_FNS[metric])))

  def all_metrics_fn(logits, labels):
    metrics_dict = {}

    for name, fn in metric_fns:
      metrics_dict[name] = fn(logits, labels)

    return metrics_dict

  return all_metrics_fn


def _remove_summaries():
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
  return list(zip(clipped_gradients, variables))


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


def create_estimator(model_fn, run_config, batch_size=16, use_tpu=True):
  if use_tpu:
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
  hparams.add_hparam("data_dir", data_dir)
  trainer_utils.add_problem_hparams(hparams, problem_name)
  batch_size = hparams.tpu_batch_size_per_shard
  if use_tpu:
    batch_size *= run_config.tpu_config.num_shards
  model_fn = get_model_fn(model_name, hparams, use_tpu=use_tpu)
  estimator = create_estimator(
      model_fn, run_config, batch_size, use_tpu=use_tpu)
  train_input_fn = get_input_fn(tf.estimator.ModeKeys.TRAIN, hparams)
  eval_input_fn = get_input_fn(tf.estimator.ModeKeys.EVAL, hparams)
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
