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

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import metrics
from tensor2tensor.utils import optimize
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils

import tensorflow as tf


def create_dummy_vars():
  """Dummy vars for restore to work when not using TPU codepath."""
  with tf.variable_scope("losses_avg"):
    with tf.variable_scope("problem_0"):
      for var_name in ["total", "extra", "training"]:
        tf.get_variable(
            "%s_loss" % var_name, initializer=100.0, trainable=False)
  with tf.variable_scope("train_stats"):
    tf.get_variable("problem_0_steps", initializer=0, trainable=False)


def get_input_fn(mode, hparams):
  """Get basic T2T input fn."""

  def input_fn(params):
    """Input fn."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    num_threads = 4 if is_training else 1
    if "batch_size" in params:
      batch_size = params["batch_size"]
    else:
      batch_size = hparams.tpu_batch_size_per_shard

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
    problem = hparams.problem_instances[0]
    data_dir = hparams.data_dir
    dataset = problem.dataset(
        mode=mode, data_dir=data_dir, num_threads=num_threads, hparams=hparams)
    dataset = dataset.map(
        data_reader.cast_int64_to_int32, num_threads=num_threads)
    if is_training:
      dataset = dataset.repeat(None)

    # Batch (and pad)
    if are_shapes_fully_defined(dataset.output_shapes):
      dataset = dataset.apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
      # If shapes are not fully defined, filter out long ones and pad to
      # hparams.max_length
      dataset = dataset.filter(valid_size)
      padded_shapes = fill_shape_nones(
          dataset.output_shapes, none_filler=hparams.max_length)
      if hasattr(tf.contrib.data, "padded_batch_and_drop_remainder"):
        dataset = dataset.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size, padded_shapes))
      else:
        dataset = data_reader.padded_batch(dataset, batch_size, padded_shapes)

    dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(1)
    features = dataset.make_one_shot_iterator().get_next()

    return features, features["targets"]

  return input_fn


def are_shapes_fully_defined(shapes_dict):
  for shape in shapes_dict.values():
    if not shape.is_fully_defined():
      return False
  return True


def fill_shape_nones(shapes_dict, none_filler=None):
  padded_shapes = {}
  for key, shape in six.iteritems(shapes_dict):
    padded_shapes[key] = [
        (dim if dim is not None else none_filler) for dim in shape.as_list()
    ]
  return padded_shapes


def get_model_fn(model_name, hp, use_tpu=True):
  """Get simple T2T model fn."""

  def model_fn(features, labels, mode, params, config):
    """Model fn."""
    del params
    del config
    create_dummy_vars()

    hparams = copy.deepcopy(hp)
    problem_hp = hparams.problems[0]
    orig_features = features

    # Instantiate model and retrieve modalities. Note that autoregressive models
    # have no input modality.
    model_class = registry.model(model_name)(hparams, mode, problem_hp)
    input_modality = problem_hp.input_modality.get("inputs")
    target_modality = problem_hp.target_modality

    # Transform features
    transformed_features = {}
    if input_modality is not None:
      with tf.variable_scope(input_modality.name):
        transformed_features["inputs"] = input_modality.bottom(
            features["inputs"])
    with tf.variable_scope(target_modality.name):
      transformed_features["targets"] = target_modality.targets_bottom(
          features["targets"])
    transformed_features["problem_choice"] = tf.constant(0)
    transformed_features["input_space_id"] = tf.constant(
        problem_hp.input_space_id)
    transformed_features["target_space_id"] = tf.constant(
        problem_hp.target_space_id)

    # Model construction
    with tf.variable_scope("body"):
      outputs = model_class.model_fn_body(transformed_features)
    with tf.variable_scope(target_modality.name):
      logits = target_modality.top(outputs, labels)

      # If the length dim is unknown fix it to max_length
      if use_tpu and logits.get_shape().as_list()[1] is None:
        shape = logits.get_shape().as_list()
        shape[1] = hparams.max_length
        logits.set_shape(shape)

      # Loss
      loss_num, loss_den = target_modality.loss(logits, labels)
      loss = loss_num / tf.maximum(1.0, loss_den)

    if mode == tf.estimator.ModeKeys.EVAL:
      problem = hp.problem_instances[0]

      if use_tpu:
        eval_metrics_fn = create_eval_metrics_fn(problem)
        _remove_summaries()
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode,
            eval_metrics=(eval_metrics_fn, [logits, orig_features["targets"]]),
            loss=loss)
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

    # Learning rate
    lr = hparams.learning_rate * optimize.learning_rate_decay(hparams)

    # Optimizer
    opt = optimize.ConditionalOptimizer(hparams.optimizer, lr, hparams)
    if use_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)

    # Optimize
    gradients = opt.compute_gradients(loss, tf.trainable_variables())
    if hparams.clip_grad_norm:
      gradients = _clip_gradients_by_norm(gradients, hparams.clip_grad_norm)
    train_op = opt.apply_gradients(
        gradients, global_step=tf.train.get_or_create_global_step())
    with tf.control_dependencies([train_op]):
      train_op = tf.identity(loss)

    _remove_summaries()
    if use_tpu:
      return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
    else:
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  return model_fn


TPU_METRIC_BLACKLIST = set([
    metrics.Metrics.APPROX_BLEU,
    metrics.Metrics.ROUGE_2_F,
    metrics.Metrics.ROUGE_L_F,
])


def create_eval_metrics_fn(problem):
  """Create the metrics_fn that TPUEstimatorSpec expects."""

  def make_metric_fn(metric_fn):

    def wrapped_metric_fn(logits, labels):
      num, den = metric_fn(
          logits, labels, weights_fn=common_layers.weights_nonzero)
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
                      save_checkpoints_steps=1000):
  """Create TPUConfig and tpu.RunConfig."""
  tpu_config = tf.contrib.tpu.TPUConfig(
      iterations_per_loop=iterations_per_loop,
      num_shards=num_shards,
      per_host_input_for_training=(num_shards <= 8))
  session_config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=log_device_placement)
  run_config = tf.contrib.tpu.RunConfig(
      model_dir=model_dir,
      session_config=session_config,
      save_summary_steps=0,
      save_checkpoints_steps=save_checkpoints_steps,
      tpu_config=tpu_config,
      master=master)
  return run_config


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
  batch_size = (
      hparams.tpu_batch_size_per_shard * run_config.tpu_config.num_shards)
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


def make_experiment_fn(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn
