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

# TODO(rsepassi):
# * Fix EVAL (breaks when loading from checkpoint)
# * Support all decoders
# * Share more code with Problem.dataset and input_pipeline
# * Support PREDICT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

# Dependency imports

from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import data_reader
from tensor2tensor.utils import metrics
from tensor2tensor.utils import model_builder
from tensor2tensor.utils import registry

import tensorflow as tf


def get_input_fn(data_dir, problem, hparams):
  """Get basic T2T input fn."""

  def input_fn(mode, params):
    """Input fn."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    num_threads = 4 if is_training else 1
    batch_size = params["batch_size"]

    data_file_patterns = [problem.filepattern(data_dir, mode)]

    batching_scheme = {
        "boundaries": [],
        "batch_sizes": [batch_size],
        "max_length": hparams.max_length,
        "window_size": batch_size,
        "padded_shapes": {
            "inputs": [hparams.max_length],
            "targets": [hparams.max_length],
        },
    }

    def decode_record(record):
      """Serialized Example to dict of <feature name, Tensor>."""
      data_fields, _ = problem.example_reading_spec()
      decoded = tf.parse_single_example(record, features=data_fields)
      decoded["inputs"] = decoded["inputs"].values
      decoded["targets"] = decoded["targets"].values
      return decoded

    data_files = tf.contrib.slim.parallel_reader.get_data_files(
        data_file_patterns)
    dataset = tf.contrib.data.TFRecordDataset(data_files)
    dataset = dataset.map(decode_record, num_threads=num_threads)

    def _preprocess(example, problem, hparams, mode):
      example = problem.preprocess_example(example, mode, hparams)
      # We do not want int64s as they are not supported on TPUs.
      example = data_reader.cast_int64_to_int32(example)
      return example

    dataset = dataset.map(
        lambda ex: _preprocess(ex, problem, hparams, mode),
        num_threads=num_threads)

    def _valid_size(example):
      return data_reader.example_valid_size(example,
                                            batching_scheme["max_length"])

    dataset = dataset.filter(_valid_size)
    if is_training:
      dataset = dataset.shuffle(100)
      dataset = dataset.repeat(None)
    dataset = data_reader.padded_batch(dataset,
                                       batching_scheme["batch_sizes"][0],
                                       batching_scheme["padded_shapes"])
    dataset.prefetch(1)

    train_features = dataset.make_one_shot_iterator().get_next()

    inputs = train_features["inputs"]
    targets = train_features["targets"]

    # Ensure inputs and targets are proper rank.
    while len(inputs.get_shape()) != 4:
      inputs = tf.expand_dims(inputs, axis=-1)
    while len(targets.get_shape()) != 4:
      targets = tf.expand_dims(targets, axis=-1)

    inputs_shape = inputs.get_shape().as_list()
    inputs_shape[0] = batch_size
    inputs.set_shape(inputs_shape)
    targets_shape = targets.get_shape().as_list()
    targets_shape[0] = batch_size
    targets.set_shape(targets_shape)

    train_features["inputs"] = inputs
    train_features["targets"] = targets

    return train_features, targets

  return input_fn


def get_model_fn(model, hp, use_tpu=True):
  """Get simple T2T model fn."""

  def model_fn(features, labels, mode, params, config):
    """Model fn."""
    del params
    hparams = copy.deepcopy(hp)
    problem_hp = hparams.problems[0]
    orig_features = features

    # Instantiate model and retrieve modalities
    model_class = registry.model(model)(hparams, mode, problem_hp)
    input_modality = problem_hp.input_modality["inputs"]
    target_modality = problem_hp.target_modality

    # Model construction
    features = {
        "inputs": input_modality.bottom(features["inputs"]),
        "targets": target_modality.targets_bottom(features["targets"]),
        "problem_choice": tf.constant(0),
        "input_space_id": tf.constant(problem_hp.input_space_id),
        "target_space_id": tf.constant(problem_hp.target_space_id)
    }
    outputs = model_class.model_fn_body(features)
    logits = target_modality.top(outputs, labels)

    # Loss
    loss_num, loss_den = target_modality.loss(logits, labels)
    loss = loss_num / tf.maximum(1.0, loss_den)

    if mode == tf.estimator.ModeKeys.EVAL:
      problem = hp.problem_instances[0]
      eval_metrics_fn = create_eval_metrics_fn(problem)
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode,
          eval_metrics=(eval_metrics_fn, [logits, orig_features["targets"]]),
          loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN

    # Learning rate
    num_shards = config.tpu_config.num_shards
    lr = hparams.learning_rate * model_builder.learning_rate_decay(
        hparams, num_worker_replicas=num_shards)
    lr /= math.sqrt(float(num_shards))

    # Optimizer
    opt = model_builder.ConditionalOptimizer(hparams.optimizer, lr, hparams)
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
    return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)

  return model_fn


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


def make_estimator(model_fn,
                   output_dir,
                   master="",
                   batch_size=16,
                   iterations_per_loop=1000,
                   num_shards=8,
                   per_host_input_for_training=True,
                   use_tpu=True,
                   log_device_placement=False,
                   save_checkpoints_steps=1000):
  """Make TPUEstimator."""
  tpu_config = tf.contrib.tpu.TPUConfig(
      iterations_per_loop=iterations_per_loop,
      num_shards=num_shards,
      per_host_input_for_training=per_host_input_for_training)
  session_config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=log_device_placement)
  run_config = tf.contrib.tpu.RunConfig(
      session_config=session_config,
      save_summary_steps=0,
      save_checkpoints_steps=save_checkpoints_steps,
      tpu_config=tpu_config,
      master=master)

  return tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=use_tpu,
      model_dir=output_dir,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size * 2)


@registry.register_hparams
def transformer_tpu():
  """HParams for Transformer model on TPU."""
  hp = transformer.transformer_base()
  hp.use_pad_remover = int(False)  # where op not supported

  # Inputs
  # Each example in the batch will be of (padded) length hp.max_length
  # Batch size per shard is governed by tpu_batch_size_per_shard
  hp.max_length = 64

  hp.optimizer = "TrueAdam"
  hp.layer_preprocess_sequence = "n"
  hp.layer_postprocess_sequence = "da"
  return hp
