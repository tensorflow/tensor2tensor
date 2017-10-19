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

"""Library for training on TPU. See tpu_trainer.py.

Currently only supports training and evaluation for text-to-text problems.
"""

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
from tensorflow.python.util import nest


def get_input_fn(data_dir, problem, hparams):
  """Get basic T2T input fn."""

  def input_fn(mode, params):
    """Input fn."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    num_threads = 4 if is_training else 1
    batch_size = params["batch_size"]

    batching_scheme = {
        "boundaries": [],
        "batch_sizes": [batch_size],
        "min_length": hparams.min_length,
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
        problem.filepattern(data_dir, mode))
    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.map(decode_record, num_parallel_calls=num_threads)

    def _preprocess(example, problem, hparams, mode):
      example = problem.preprocess_example(example, mode, hparams)
      # We do not want int64s as they are not supported on TPUs.
      example = data_reader.cast_int64_to_int32(example)
      return example

    dataset = dataset.map(
        lambda ex: _preprocess(ex, problem, hparams, mode),
        num_parallel_calls=num_threads)

    def _valid_size(example):
      return data_reader.example_valid_size(
          example, batching_scheme["min_length"], batching_scheme["max_length"])

    dataset = dataset.filter(_valid_size)
    if is_training:
      dataset = dataset.shuffle(100)
    # TODO(rsepassi): In eval mode, should not repeat
    dataset = dataset.repeat(None)
    dataset = data_reader.padded_batch(dataset, batch_size,
                                       batching_scheme["padded_shapes"])

    if not is_training:
      dataset = dataset.map(
          lambda f: pad_batch(f, batch_size), num_parallel_calls=num_threads)

    def shape_def(example):
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
      for _, t in example.iteritems():
        shape = t.get_shape().as_list()
        shape[0] = batch_size
        t.set_shape(t.get_shape().merge_with(shape))
        # Assert shapes are fully known
        t.get_shape().assert_is_fully_defined()

      return example

    dataset = dataset.map(shape_def, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(1)
    features = dataset.make_one_shot_iterator().get_next()

    return features, features["targets"]

  return input_fn


def pad_batch(features, batch_size):
  """Pad each feature in features to batch_size on dim 0."""
  ts = []
  for t in nest.flatten(features):
    before_pads = [0] * t.get_shape().ndims
    after_pads = [0] * t.get_shape().ndims
    batch_pad = tf.convert_to_tensor(batch_size) - tf.shape(t)[0]
    after_pads[0] = batch_pad
    pads = list(zip(before_pads, after_pads))
    old_shape = t.get_shape().as_list()
    old_shape[0] = batch_size
    t = tf.pad(t, pads)
    t.set_shape(old_shape)
    ts.append(t)
  return nest.pack_sequence_as(features, ts)


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

    # Ensure the length is known statically
    shape = [None] * logits.get_shape().ndims
    shape[1] = hparams.max_length
    logits.set_shape(logits.get_shape().merge_with(shape))

    # Loss
    loss_num, loss_den = target_modality.loss(logits, labels)
    loss = loss_num / tf.maximum(1.0, loss_den)

    if mode == tf.estimator.ModeKeys.EVAL:
      problem = hp.problem_instances[0]
      eval_metrics_fn = create_eval_metrics_fn(problem)
      _remove_summaries()
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
      master=master,
      evaluation_master=master)

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
  hp.optimizer = "TrueAdam"
  hp.learning_rate = 0.4

  # Inputs
  # Each example in the batch will be of (padded) length hp.max_length
  hp.max_length = 64
  hp.tpu_batch_size_per_shard = 20

  return hp
