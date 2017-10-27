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

"""Input function building."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.utils import data_reader

import tensorflow as tf


def build_input_fn(mode,
                   hparams,
                   data_dir=None,
                   num_datashards=None,
                   fixed_problem=None,
                   worker_replicas=None,
                   worker_id=None,
                   batch_size=None,
                   dataset_split=None,
                   shard=None):
  """Provides input to the graph, either from disk or via a placeholder.

  This function produces an input function that will feed data into
  the network. There are two modes of operation:

  1. If data_file_pattern and all subsequent arguments are None, then
     it creates a placeholder for a serialized tf.Example proto.
  2. If data_file_pattern is defined, it will read the data from the
     files at the given location. Use this mode for training,
     evaluation, and testing prediction.

  Args:
    mode: The execution mode, as defined in tf.estimator.ModeKeys.
    hparams: HParams object.
    data_dir: directory with input data.
    num_datashards: An integer.
    fixed_problem: An integer indicating the problem to fetch data for, or None
      if the input is to be randomly selected.
    worker_replicas: int, number of worker replicas. Used in multiproblem
      setting with hparams.problem_choice == distributed.
    worker_id: int, id of this worker replica. Used in multiproblem setting with
      hparams.problem_choice == distributed.
    batch_size: int, if provided, will use a fixed batch size.
    dataset_split: tf.estimator.ModeKeys + ["test"], which split of the dataset
      to use. Defaults to mode.
    shard: int, if provided, will only read data from the specified shard.

  Returns:
    A function that returns a dictionary of features and the target labels.
  """

  def input_fn():
    """Supplies input to our model.

    This function supplies input to our model, where this input is a
    function of the mode. For example, we supply different data if
    we're performing training versus evaluation.

    Returns:
      A tuple consisting of 1) a dictionary of tensors whose keys are
      the feature names, and 2) a tensor of target labels if the mode
      is not INFER (and None, otherwise).

    Raises:
      ValueError: if one of the parameters has an unsupported value.
    """
    problem_count = len(hparams.problems)
    problem_batches = []
    with tf.name_scope("input_fn"):
      for problem_idx in xrange(problem_count):
        if fixed_problem is not None and problem_idx != fixed_problem:
          continue
        problem_instance = hparams.problem_instances[problem_idx]
        p_hparams = hparams.problems[problem_idx]
        feature_map = features_for_problem(
            problem_instance,
            p_hparams,
            hparams,
            data_dir,
            num_datashards,
            mode,
            batch_size=batch_size,
            dataset_split=dataset_split,
            shard=shard,
            name="problem_%d" % problem_idx)
        problem_batches.append(feature_map)

    # We choose which problem to process.
    loss_moving_avgs = []  # Need loss moving averages for that.
    for problem_idx in xrange(problem_count):
      with tf.variable_scope("losses_avg"):
        loss_moving_avgs.append(
            tf.get_variable(
                "problem_%d/total_loss" % problem_idx,
                initializer=100.0,
                trainable=False))
    if fixed_problem is None:
      problem_choice = _problem_choice(hparams.problem_choice, mode,
                                       problem_count, loss_moving_avgs,
                                       worker_replicas, worker_id)

      # Problem conditional on problem_choice.
      feature_map = cond_on_index(
          lambda problem_idx: problem_batches[problem_idx], problem_choice,
          problem_count - 1)
    else:
      problem_choice = tf.constant(fixed_problem)
      # Take the only constructed batch, which is the fixed_problem.
      feature_map = problem_batches[0]

    feature_map["problem_choice"] = problem_choice

    # Set shapes so the ranks are clear.
    if problem_instance.has_inputs:
      feature_map["inputs"].set_shape([None, None, None, None])
      feature_map["input_space_id"].set_shape([])
    feature_map["targets"].set_shape([None, None, None, None])
    feature_map["problem_choice"].set_shape([])
    feature_map["target_space_id"].set_shape([])

    if mode == tf.estimator.ModeKeys.PREDICT:
      feature_map["infer_targets"] = feature_map["targets"]
      #  Forced shape obfuscation is necessary for inference.
      if problem_instance.has_inputs:
        feature_map["inputs"]._shape = tf.TensorShape([None, None, None, None])  # pylint: disable=protected-access
      feature_map["targets"]._shape = tf.TensorShape([None, None, None, None])  # pylint: disable=protected-access

      # This is because of a bug in the Estimator that short-circuits prediction
      # if it doesn't see a QueueRunner.  DummyQueueRunner implements the
      # minimal expected interface but does nothing.
      tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, DummyQueueRunner())
      return feature_map, None

    return feature_map, feature_map["targets"]

  return input_fn


def _problem_choice(choice_mode, mode, problem_count, loss_moving_avgs,
                    worker_replicas, worker_id):
  """Return idx of problem based on choice_mode and mode."""
  if choice_mode == "uniform" or mode != tf.estimator.ModeKeys.TRAIN:
    problem_choice = tf.random_uniform([], maxval=problem_count, dtype=tf.int32)
  elif choice_mode == "adaptive":
    loss_moving_avgs = tf.stack(loss_moving_avgs)
    problem_choice = tf.multinomial(tf.reshape(loss_moving_avgs, [1, -1]), 1)
    problem_choice = tf.to_int32(tf.squeeze(problem_choice))
  elif choice_mode == "distributed":
    assert worker_replicas >= problem_count
    assert worker_replicas % problem_count == 0
    problem_choice = tf.to_int32(worker_id % problem_count)
  else:
    raise ValueError("Value of hparams.problem_choice is %s and must be "
                     "one of [uniform, adaptive, distributed]" % choice_mode)

  return problem_choice


def cond_on_index(fn, index_tensor, max_idx, cur_idx=0):
  """Call fn(index_tensor) using tf.cond in [cur_id, max_idx]."""
  if cur_idx == max_idx:
    return fn(cur_idx)

  return tf.cond(
      tf.equal(index_tensor, cur_idx),
      lambda: fn(cur_idx),
      lambda: cond_on_index(fn, index_tensor, max_idx, cur_idx + 1)
  )


class DummyQueueRunner(object):
  """Can stand-in for a QueueRunner but does nothing."""

  def __init__(self):
    pass

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    del sess, coord, daemon, start
    return []


def features_for_problem(problem_instance,
                         p_hparams,
                         hparams,
                         data_dir,
                         num_datashards,
                         mode,
                         batch_size=None,
                         dataset_split=None,
                         shard=None,
                         name="problem_inputs"):
  """Feature map for Problem."""
  with tf.name_scope(name):
    with tf.device("/cpu:0"):  # Input reading on CPU
      capacity = (p_hparams.max_expected_batch_size_per_shard * num_datashards)
      batching_scheme = data_reader.hparams_to_batching_scheme(
          hparams,
          shard_multiplier=num_datashards,
          drop_long_sequences=(mode == tf.estimator.ModeKeys.TRAIN or
                               hparams.eval_drop_long_sequences),
          length_multiplier=(p_hparams.batch_size_multiplier))
      if batch_size:
        # If batch_size is fixed, use a single input bucket
        batching_scheme["batch_sizes"] = [batch_size]
        batching_scheme["boundaries"] = []
        # Log new batching scheme if updated
        tf.logging.info("Updated batching_scheme = %s", batching_scheme)
      feature_map = data_reader.input_pipeline(
          problem_instance,
          data_dir,
          capacity,
          mode,
          hparams,
          batching_scheme,
          dataset_split=dataset_split,
          shard=shard)

  # Ensure inputs and targets are proper rank.
  if problem_instance.has_inputs:
    while len(feature_map["inputs"].get_shape()) != 4:
      feature_map["inputs"] = tf.expand_dims(feature_map["inputs"], axis=-1)
  while len(feature_map["targets"].get_shape()) != 4:
    feature_map["targets"] = tf.expand_dims(feature_map["targets"], axis=-1)

  if problem_instance.has_inputs:
    feature_map["input_space_id"] = tf.constant(p_hparams.input_space_id)
  feature_map["target_space_id"] = tf.constant(p_hparams.target_space_id)
  return feature_map
