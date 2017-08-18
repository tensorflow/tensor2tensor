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
                   data_file_patterns=None,
                   num_datashards=None,
                   fixed_problem=None,
                   worker_replicas=None,
                   worker_id=None):
  """Provides input to the graph, either from disk or via a placeholder.

  This function produces an input function that will feed data into
  the network. There are two modes of operation:

  1. If data_file_pattern and all subsequent arguments are None, then
     it creates a placeholder for a serialized tf.Example proto.
  2. If data_file_pattern is defined, it will read the data from the
     files at the given location. Use this mode for training,
     evaluation, and testing prediction.

  Args:
    mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.
    hparams: HParams object.
    data_file_patterns: The list of file patterns to use to read in data. Set to
      `None` if you want to create a placeholder for the input data. The
      `problems` flag is a list of problem names joined by the `-` character.
      The flag's string is then split along the `-` and each problem gets its
      own example queue.
    num_datashards: An integer.
    fixed_problem: An integer indicating the problem to fetch data for, or None
      if the input is to be randomly selected.
    worker_replicas: int, number of worker replicas. Used in multiproblem
      setting with hparams.problem_choice == distributed.
    worker_id: int, id of this worker replica. Used in multiproblem setting with
      hparams.problem_choice == distributed.

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
    problem_count, batches = len(hparams.problems), []
    with tf.name_scope("input_reader"):
      for n in xrange(problem_count):
        if fixed_problem is not None and n != fixed_problem:
          continue
        problem_instance = hparams.problem_instances[n]
        p_hparams = hparams.problems[n]
        with tf.name_scope("problem_%d" % n):
          with tf.device("/cpu:0"):  # Input reading on CPU
            capacity = (
                p_hparams.max_expected_batch_size_per_shard * num_datashards)
            feature_map = data_reader.input_pipeline(
                problem_instance, data_file_patterns and data_file_patterns[n],
                capacity, mode, hparams,
                data_reader.hparams_to_batching_scheme(
                    hparams,
                    shard_multiplier=num_datashards,
                    drop_long_sequences=(mode == tf.contrib.learn.ModeKeys.TRAIN
                                         or hparams.eval_drop_long_sequences),
                    length_multiplier=(p_hparams.batch_size_multiplier)))

        # Reverse inputs and targets features if the problem was reversed.
        if problem_instance is not None:
          problem_instance.maybe_reverse_features(feature_map)
          problem_instance.maybe_copy_features(feature_map)
        else:
          if p_hparams.was_reversed:
            inputs = feature_map["inputs"]
            targets = feature_map["targets"]
            feature_map["inputs"] = targets
            feature_map["targets"] = inputs
          # Use the inputs as the targets if the problem is a copy problem.
          if p_hparams.was_copy:
            feature_map["targets"] = feature_map["inputs"]

        # Ensure inputs and targets are proper rank.
        while len(feature_map["inputs"].get_shape()) != 4:
          feature_map["inputs"] = tf.expand_dims(feature_map["inputs"], axis=-1)
        while len(feature_map["targets"].get_shape()) != 4:
          feature_map["targets"] = tf.expand_dims(
              feature_map["targets"], axis=-1)

        batches.append((feature_map["inputs"], feature_map["targets"],
                        tf.constant(n), tf.constant(p_hparams.input_space_id),
                        tf.constant(p_hparams.target_space_id)))

    # We choose which problem to process.
    loss_moving_avgs = []  # Need loss moving averages for that.
    for n in xrange(problem_count):
      with tf.variable_scope("losses_avg"):
        loss_moving_avgs.append(
            tf.get_variable(
                "problem_%d/total_loss" % n, initializer=100.0,
                trainable=False))
    if fixed_problem is None:
      if (hparams.problem_choice == "uniform" or
          mode != tf.contrib.learn.ModeKeys.TRAIN):
        problem_choice = tf.random_uniform(
            [], maxval=problem_count, dtype=tf.int32)
      elif hparams.problem_choice == "adaptive":
        loss_moving_avgs = tf.stack(loss_moving_avgs)
        problem_choice = tf.multinomial(
            tf.reshape(loss_moving_avgs, [1, -1]), 1)
        problem_choice = tf.to_int32(tf.squeeze(problem_choice))
      elif hparams.problem_choice == "distributed":
        assert worker_replicas >= problem_count
        assert worker_replicas % problem_count == 0
        problem_choice = tf.to_int32(worker_id % problem_count)
      else:
        raise ValueError(
            "Value of hparams.problem_choice is %s and must be "
            "one of [uniform, adaptive, distributed]" % hparams.problem_choice)

      # Inputs and targets conditional on problem_choice.
      rand_inputs, rand_target, choice, inp_id, tgt_id = cond_on_index(
          lambda n: batches[n], problem_choice, 0, problem_count - 1)
    else:
      problem_choice = tf.constant(fixed_problem)
      # Take the only constructed batch, which is the fixed_problem.
      rand_inputs, rand_target, choice, inp_id, tgt_id = batches[0]

    # Set shapes so the ranks are clear.
    rand_inputs.set_shape([None, None, None, None])
    rand_target.set_shape([None, None, None, None])
    choice.set_shape([])
    inp_id.set_shape([])
    tgt_id.set_shape([])
    #  Forced shape obfuscation is necessary for inference.
    if mode == tf.contrib.learn.ModeKeys.INFER:
      rand_inputs._shape = tf.TensorShape([None, None, None, None])  # pylint: disable=protected-access
      rand_target._shape = tf.TensorShape([None, None, None, None])  # pylint: disable=protected-access

    # Final feature map.
    rand_feature_map = {
        "inputs": rand_inputs,
        "problem_choice": choice,
        "input_space_id": inp_id,
        "target_space_id": tgt_id
    }
    if mode == tf.contrib.learn.ModeKeys.INFER:
      rand_feature_map["infer_targets"] = rand_target
      rand_target = None
    return rand_feature_map, rand_target

  return input_fn


def cond_on_index(fn, index_tensor, cur_idx, max_idx):
  """Call fn(index_tensor) using tf.cond in [cur_id, max_idx]."""
  if cur_idx == max_idx:
    return fn(cur_idx)
  return tf.cond(
      tf.equal(index_tensor, cur_idx), lambda: fn(cur_idx),
      lambda: cond_on_index(fn, index_tensor, cur_idx + 1, max_idx))
