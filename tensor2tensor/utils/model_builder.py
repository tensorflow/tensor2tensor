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

"""Model building."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

# Dependency imports

import numpy as np
import six
# pylint: disable=redefined-builtin
from six.moves import xrange
# pylint: enable=redefined-builtin

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor.utils import devices
from tensor2tensor.utils import input_fn_builder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import optimize
from tensor2tensor.utils import registry

import tensorflow as tf
from tensorflow.python.framework import dtypes


def model_fn(model,
             features,
             mode,
             hparams,
             problem_names,
             train_steps=100000,
             worker_id=0,
             worker_replicas=1,
             eval_run_autoregressive=False,
             decode_hparams=None):
  """Builds the model for all modes.

  * TRAIN: Constructs loss and train_op
  * EVAL: Constructs the loss and eval metrics
  * PREDICT: Constructs the predictions

  Args:
    model: str, name of model.
    features: dict<feature name, Tensor>. Expected to have keys
      {inputs, targets, problem_choice}.
    mode: tf.estimator.ModeKeys.
    hparams: model HParams.
    problem_names: list of str, names of the problems.
    train_steps: int, total number of training steps. Used to compute learning
      rate decay.
    worker_id: int, id of this worker.
    worker_replicas: int, number of workers.
    eval_run_autoregressive: bool, whether to run evaluation autoregressively.
    decode_hparams: HParams for decode settings. Used when mode == PREDICT.

  Returns:
    tf.estimator.EstimatorSpec
  """
  assert len(problem_names) == len(hparams.problem_instances)
  decode_hp = decode_hparams

  # TODO(rsepassi): This still depends on FLAGS. Rm eventually.
  dp = devices.data_parallelism()

  tf.get_variable_scope().set_initializer(_get_variable_initializer(hparams))
  is_training = mode == tf.estimator.ModeKeys.TRAIN

  # Add input statistics for incoming features.
  with tf.name_scope("input_stats"):
    for (k, v) in six.iteritems(features):
      if isinstance(v, tf.Tensor) and v.get_shape().ndims > 1:
        tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // dp.n)
        tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
        nonpadding = tf.to_float(tf.not_equal(v, 0))
        nonpadding_tokens = tf.reduce_sum(nonpadding)
        if k == "targets":
          targets_nonpadding_tokens = nonpadding_tokens
        tf.summary.scalar("%s_nonpadding_tokens" % k, nonpadding_tokens)
        tf.summary.scalar("%s_nonpadding_fraction" % k,
                          tf.reduce_mean(nonpadding))

  # Get multi-problem logits and loss based on features["problem_choice"].
  loss_variable_names = []

  def nth_model(n):
    """Build the model for the n-th problem, plus some added variables."""
    model_class = registry.model(model)(
        hparams,
        mode,
        hparams.problems[n],
        n,
        dp,
        devices.ps_devices(all_workers=True),
        decode_hparams=decode_hparams)
    if mode == tf.estimator.ModeKeys.PREDICT:
      return model_class.infer(
          features,
          beam_size=decode_hp.beam_size,
          top_beams=(decode_hp.beam_size if decode_hp.return_beams else 1),
          alpha=decode_hp.alpha,
          decode_length=decode_hp.extra_length)
    # In distributed mode, we build graph for problem=0 and problem=worker_id.
    skipping_is_on = hparams.problem_choice == "distributed" and is_training
    problem_worker_id = worker_id % len(hparams.problems)
    skip_this_one = n != 0 and n % worker_replicas != problem_worker_id
    # On worker 0 also build graph for problems <= 1.
    # TODO(lukaszkaiser): why is this hack needed for variables init? Repair.
    skip_this_one = skip_this_one and (worker_id != 0 or n > 1)
    if eval_run_autoregressive and mode == tf.estimator.ModeKeys.EVAL:
      sharded_logits, losses_dict = model_class.eval_autoregressive(features)
    else:
      sharded_logits, losses_dict = model_class.model_fn(
          features, skip=(skipping_is_on and skip_this_one))
    with tf.variable_scope("losses_avg"):
      total_loss, ops = 0.0, []
      for loss_key, loss_value in six.iteritems(losses_dict):
        loss_name = "problem_%d/%s_loss" % (n, loss_key)
        loss_moving_avg = tf.get_variable(
            loss_name, initializer=100.0, trainable=False)
        loss_variable_names.append(loss_name)
        ops.append(
            loss_moving_avg.assign(loss_moving_avg * 0.9 + loss_value * 0.1))
        total_loss += loss_value
      try:  # Total loss avg might be reused or not, we try both.
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          # Total loss was already constructed on input.
          loss_moving_avg = tf.get_variable("problem_%d/total_loss" % n)
      except ValueError:
        loss_moving_avg = tf.get_variable(
            "problem_%d/total_loss" % n, initializer=100.0, trainable=False)
      ops.append(
          loss_moving_avg.assign(loss_moving_avg * 0.9 + total_loss * 0.1))
    with tf.variable_scope("train_stats"):  # Count steps for this problem.
      problem_steps = tf.get_variable(
          "problem_%d_steps" % n, initializer=0, trainable=False)
      ops.append(problem_steps.assign_add(1))
    with tf.control_dependencies(ops):  # Make sure the ops run.
      # Ensure the loss is a scalar here.
      total_loss = tf.reshape(total_loss, [], name="total_loss_control_id")
    return [total_loss, tf.concat(sharded_logits, 0)]

  model_output = input_fn_builder.cond_on_index(
      nth_model,
      index_tensor=features["problem_choice"],
      max_idx=len(hparams.problems) - 1)

  if mode == tf.estimator.ModeKeys.PREDICT:
    # If beam searching, model_output will be a dict with keys "outputs" and
    # "scores".
    if isinstance(model_output, dict):
      outputs = model_output["outputs"]
      scores = model_output["scores"]
    else:
      outputs = model_output
      scores = None

    batched_problem_choice = (
        features["problem_choice"] * tf.ones(
            (tf.shape(features["inputs"])[0],), dtype=tf.int32))
    predictions = {
        "outputs": outputs,
        "scores": scores,
        "inputs": features.get("inputs", None),
        "targets": features.get("infer_targets", None),
        "problem_choice": batched_problem_choice,
    }
    _del_dict_nones(predictions)

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        export_outputs={
            "output": tf.estimator.export.PredictOutput(export_out)
        })

  total_loss, logits = model_output

  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metrics_fns = metrics.create_evaluation_metrics(
        hparams.problem_instances, hparams)

    eval_metrics = {}
    for metric_name, metric_fn in six.iteritems(eval_metrics_fns):
      eval_metrics[metric_name] = metric_fn(logits, features)

    return tf.estimator.EstimatorSpec(
        mode,
        predictions={"predictions": logits},
        eval_metric_ops=eval_metrics,
        loss=total_loss)

  assert mode == tf.estimator.ModeKeys.TRAIN

  # Set learning rate
  learning_rate = hparams.learning_rate * optimize.learning_rate_decay(
      hparams, num_worker_replicas=worker_replicas, num_train_steps=train_steps)
  learning_rate /= math.sqrt(float(worker_replicas))

  # Get global step
  global_step = tf.train.get_or_create_global_step()

  # Some training statistics.
  with tf.name_scope("training_stats"):
    tf.summary.scalar("learning_rate", learning_rate)
    for n in xrange(len(hparams.problems)):
      names_and_vars = []
      with tf.variable_scope("losses_avg", reuse=True):
        total_loss_var = tf.get_variable("problem_%d/total_loss" % n)
        names_and_vars.append(("total_loss", total_loss_var))
      with tf.variable_scope("losses_avg", reuse=True):
        for loss_name in loss_variable_names:
          if loss_name.startswith("problem_%d/" % n):
            loss_var = tf.get_variable(loss_name)
            loss_suffix = loss_name[loss_name.index("/") + 1:]
            names_and_vars.append((loss_suffix, loss_var))
      for (loss_name, loss_var) in names_and_vars:
        tf.summary.scalar("loss_avg_%d/%s" % (n, loss_name), loss_var)
      with tf.variable_scope("train_stats", reuse=True):
        nth_steps = tf.get_variable("problem_%d_steps" % n, dtype=tf.int32)
      tf.summary.scalar("problem_%d_frequency" % n,
                        tf.to_float(nth_steps) /
                        (tf.to_float(global_step) + 1.0))

  # Add weight decay and noise.
  total_size, weight_decay_loss = 0, 0.0
  all_weights = {v.name: v for v in tf.trainable_variables()}
  for v_name in sorted(list(all_weights)):
    v = all_weights[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    total_size += v_size
    if hparams.weight_decay > 0.0 and len(v.shape.as_list()) > 1:
      # Add weight regularization if set and the weight is not a bias (dim>1).
      with tf.device(v._ref().device):  # pylint: disable=protected-access
        v_loss = tf.nn.l2_loss(v) / v_size
      weight_decay_loss += v_loss
    is_body = len(v_name) > 5 and v_name[:5] == "body/"
    if hparams.weight_noise > 0.0 and is_body:
      # Add weight noise if set in hparams.
      with tf.device(v._ref().device):  # pylint: disable=protected-access
        scale = learning_rate * 0.001
        noise = tf.truncated_normal(v.shape) * hparams.weight_noise * scale
        noise_op = v.assign_add(noise)
      with tf.control_dependencies([noise_op]):
        total_loss = tf.identity(total_loss)
  if hparams.weight_decay > 0.0:
    total_loss += weight_decay_loss * hparams.weight_decay

  # The new data reader occasionally emits very small batches, which
  # cause the examples in those batches to be grossly overweighted.
  # We decrease the loss proportionally to the ratio of the size of this
  # batch to the size of the largest training batch ever.
  # TODO(noam): to be more sophisticated, we could keep separate
  # maxima based on problem choice.
  max_nonpadding_var = tf.get_variable(
      "max_nonpadding",
      shape=[],
      initializer=tf.ones_initializer(),
      trainable=False)
  max_nonpadding = tf.maximum(max_nonpadding_var, targets_nonpadding_tokens)
  with tf.control_dependencies([tf.assign(max_nonpadding_var, max_nonpadding)]):
    small_batch_multiplier = targets_nonpadding_tokens / max_nonpadding
  tf.summary.scalar("small_batch_multiplier", small_batch_multiplier)
  total_loss *= small_batch_multiplier

  # Log variable sizes
  _log_variable_sizes(tf.trainable_variables(), "Trainable Variables")
  diet_vars = [
      v for v in tf.global_variables() if v.dtype == dtypes.float16_ref
  ]
  _log_variable_sizes(diet_vars, "Diet Variables")

  # Optimize
  train_op = optimize.optimize(total_loss, learning_rate, hparams)

  # Remove summaries that will fail to run because they are in conditionals.
  # TODO(cwhipkey): Test with this code removed, later in 2017.
  summaries = tf.get_collection_ref(tf.GraphKeys.SUMMARIES)
  for i in reversed(range(len(summaries))):
    if summaries[i].name.startswith("cond_"):
      del summaries[i]

  tf.logging.info("Global model_fn finished.")
  return tf.estimator.EstimatorSpec(
      mode,
      predictions={"problem_choice": features["problem_choice"]},
      loss=total_loss,
      train_op=train_op)


def build_model_fn(model, **kwargs):
  """Returns a function to build the model. See model_fn."""

  # Model function as expected by Estimator
  def wrapping_model_fn(features, labels, mode, params):
    # Deep-copy the model hparams between modes to eliminate
    # side-effects caused by abuse of the linked problem_hparams
    # objects which are used to share modality objects between
    # problems.  We do not want to share the modality objects between
    # modes, since the modality objects may decide to do something
    # mode-specific.  A better fix would be to stop abusing the
    # hparams in this way and instead use a separate dictionary to
    # share the modality objects between problems.  This dictionary
    # could be created once per mode and passed to the constructor of
    # t2t_model.
    hparams = copy.deepcopy(params)
    del params

    if labels is not None:
      features["targets"] = labels
    del labels

    return model_fn(model, features, mode, hparams, **kwargs)

  return wrapping_model_fn


def _log_variable_sizes(var_list, tag):
  """Log the sizes and shapes of variables, and the total size.

  Args:
    var_list: a list of varaibles
    tag: a string
  """
  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
                    v.name[:-2].ljust(80),
                    str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)


def _get_variable_initializer(hparams):
  if hparams.initializer == "orthogonal":
    return tf.orthogonal_initializer(gain=hparams.initializer_gain)
  elif hparams.initializer == "uniform":
    max_val = 0.1 * hparams.initializer_gain
    return tf.random_uniform_initializer(-max_val, max_val)
  elif hparams.initializer == "normal_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="normal")
  elif hparams.initializer == "uniform_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="uniform")
  else:
    raise ValueError("Unrecognized initializer: %s" % hparams.initializer)


def _del_dict_nones(d):
  for k in list(d.keys()):
    if d[k] is None:
      del d[k]
