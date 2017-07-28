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

"""Utils for metrics used in eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six

from tensor2tensor.models import common_layers
from tensor2tensor.utils import bleu_hook

import tensorflow as tf


class Metrics(object):
  """Available evaluation metrics."""
  # Entries here should match the keys in METRICS_FN below
  ACC = "accuracy"
  ACC_TOP5 = "accuracy_top5"
  ACC_PER_SEQ = "accuracy_per_sequence"
  NEG_LOG_PERPLEXITY = "neg_log_perplexity"
  APPROX_BLEU = "approx_bleu_score"
  RMSE = "rmse"


def padded_rmse(predictions, labels, weights_fn=common_layers.weights_nonzero):
  predictions, labels = common_layers.pad_with_zeros(predictions, labels)
  targets = labels
  weights = weights_fn(targets)
  error = tf.sqrt(tf.pow(predictions - labels, 2))
  return tf.reduce_sum(error * weights), tf.reduce_sum(weights)


def padded_accuracy_topk(predictions,
                         labels,
                         k,
                         weights_fn=common_layers.weights_nonzero):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy_topk", values=[predictions, labels]):
    padded_predictions, padded_labels = common_layers.pad_with_zeros(
        predictions, labels)
    weights = weights_fn(padded_labels)
    effective_k = tf.minimum(k, tf.shape(padded_predictions)[-1])
    _, outputs = tf.nn.top_k(padded_predictions, k=effective_k)
    outputs = tf.to_int32(outputs)
    padded_labels = tf.to_int32(padded_labels)
    padded_labels = tf.expand_dims(padded_labels, axis=-1)
    padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
    same = tf.to_float(tf.equal(outputs, padded_labels))
    same_topk = tf.reduce_sum(same, axis=-1)
    return same_topk, weights


def padded_accuracy_top5(predictions,
                         labels,
                         weights_fn=common_layers.weights_nonzero):
  return padded_accuracy_topk(predictions, labels, 5, weights_fn)


def padded_sequence_accuracy(predictions,
                             labels,
                             weights_fn=common_layers.weights_nonzero):
  """Percentage of times that predictions matches labels everywhere (non-0)."""
  with tf.variable_scope(
      "padded_sequence_accuracy", values=[predictions, labels]):
    padded_predictions, padded_labels = common_layers.pad_with_zeros(
        predictions, labels)
    weights = weights_fn(padded_labels)
    outputs = tf.to_int32(tf.argmax(padded_predictions, axis=-1))
    padded_labels = tf.to_int32(padded_labels)
    not_correct = tf.to_float(tf.not_equal(outputs, padded_labels)) * weights
    axis = list(range(1, len(outputs.get_shape())))
    correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
    return correct_seq, tf.constant(1.0)


def padded_neg_log_perplexity(predictions,
                              labels,
                              weights_fn=common_layers.weights_nonzero):
  """Average log-perplexity exluding padding 0s. No smoothing."""
  num, den = common_layers.padded_cross_entropy(
      predictions, labels, 0.0, weights_fn=weights_fn, reduce_sum=False)
  return (-num, den)


def padded_accuracy(predictions,
                    labels,
                    weights_fn=common_layers.weights_nonzero):
  """Percentage of times that predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy", values=[predictions, labels]):
    padded_predictions, padded_labels = common_layers.pad_with_zeros(
        predictions, labels)
    weights = weights_fn(padded_labels)
    outputs = tf.to_int32(tf.argmax(padded_predictions, axis=-1))
    padded_labels = tf.to_int32(padded_labels)
    return tf.to_float(tf.equal(outputs, padded_labels)), weights


def create_evaluation_metrics(problems):
  """Creates the evaluation metrics for the model.

  Args:
    problems: List of tuples (problem name, problem instance).

  Returns:
    A dictionary with keys that are strings naming the evaluation
    metrics and values that are functions taking arguments of
    (predictions, targets), returning a tuple of a tensor of the
    metric's value together with an op to update the metric's value.

  Raises:
    ValueError: if the metrics specified by a problem are not recognized (i.e.
      are not defined in the Metrics enum.
  """

  def make_problem_specific_metric_fn(metric_fn, problem_idx, weights_fn):
    """Create a metric fn conditioned on problem_idx."""

    def problem_metric_fn(predictions, labels, weights):
      problem_choice = weights
      (scores, weights) = tf.cond(
          tf.equal(problem_idx, problem_choice),
          lambda: metric_fn(predictions, labels, weights_fn=weights_fn),
          lambda: (tf.constant(0.0), tf.constant(0.0)))
      # The tf.metrics.mean function assures correct aggregation.
      return tf.metrics.mean(scores, weights)

    return problem_metric_fn

  eval_metrics = dict()
  for problem_idx, (problem_name, problem_instance) in enumerate(problems):
    if problem_instance is None:
      # For problems in problem_hparams
      metrics = [
          Metrics.ACC, Metrics.ACC_TOP5, Metrics.ACC_PER_SEQ,
          Metrics.NEG_LOG_PERPLEXITY
      ]
      if "wmt" in problem_name:
        metrics.append(Metrics.APPROX_BLEU)
    else:
      # For registered Problems
      metrics = problem_instance.eval_metrics()
      if not all([m in METRICS_FNS for m in metrics]):
        raise ValueError("Unrecognized metric. Problem %s specified metrics "
                         "%s. Recognized metrics are %s." %
                         (problem_name, metrics, METRICS_FNS.keys()))

    class_output = "image" in problem_name and "coco" not in problem_name
    weights_fn = (common_layers.weights_all
                  if class_output else common_layers.weights_nonzero)

    for metric in metrics:
      metric_fn = METRICS_FNS[metric]
      problem_metric_fn = make_problem_specific_metric_fn(
          metric_fn, problem_idx, weights_fn)
      eval_metrics["metrics-%s/%s" % (problem_name, metric)] = problem_metric_fn

  return {
      k: tf.contrib.learn.MetricSpec(
          v, prediction_key="predictions", weight_key="problem_choice")
      for (k, v) in six.iteritems(eval_metrics)
  }


# Metrics are functions that take predictions and labels and return
# a tensor of metrics and a tensor of weights.
# The results are passed to tf.metrics.mean to accumulate properly.
METRICS_FNS = {
    Metrics.ACC: padded_accuracy,
    Metrics.ACC_TOP5: padded_accuracy_top5,
    Metrics.ACC_PER_SEQ: padded_sequence_accuracy,
    Metrics.NEG_LOG_PERPLEXITY: padded_neg_log_perplexity,
    Metrics.APPROX_BLEU: bleu_hook.bleu_score,
    Metrics.RMSE: padded_rmse,
}
