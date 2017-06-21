# Copyright 2017 Google Inc.
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

import functools

# Dependency imports

import six

from tensor2tensor.models import common_layers
from tensor2tensor.utils import bleu_hook

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def padded_accuracy_topk(predictions,
                         labels,
                         k,
                         weights_fn=common_layers.weights_nonzero):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy_topk", values=[predictions, labels]):
    padded_labels = common_layers.pad_with_zeros(predictions, labels)
    weights = weights_fn(padded_labels)
    effective_k = tf.minimum(k, tf.shape(predictions)[-1])
    _, outputs = tf.nn.top_k(predictions, k=effective_k)
    outputs = tf.to_int32(outputs)
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
    padded_labels = common_layers.pad_with_zeros(predictions, labels)
    weights = weights_fn(padded_labels)
    outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
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
    padded_labels = common_layers.pad_with_zeros(predictions, labels)
    weights = weights_fn(padded_labels)
    outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
    return tf.to_float(tf.equal(outputs, padded_labels)), weights


def create_evaluation_metrics(problems):
  """Creates the evaluation metrics for the model.

  Args:
    problems: List of strings containing the name of the problems.

  Returns:
    A dictionary with keys that are strings naming the evaluation
    metrics and values that are functions taking arguments of
    (predictions, targets), returning a tuple of a tensor of the
    metric's value together with an op to update the metric's value.
  """

  def append_metric_fns(metric_tup, eval_metrics):
    """Append problem-specific and global metrics to eval_metrics."""
    metric_name, metric_function = metric_tup
    def fn(predictions, labels, weights, idx, weights_fn):
      # The 'weights' argument represents problem-choice here,
      # we need to keep this name because MetricSpecs checks it.
      problem_choice = weights
      (scores, weights) = tf.cond(
          tf.equal(idx, problem_choice),  # pylint: disable=cell-var-from-loop
          lambda: metric_function(predictions, labels, weights_fn=weights_fn),
          lambda: (tf.constant(0.0), tf.constant(0.0)))
      # The tf.metrics.mean function assures correct aggregation.
      return tf.metrics.mean(scores, weights)

    for i, problem in enumerate(problems):
      name = "metrics-%s/%s" % (problem, metric_name)
      weights_fn = (common_layers.weights_concatenated
                    if "concat" in problem else common_layers.weights_nonzero)
      eval_metrics[name] = functools.partial(fn, idx=i, weights_fn=weights_fn)

    def global_fn(predictions, labels, weights):
      (scores, weights) = metric_function(predictions, labels)
      return tf.metrics.mean(scores, weights)

    eval_metrics["metrics/%s" % metric_name] = global_fn

  eval_metrics = dict()

  # Metrics are functions that take predictions and labels and return
  # a tensor of metrics and a tensor of weights.
  # The results are passed to tf.metrics.mean to accumulate properly.
  metrics_list = [("accuracy", padded_accuracy), ("accuracy_top5",
                                                  padded_accuracy_top5),
                  ("accuracy_per_sequence", padded_sequence_accuracy),
                  ("neg_log_perplexity", padded_neg_log_perplexity)]

  # TODO(nikip): Extend this to support use of custom metrics for problems.
  for problem in problems:
    if "wmt" in problem:
      metrics_list.append(("bleu_score", bleu_hook.padded_bleu_score))

  for metric in metrics_list:
    append_metric_fns(metric, eval_metrics)

  return {
      k: tf.contrib.learn.MetricSpec(
          v, prediction_key="predictions", weight_key="problem_choice")
      for (k, v) in six.iteritems(eval_metrics)
  }
