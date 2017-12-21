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

import inspect

# Dependency imports

import numpy as np

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import bleu_hook
from tensor2tensor.utils import registry
from tensor2tensor.utils import rouge

import tensorflow as tf

from tensorflow.contrib.eager.python import tfe


class Metrics(object):
  """Available evaluation metrics."""
  # Entries here should match the keys in METRICS_FN below
  ACC = "accuracy"
  ACC_TOP5 = "accuracy_top5"
  ACC_PER_SEQ = "accuracy_per_sequence"
  NEG_LOG_PERPLEXITY = "neg_log_perplexity"
  APPROX_BLEU = "approx_bleu_score"
  RMSE = "rmse"
  LOG_POISSON = "log_poisson"
  R2 = "r_squared"
  ROUGE_2_F = "rouge_2_fscore"
  ROUGE_L_F = "rouge_L_fscore"
  EDIT_DISTANCE = "edit_distance"
  SET_PRECISION = "set_precision"
  SET_RECALL = "set_recall"
  IMAGE_SUMMARY = "image_summary"


def padded_rmse(predictions, labels, weights_fn=common_layers.weights_all):
  predictions, labels = common_layers.pad_with_zeros(predictions, labels)
  targets = labels
  weights = weights_fn(targets)
  error = tf.sqrt(tf.pow(predictions - labels, 2))
  return tf.reduce_sum(error * weights), tf.reduce_sum(weights)


def padded_log_poisson(predictions,
                       labels,
                       weights_fn=common_layers.weights_all):
  # Expects predictions to already be transformed into log space
  predictions, labels = common_layers.pad_with_zeros(predictions, labels)
  targets = labels
  weights = weights_fn(targets)

  lp_loss = tf.nn.log_poisson_loss(targets, predictions, compute_full_loss=True)
  return tf.reduce_sum(lp_loss * weights), tf.reduce_sum(weights)


def padded_variance_explained(predictions,
                              labels,
                              weights_fn=common_layers.weights_all):
  # aka R^2
  predictions, labels = common_layers.pad_with_zeros(predictions, labels)
  targets = labels
  weights = weights_fn(targets)

  y_bar = tf.reduce_mean(weights * targets)
  tot_ss = tf.reduce_sum(weights * tf.pow(targets - y_bar, 2))
  res_ss = tf.reduce_sum(weights * tf.pow(targets - predictions, 2))
  r2 = 1. - res_ss / tot_ss
  return r2, tf.reduce_sum(weights)


def padded_accuracy_topk(predictions,
                         labels,
                         k,
                         weights_fn=common_layers.weights_nonzero):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy_topk", values=[predictions, labels]):
    padded_predictions, padded_labels = common_layers.pad_with_zeros(
        predictions, labels)
    weights = weights_fn(padded_labels)
    effective_k = tf.minimum(k,
                             common_layers.shape_list(padded_predictions)[-1])
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


def sequence_edit_distance(predictions,
                           labels,
                           weights_fn=common_layers.weights_nonzero):
  """Average edit distance, ignoring padding 0s.

  The score returned is the edit distance divided by the total length of
  reference truth and the weight returned is the total length of the truth.

  Args:
    predictions: Tensor of shape [`batch_size`, `length`, 1, `num_classes`] and
        type tf.float32 representing the logits, 0-padded.
    labels: Tensor of shape [`batch_size`, `length`, 1, 1] and type tf.int32
        representing the labels of same length as logits and 0-padded.
    weights_fn: ignored. The weights returned are the total length of the ground
        truth labels, excluding 0-paddings.

  Returns:
    (edit distance / reference length, reference length)

  Raises:
    ValueError: if weights_fn is not common_layers.weights_nonzero.
  """
  if weights_fn is not common_layers.weights_nonzero:
    raise ValueError("Only weights_nonzero can be used for this metric.")

  with tf.variable_scope("edit_distance", values=[predictions, labels]):
    # Transform logits into sequence classes by taking max at every step.
    predictions = tf.to_int32(
        tf.squeeze(tf.argmax(predictions, axis=-1), axis=(2, 3)))
    nonzero_idx = tf.where(tf.not_equal(predictions, 0))
    sparse_outputs = tf.SparseTensor(nonzero_idx,
                                     tf.gather_nd(predictions, nonzero_idx),
                                     tf.shape(predictions, out_type=tf.int64))
    labels = tf.squeeze(labels, axis=(2, 3))
    nonzero_idx = tf.where(tf.not_equal(labels, 0))
    label_sparse_outputs = tf.SparseTensor(nonzero_idx,
                                           tf.gather_nd(labels, nonzero_idx),
                                           tf.shape(labels, out_type=tf.int64))
    distance = tf.reduce_sum(
        tf.edit_distance(sparse_outputs, label_sparse_outputs, normalize=False))
    reference_length = tf.to_float(common_layers.shape_list(nonzero_idx)[0])
    return distance / reference_length, reference_length


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


def set_precision(predictions, labels,
                  weights_fn=common_layers.weights_nonzero):
  """Precision of set predictions.

  Args:
    predictions : A Tensor of scores of shape [batch, nlabels].
    labels: A Tensor of int32s giving true set elements,
      of shape [batch, seq_length].
    weights_fn: A function to weight the elements.

  Returns:
    hits: A Tensor of shape [batch, nlabels].
    weights: A Tensor of shape [batch, nlabels].
  """
  with tf.variable_scope("set_precision", values=[predictions, labels]):
    labels = tf.squeeze(labels, [2, 3])
    weights = weights_fn(labels)
    labels = tf.one_hot(labels, predictions.shape[-1])
    labels = tf.reduce_max(labels, axis=1)
    labels = tf.cast(labels, tf.bool)
    return tf.to_float(tf.equal(labels, predictions)), weights


def set_recall(predictions, labels, weights_fn=common_layers.weights_nonzero):
  """Recall of set predictions.

  Args:
    predictions : A Tensor of scores of shape [batch, nlabels].
    labels: A Tensor of int32s giving true set elements,
      of shape [batch, seq_length].
    weights_fn: A function to weight the elements.

  Returns:
    hits: A Tensor of shape [batch, nlabels].
    weights: A Tensor of shape [batch, nlabels].
  """
  with tf.variable_scope("set_recall", values=[predictions, labels]):
    labels = tf.squeeze(labels, [2, 3])
    weights = weights_fn(labels)
    labels = tf.one_hot(labels, predictions.shape[-1])
    labels = tf.reduce_max(labels, axis=1)
    labels = tf.cast(labels, tf.bool)
    return tf.to_float(tf.equal(labels, predictions)), weights


def image_summary(predictions, hparams):
  """Reshapes predictions and passes it to tensorboard.

  Args:
    predictions : A Tensor of scores of shape [batch, nlabels].
    hparams: model_hparams

  Returns:
    summary_proto: containing the summary image for predictions
    weights: A Tensor of zeros of shape [batch, nlabels].
  """
  predictions_reshaped = tf.reshape(
      predictions, [-1, hparams.height, hparams.width, hparams.colors])
  return tf.summary.image(
      "image_summary", predictions_reshaped,
      max_outputs=1), tf.zeros_like(predictions)


def create_evaluation_metrics(problems, model_hparams):
  """Creates the evaluation metrics for the model.

  Args:
    problems: List of Problem instances.
    model_hparams: a set of hparams.

  Returns:
    dict<metric name, metric function>. The metric functions have signature
    (Tensor predictions, features) -> (metric Tensor, update op), where features
    is a dict with keys {targets, problem_choice}.

  Raises:
    ValueError: if the metrics specified by a problem are not recognized (i.e.
      are not defined in the Metrics enum.
  """

  def make_problem_specific_metric_fn(metric_fn, problem_idx, weights_fn):
    """Create a metric fn conditioned on problem_idx."""

    def problem_metric_fn(predictions, features):
      """Metric fn."""
      labels = features.get("targets", None)
      problem_choice = features.get("problem_choice", 0)

      # Send along the entire features dict if the metric fn has the kwarg
      # "features".
      kwargs = {}
      args, _, keywords, _ = inspect.getargspec(metric_fn)
      if ("features" in args) or keywords:
        kwargs["features"] = features

      def wrapped_metric_fn():
        return metric_fn(predictions, labels, weights_fn=weights_fn, **kwargs)

      (scores, weights) = tf.cond(
          tf.equal(problem_idx, problem_choice), wrapped_metric_fn,
          lambda: (tf.constant(0.0), tf.constant(0.0)))
      # The tf.metrics.mean function assures correct aggregation.
      return tf.metrics.mean(scores, weights)

    return problem_metric_fn

  eval_metrics = dict()
  for problem_idx, problem_instance in enumerate(problems):
    problem_name = problem_instance.name
    metrics = problem_instance.eval_metrics()
    if not all([m in METRICS_FNS for m in metrics]):
      error_str = ("Unrecognized metric. Problem %s specified metrics "
                   "%s. Recognized metrics are %s.")
      raise ValueError(error_str % (problem_name,
                                    metrics,
                                    list(METRICS_FNS.keys())))

    def image_wrapped_metric_fn(predictions,
                                labels,
                                weights_fn=common_layers.weights_nonzero):
      _, _ = labels, weights_fn
      return metric_fn(predictions, model_hparams)

    tm = problem_instance.get_hparams().target_modality
    if isinstance(tm, tuple):
      tm = registry.create_modality(tm, model_hparams)
    weights_fn = tm.targets_weights_fn

    for metric in metrics:
      metric_fn = METRICS_FNS[metric]
      metric_name = "metrics-%s/%s" % (problem_name, metric)
      if metric == Metrics.IMAGE_SUMMARY:
        eval_metrics[metric_name] = image_wrapped_metric_fn
      else:
        problem_metric_fn = make_problem_specific_metric_fn(
            metric_fn, problem_idx, weights_fn)
        eval_metrics[metric_name] = problem_metric_fn

  return eval_metrics


def create_eager_metrics_for_problem(problem, model_hparams=None):
  """See create_eager_metrics."""
  metric_names = problem.eval_metrics()
  tm = problem.get_hparams().target_modality
  if isinstance(tm, tuple):
    assert model_hparams is not None
    tm = registry.create_modality(tm, model_hparams)
  return create_eager_metrics(metric_names, weights_fn=tm.targets_weights_fn)


def create_eager_metrics(metric_names, weights_fn=common_layers.weights_all):
  """Create metrics accumulators and averager for Eager mode.

  Args:
    metric_names: list<str> from Metrics enum
    weights_fn: function that takes labels and returns a weights mask. Defaults
      to weights of all 1, i.e. common_layers.weights_all. Use
      common_layers.weights_nonzero if labels have 0-padding.

  Returns:
    (accum_fn(predictions, targets) => None,
     result_fn() => dict<str metric_name, float avg_val>
  """
  metric_fns = dict(
      [(name, METRICS_FNS[name]) for name in metric_names])
  tfe_metrics = dict()

  for name in metric_names:
    tfe_metrics[name] = tfe.metrics.Mean(name=name)

  def metric_accum(predictions, targets):
    for name, metric_fn in metric_fns.items():
      val, weight = metric_fn(predictions, targets,
                              weights_fn=weights_fn)
      tfe_metrics[name](np.squeeze(val), np.squeeze(weight))

  def metric_means():
    avgs = {}
    for name in metric_names:
      avgs[name] = tfe_metrics[name].result().numpy()
    return avgs

  return metric_accum, metric_means


# Metrics are functions that take predictions and labels and return
# a tensor of metrics and a tensor of weights.
# If the function has "features" as an argument, it will receive the whole
# features dict as well.
# The results are passed to tf.metrics.mean to accumulate properly.
METRICS_FNS = {
    Metrics.ACC: padded_accuracy,
    Metrics.ACC_TOP5: padded_accuracy_top5,
    Metrics.ACC_PER_SEQ: padded_sequence_accuracy,
    Metrics.NEG_LOG_PERPLEXITY: padded_neg_log_perplexity,
    Metrics.APPROX_BLEU: bleu_hook.bleu_score,
    Metrics.RMSE: padded_rmse,
    Metrics.LOG_POISSON: padded_log_poisson,
    Metrics.R2: padded_variance_explained,
    Metrics.ROUGE_2_F: rouge.rouge_2_fscore,
    Metrics.ROUGE_L_F: rouge.rouge_l_fscore,
    Metrics.EDIT_DISTANCE: sequence_edit_distance,
    Metrics.SET_PRECISION: set_precision,
    Metrics.SET_RECALL: set_recall,
    Metrics.IMAGE_SUMMARY: image_summary,
}
