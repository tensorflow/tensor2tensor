# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""SARI score for evaluating paraphrasing and other text generation models.

The score is introduced in the following paper:

   Optimizing Statistical Machine Translation for Text Simplification
   Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen and Chris Callison-Burch
   In Transactions of the Association for Computational Linguistics (TACL) 2015
   http://cs.jhu.edu/~napoles/res/tacl2016-optimizing.pdf

This implementation has two differences with the GitHub [1] implementation:
  (1) Define 0/0=1 instead of 0 to give higher scores for predictions that match
      a target exactly.
  (2) Fix an alleged bug [2] in the deletion score computation.

[1] https://github.com/cocoxu/simplification/blob/master/SARI.py
    (commit 0210f15)
[2] https://github.com/cocoxu/simplification/issues/6
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

# The paper that intoduces the SARI score uses only the precision of the deleted
# tokens (i.e. beta=0). To give more emphasis on recall, you may set, e.g.,
# beta=1.
BETA_FOR_SARI_DELETION_F_MEASURE = 0


def _get_ngram_counter(ids, n):
  """Get a Counter with the ngrams of the given ID list.

  Args:
    ids: np.array or a list corresponding to a single sentence
    n: n-gram size

  Returns:
    collections.Counter with ID tuples as keys and 1s as values.
  """
  # Remove zero IDs used to pad the sequence.
  ids = [token_id for token_id in ids if token_id != 0]
  ngram_list = [tuple(ids[i:i + n]) for i in range(len(ids) + 1 - n)]
  ngrams = set(ngram_list)
  counts = collections.Counter()
  for ngram in ngrams:
    counts[ngram] = 1
  return counts


def _get_fbeta_score(true_positives, selected, relevant, beta=1):
  """Compute Fbeta score.

  Args:
    true_positives: Number of true positive ngrams.
    selected: Number of selected ngrams.
    relevant: Number of relevant ngrams.
    beta: 0 gives precision only, 1 gives F1 score, and Inf gives recall only.

  Returns:
    Fbeta score.
  """
  precision = 1
  if selected > 0:
    precision = true_positives / selected
  if beta == 0:
    return precision
  recall = 1
  if relevant > 0:
    recall = true_positives / relevant
  if precision > 0 and recall > 0:
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)
  else:
    return 0


def get_addition_score(source_counts, prediction_counts, target_counts):
  """Compute the addition score (Equation 4 in the paper)."""
  added_to_prediction_counts = prediction_counts - source_counts
  true_positives = sum((added_to_prediction_counts & target_counts).values())
  selected = sum(added_to_prediction_counts.values())
  # Note that in the paper the summation is done over all the ngrams in the
  # output rather than the ngrams in the following set difference. Since the
  # former does not make as much sense we compute the latter, which is also done
  # in the GitHub implementation.
  relevant = sum((target_counts - source_counts).values())
  return _get_fbeta_score(true_positives, selected, relevant)


def get_keep_score(source_counts, prediction_counts, target_counts):
  """Compute the keep score (Equation 5 in the paper)."""
  source_and_prediction_counts = source_counts & prediction_counts
  source_and_target_counts = source_counts & target_counts
  true_positives = sum((source_and_prediction_counts &
                        source_and_target_counts).values())
  selected = sum(source_and_prediction_counts.values())
  relevant = sum(source_and_target_counts.values())
  return _get_fbeta_score(true_positives, selected, relevant)


def get_deletion_score(source_counts, prediction_counts, target_counts, beta=0):
  """Compute the deletion score (Equation 6 in the paper)."""
  source_not_prediction_counts = source_counts - prediction_counts
  source_not_target_counts = source_counts - target_counts
  true_positives = sum((source_not_prediction_counts &
                        source_not_target_counts).values())
  selected = sum(source_not_prediction_counts.values())
  relevant = sum(source_not_target_counts.values())
  return _get_fbeta_score(true_positives, selected, relevant, beta=beta)


def get_sari_score(source_ids, prediction_ids, list_of_targets,
                   max_gram_size=4, beta_for_deletion=0):
  """Compute the SARI score for a single prediction and one or more targets.

  Args:
    source_ids: a list / np.array of SentencePiece IDs
    prediction_ids: a list / np.array of SentencePiece IDs
    list_of_targets: a list of target ID lists / np.arrays
    max_gram_size: int. largest n-gram size we care about (e.g. 3 for unigrams,
        bigrams, and trigrams)
    beta_for_deletion: beta for deletion F score.

  Returns:
    the SARI score and its three components: add, keep, and deletion scores
  """
  addition_scores = []
  keep_scores = []
  deletion_scores = []
  for n in range(1, max_gram_size + 1):
    source_counts = _get_ngram_counter(source_ids, n)
    prediction_counts = _get_ngram_counter(prediction_ids, n)
    # All ngrams in the targets with count 1.
    target_counts = collections.Counter()
    # All ngrams in the targets with count r/num_targets, where r is the number
    # of targets where the ngram occurs.
    weighted_target_counts = collections.Counter()
    num_nonempty_targets = 0
    for target_ids_i in list_of_targets:
      target_counts_i = _get_ngram_counter(target_ids_i, n)
      if target_counts_i:
        weighted_target_counts += target_counts_i
        num_nonempty_targets += 1
    for gram in weighted_target_counts.keys():
      weighted_target_counts[gram] /= num_nonempty_targets
      target_counts[gram] = 1
    keep_scores.append(get_keep_score(source_counts, prediction_counts,
                                      weighted_target_counts))
    deletion_scores.append(get_deletion_score(source_counts, prediction_counts,
                                              weighted_target_counts,
                                              beta_for_deletion))
    addition_scores.append(get_addition_score(source_counts, prediction_counts,
                                              target_counts))

  avg_keep_score = sum(keep_scores) / max_gram_size
  avg_addition_score = sum(addition_scores) / max_gram_size
  avg_deletion_score = sum(deletion_scores) / max_gram_size
  sari = (avg_keep_score + avg_addition_score + avg_deletion_score) / 3.0
  return sari, avg_keep_score, avg_addition_score, avg_deletion_score


def get_sari(source_ids, prediction_ids, target_ids, max_gram_size=4):
  """Computes the SARI scores from the given source, prediction and targets.

  Args:
    source_ids: A 2D tf.Tensor of size (batch_size , sequence_length)
    prediction_ids: A 2D tf.Tensor of size (batch_size, sequence_length)
    target_ids: A 3D tf.Tensor of size (batch_size, number_of_targets,
        sequence_length)
    max_gram_size: int. largest n-gram size we care about (e.g. 3 for unigrams,
        bigrams, and trigrams)

  Returns:
    A 4-tuple of 1D float Tensors of size (batch_size) for the SARI score and
        the keep, addition and deletion scores.
  """

  def get_sari_numpy(source_ids, prediction_ids, target_ids):
    """Iterate over elements in the batch and call the SARI function."""
    sari_scores = []
    keep_scores = []
    add_scores = []
    deletion_scores = []
    # Iterate over elements in the batch.
    for source_ids_i, prediction_ids_i, target_ids_i in zip(
        source_ids, prediction_ids, target_ids):
      sari, keep, add, deletion = get_sari_score(
          source_ids_i, prediction_ids_i, target_ids_i, max_gram_size,
          BETA_FOR_SARI_DELETION_F_MEASURE)
      sari_scores.append(sari)
      keep_scores.append(keep)
      add_scores.append(add)
      deletion_scores.append(deletion)
    return (np.asarray(sari_scores), np.asarray(keep_scores),
            np.asarray(add_scores), np.asarray(deletion_scores))

  sari, keep, add, deletion = tf.py_func(
      get_sari_numpy,
      [source_ids, prediction_ids, target_ids],
      [tf.float64, tf.float64, tf.float64, tf.float64])
  return sari, keep, add, deletion


def sari_score(predictions, labels, features, **unused_kwargs):
  """Computes the SARI scores from the given source, prediction and targets.

  An approximate SARI scoring method since we do not glue word pieces or
  decode the ids and tokenize the output. By default, we use ngram order of 4.
  Also, this does not have beam search.

  Args:
    predictions: tensor, model predictions.
    labels: tensor, gold output.
    features: dict, containing inputs.

  Returns:
    sari: int, approx sari score
  """
  if "inputs" not in features:
    raise ValueError("sari_score requires inputs feature")

  # Convert the inputs and outputs to a [batch_size, sequence_length] tensor.
  inputs = tf.squeeze(features["inputs"], axis=[-1, -2])
  outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
  outputs = tf.squeeze(outputs, axis=[-1, -2])

  # Convert the labels to a [batch_size, 1, sequence_length] tensor.
  labels = tf.squeeze(labels, axis=[-1, -2])
  labels = tf.expand_dims(labels, axis=1)

  score, _, _, _ = get_sari(inputs, outputs, labels)
  return score, tf.constant(1.0)
