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

"""BLEU metric util used during eval for MT."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import re
import sys
import time
import unicodedata

# Dependency imports

import numpy as np
import six
# pylint: disable=redefined-builtin
from six.moves import xrange
from six.moves import zip
# pylint: enable=redefined-builtin

import tensorflow as tf


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in xrange(1, max_order + 1):
    for i in xrange(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  precisions = []

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)

    overlap = dict((ngram,
                    min(count, translation_ngram_counts[ngram]))
                   for ngram, count in ref_ngram_counts.items())

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]
  precisions = [0] * max_order
  smooth = 1.0
  for i in xrange(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum/max_order)

  if use_bp:
    ratio = translation_length / reference_length
    bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
  bleu = geo_mean * bp
  return np.float32(bleu)


def bleu_score(predictions, labels, **unused_kwargs):
  """BLEU score computation between labels and predictions.

  An approximate BLEU scoring method since we do not glue word pieces or
  decode the ids and tokenize the output. By default, we use ngram order of 4
  and use brevity penalty. Also, this does not have beam search.

  Args:
    predictions: tensor, model predicitons
    labels: tensor, gold output.

  Returns:
    bleu: int, approx bleu score
  """
  outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
  # Convert the outputs and labels to a [batch_size, input_length] tensor.
  outputs = tf.squeeze(outputs, axis=[-1, -2])
  labels = tf.squeeze(labels, axis=[-1, -2])

  bleu = tf.py_func(compute_bleu, (labels, outputs), tf.float32)
  return bleu, tf.constant(1.0)


class UnicodeRegex(object):
  """Ad-hoc hack to recognize all punctuation and symbols."""

  def __init__(self):
    punctuation = self.property_chars("P")
    self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
    self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
    self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

  def property_chars(self, prefix):
    return "".join(six.unichr(x) for x in range(sys.maxunicode)
                   if unicodedata.category(six.unichr(x)).startswith(prefix))


def bleu_tokenize(string):
  r"""Tokenize a string following the official BLEU implementation.

  See https://github.com/moses-smt/mosesdecoder/"
           "blob/master/scripts/generic/mteval-v14.pl#L954-L983
  In our case, the input string is expected to be just one line
  and no HTML entities de-escaping is needed.
  So we just tokenize on punctuation and symbols,
  except when a punctuation is preceded and followed by a digit
  (e.g. a comma/dot as a thousand/decimal separator).

  Note that a numer (e.g. a year) followed by a dot at the end of sentence
  is NOT tokenized,
  i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
  does not match this case (unless we add a space after each sentence).
  However, this error is already in the original mteval-v14.pl
  and we want to be consistent with it.

  Args:
    string: the input string

  Returns:
    a list of tokens
  """
  uregex = UnicodeRegex()
  string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
  string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
  string = uregex.symbol_re.sub(r" \1 ", string)
  return string.split()


def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
  """Compute BLEU for two files (reference and hypothesis translation)."""
  ref_lines = open(ref_filename).read().splitlines()
  hyp_lines = open(hyp_filename).read().splitlines()
  assert len(ref_lines) == len(hyp_lines)
  if not case_sensitive:
    ref_lines = [x.lower() for x in ref_lines]
    hyp_lines = [x.lower() for x in hyp_lines]
  ref_tokens = [bleu_tokenize(x) for x in ref_lines]
  hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
  return compute_bleu(ref_tokens, hyp_tokens)


StepFile = collections.namedtuple("StepFile", "filename mtime ctime steps")


def _read_stepfiles_list(path_prefix, path_suffix=".index", min_steps=0):
  """Return list of StepFiles sorted by step from files at path_prefix."""
  stepfiles = []
  for filename in tf.gfile.Glob(path_prefix + "*-[0-9]*" + path_suffix):
    basename = filename[:-len(path_suffix)] if len(path_suffix) else filename
    try:
      steps = int(basename.rsplit("-")[-1])
    except ValueError:  # The -[0-9]* part is not an integer.
      continue
    if steps < min_steps:
      continue
    if not os.path.exists(filename):
      tf.logging.info(filename + " was deleted, so skipping it")
      continue
    stepfiles.append(StepFile(basename, os.path.getmtime(filename),
                              os.path.getctime(filename), steps))
  return sorted(stepfiles, key=lambda x: -x.steps)


def stepfiles_iterator(path_prefix, wait_minutes=0, min_steps=0,
                       path_suffix=".index", sleep_sec=10):
  """Continuously yield new files with steps in filename as they appear.

  This is useful for checkpoint files or other files whose names differ just in
  an integer marking the number of steps and match the wildcard path_prefix +
  "*-[0-9]*" + path_suffix.

  Unlike `tf.contrib.training.checkpoints_iterator`, this implementation always
  starts from the oldest files (and it cannot miss any file). Note that the
  oldest checkpoint may be deleted anytime by Tensorflow (if set up so). It is
  up to the user to check that the files returned by this generator actually
  exist.

  Args:
    path_prefix: The directory + possible common filename prefix to the files.
    wait_minutes: The maximum amount of minutes to wait between files.
    min_steps: Skip files with lower global step.
    path_suffix: Common filename suffix (after steps), including possible
      extension dot.
    sleep_sec: How often to check for new files.

  Yields:
    named tuples (filename, mtime, ctime, steps) of the files as they arrive.
  """
  # Wildcard D*-[0-9]* does not match D/x-1, so if D is a directory let
  # path_prefix="D/".
  if not path_prefix.endswith(os.sep) and os.path.isdir(path_prefix):
    path_prefix += os.sep
  stepfiles = _read_stepfiles_list(path_prefix, path_suffix, min_steps)
  tf.logging.info("Found %d files with steps: %s",
                  len(stepfiles),
                  ", ".join(str(x.steps) for x in reversed(stepfiles)))
  exit_time = time.time() + wait_minutes * 60
  while True:
    if not stepfiles and wait_minutes:
      tf.logging.info(
          "Waiting till %s if a new file matching %s*-[0-9]*%s appears",
          time.asctime(time.localtime(exit_time)), path_prefix, path_suffix)
      while True:
        stepfiles = _read_stepfiles_list(path_prefix, path_suffix, min_steps)
        if stepfiles or time.time() > exit_time:
          break
        time.sleep(sleep_sec)
    if not stepfiles:
      return

    stepfile = stepfiles.pop()
    exit_time, min_steps = (stepfile.ctime + wait_minutes * 60,
                            stepfile.steps + 1)
    yield stepfile
