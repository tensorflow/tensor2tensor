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

"""Data generators for LM1B data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer

import tensorflow as tf


# End-of-sentence marker (should correspond to the position of EOS in the
# RESERVED_TOKENS list in text_encoder.py)
EOS = 1


def _original_vocab(tmp_dir):
  """Returns a set containing the original vocabulary.

  This is important for comparing with published results.

  Args:
    tmp_dir: directory containing dataset.

  Returns:
    a set of strings
  """
  vocab_url = ("http://download.tensorflow.org/models/LM_LSTM_CNN/"
               "vocab-2016-09-10.txt")
  vocab_filename = os.path.basename(vocab_url)
  vocab_filepath = os.path.join(tmp_dir, vocab_filename)
  if not os.path.exists(vocab_filepath):
    generator_utils.maybe_download(tmp_dir, vocab_filename, vocab_url)
  return set(
      [text_encoder.native_to_unicode(l.strip()) for l in
       tf.gfile.Open(vocab_filepath)])


def _replace_oov(original_vocab, line):
  """Replace out-of-vocab words with "UNK".

  This maintains compatability with published results.

  Args:
    original_vocab: a set of strings (The standard vocabulary for the dataset)
    line: a unicode string - a space-delimited sequence of words.

  Returns:
    a unicode string - a space-delimited sequence of words.
  """
  return u" ".join(
      [word if word in original_vocab else u"UNK" for word in line.split()])


def _train_data_filenames(tmp_dir):
  return [os.path.join(
      tmp_dir,
      "1-billion-word-language-modeling-benchmark-r13output",
      "training-monolingual.tokenized.shuffled",
      "news.en-%05d-of-00100" % i) for i in xrange(1, 100)]


def _dev_data_filename(tmp_dir):
  return os.path.join(
      tmp_dir,
      "1-billion-word-language-modeling-benchmark-r13output",
      "heldout-monolingual.tokenized.shuffled",
      "news.en.heldout-00000-of-00050")


def _maybe_download_corpus(tmp_dir):
  """Download and unpack the corpus.

  Args:
    tmp_dir: directory containing dataset.
  """
  corpus_url = ("http://www.statmt.org/lm-benchmark/"
                "1-billion-word-language-modeling-benchmark-r13output.tar.gz")
  corpus_filename = os.path.basename(corpus_url)
  corpus_filepath = os.path.join(tmp_dir, corpus_filename)
  if not os.path.exists(corpus_filepath):
    generator_utils.maybe_download(tmp_dir, corpus_filename, corpus_url)
    with tarfile.open(corpus_filepath, "r:gz") as corpus_tar:
      corpus_tar.extractall(tmp_dir)


def _get_or_build_subword_text_encoder(tmp_dir):
  """Builds a SubwordTextEncoder based on the corpus.

  Args:
    tmp_dir: directory containing dataset.
  Returns:
    a SubwordTextEncoder.
  """
  filepath = os.path.join(tmp_dir, "lm1b_32k.subword_text_encoder")
  if tf.gfile.Exists(filepath):
    return text_encoder.SubwordTextEncoder(filepath)
  _maybe_download_corpus(tmp_dir)
  original_vocab = _original_vocab(tmp_dir)
  token_counts = defaultdict(int)
  line_count = 0
  max_lines = 63000
  for line in tf.gfile.Open(_train_data_filenames(tmp_dir)[0]):
    tokens = tokenizer.encode(
        _replace_oov(original_vocab, text_encoder.native_to_unicode(line)))
    for tok in tokens:
      token_counts[tok] += 1
    line_count += 1
    if line_count >= max_lines:
      break
  ret = text_encoder.SubwordTextEncoder()
  ret.build_from_token_counts(token_counts, min_count=5)
  ret.store_to_file(filepath)
  return ret


def generator(tmp_dir, train):
  """Generator for lm1b sentences.

  Args:
    tmp_dir: a string.
    train: a boolean.

  Yields:
    A dictionary {"inputs": [0], "targets": [<subword ids>]}
  """
  _maybe_download_corpus(tmp_dir)
  original_vocab = _original_vocab(tmp_dir)
  files = (_train_data_filenames(tmp_dir) if train
           else [_dev_data_filename(tmp_dir)])
  encoder = _get_or_build_subword_text_encoder(tmp_dir)
  for filepath in files:
    tf.logging.info("filepath = %s", filepath)
    for line in tf.gfile.Open(filepath):
      tokens = encoder.encode(
          _replace_oov(original_vocab, text_encoder.native_to_unicode(line)))
      tokens.append(EOS)
      yield {"inputs": [0], "targets": tokens}
