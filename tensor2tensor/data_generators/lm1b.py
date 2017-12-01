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

"""Data generators for LM1B data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import os
import tarfile

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import registry

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
  vocab_filename = os.path.basename(vocab_url + ".en")
  vocab_filepath = os.path.join(tmp_dir, vocab_filename)
  if not os.path.exists(vocab_filepath):
    generator_utils.maybe_download(tmp_dir, vocab_filename, vocab_url)
  return set([
      text_encoder.native_to_unicode(l.strip())
      for l in tf.gfile.Open(vocab_filepath)
  ])


def _replace_oov(original_vocab, line):
  """Replace out-of-vocab words with "UNK".

  This maintains compatibility with published results.

  Args:
    original_vocab: a set of strings (The standard vocabulary for the dataset)
    line: a unicode string - a space-delimited sequence of words.

  Returns:
    a unicode string - a space-delimited sequence of words.
  """
  return u" ".join(
      [word if word in original_vocab else u"UNK" for word in line.split()])


def _train_data_filenames(tmp_dir):
  return [
      os.path.join(tmp_dir,
                   "1-billion-word-language-modeling-benchmark-r13output",
                   "training-monolingual.tokenized.shuffled",
                   "news.en-%05d-of-00100" % i) for i in xrange(1, 100)
  ]


def _dev_data_filename(tmp_dir):
  return os.path.join(tmp_dir,
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


def _get_or_build_subword_text_encoder(tmp_dir,
                                       vocab_filepath,
                                       target_size):
  """Builds a SubwordTextEncoder based on the corpus.

  Args:
    tmp_dir: directory containing dataset.
    vocab_filepath: path to store (or load) vocab.
    target_size: an optional integer.

  Returns:
    a SubwordTextEncoder.
  """
  if tf.gfile.Exists(vocab_filepath):
    return text_encoder.SubwordTextEncoder(vocab_filepath)
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
  if target_size == 2 ** 15:
    # legacy behavior
    ret = text_encoder.SubwordTextEncoder()
    ret.build_from_token_counts(token_counts, min_count=5)
  else:
    ret = text_encoder.SubwordTextEncoder.build_to_target_size(
        target_size, token_counts, 1, 1000)
  ret.store_to_file(vocab_filepath)
  return ret


@registry.register_problem
class LanguagemodelLm1b32k(problem.Text2TextProblem):
  """A language model on the 1B words corpus."""

  @property
  def is_character_level(self):
    return False

  @property
  def has_inputs(self):
    return False

  @property
  def input_space_id(self):
    # Ratio of dev tokens (including eos) to dev words (including eos)
    # 176884 / 159658 = 1.107893; multiply ppx by this to compare results.
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def num_shards(self):
    return 100

  @property
  def vocab_name(self):
    return "vocab.lm1b.en"

  @property
  def use_subword_tokenizer(self):
    return True

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def use_train_shards_for_dev(self):
    return False

  def generator(self, data_dir, tmp_dir, is_training):
    """Generator for lm1b sentences.

    Args:
      data_dir: data dir.
      tmp_dir: tmp dir.
      is_training: a boolean.

    Yields:
      A dictionary {"inputs": [0], "targets": [<subword ids>]}
    """
    _maybe_download_corpus(tmp_dir)
    original_vocab = _original_vocab(tmp_dir)
    files = (_train_data_filenames(tmp_dir)
             if is_training else [_dev_data_filename(tmp_dir)])
    if self.is_character_level:
      encoder = text_encoder.ByteTextEncoder()
    else:
      vocab_filepath = os.path.join(data_dir, self.vocab_file)
      encoder = _get_or_build_subword_text_encoder(
          tmp_dir, vocab_filepath, self.targeted_vocab_size)
    for filepath in files:
      tf.logging.info("filepath = %s", filepath)
      for line in tf.gfile.Open(filepath):
        tokens = encoder.encode(
            _replace_oov(original_vocab, text_encoder.native_to_unicode(line)))
        tokens.append(EOS)
        yield {"inputs": [0], "targets": tokens}


@registry.register_problem
class LanguagemodelLm1b8kPacked(LanguagemodelLm1b32k):
  """A language model on the 1B words corpus.

  8k vocabualry.
  Training/eval examples are concatenated to a maximum length of 256.

  Happy TPU Training.

  Ratio of dev tokens (including eos) to dev words (including eos)
  207351 / 159658 = 1.29872; multiply ppx by this to compare results.
  """

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def packed_length(self):
    return 256


@registry.register_problem
class LanguagemodelLm1bCharacters(LanguagemodelLm1b32k):
  """A language model on the 1B words corpus, character level."""

  @property
  def is_character_level(self):
    return True
