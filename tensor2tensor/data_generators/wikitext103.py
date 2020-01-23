# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Data generators for wikitext-103.

Wikitext-103: Long term dependency language modeling dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


def _build_vocab(filename, vocab_dir, vocab_name):
  """Reads a file to build a vocabulary.

  Args:
    filename: file to read list of words from.
    vocab_dir: directory where to save the vocabulary.
    vocab_name: vocab file name.

  Returns:
    text encoder.
  """
  vocab_path = os.path.join(vocab_dir, vocab_name)
  if not tf.gfile.Exists(vocab_path):
    with tf.gfile.GFile(filename, "r") as f:
      data = f.read().split()
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    encoder = text_encoder.TokenTextEncoder(None, vocab_list=words)
    encoder.store_to_file(vocab_path)
  else:
    encoder = text_encoder.TokenTextEncoder(vocab_path)
  return encoder


def _maybe_download_corpus(tmp_dir, vocab_type):
  """Download and unpack the corpus.

  Args:
    tmp_dir: directory containing dataset.
    vocab_type: which vocabulary are we using.

  Returns:
    The list of names of files.
  """
  if vocab_type == text_problems.VocabType.CHARACTER:

    dataset_url = ("https://s3.amazonaws.com/research.metamind.io/wikitext"
                   "/wikitext-103-raw-v1.zip")
    dir_name = "wikitext-103-raw"
  else:
    dataset_url = ("https://s3.amazonaws.com/research.metamind.io/wikitext"
                   "/wikitext-103-v1.zip")
    dir_name = "wikitext-103"

  fname = os.path.basename(dataset_url)
  compressed_filepath = generator_utils.maybe_download(tmp_dir, fname,
                                                       dataset_url)
  zip_ref = zipfile.ZipFile(compressed_filepath, "r")
  zip_ref.extractall(tmp_dir)
  zip_ref.close()

  files = os.path.join(tmp_dir, dir_name, "*")
  train_file, valid_file, test_file = None, None, None
  for f in tf.gfile.Glob(files):
    fname = os.path.basename(f)
    if "train" in fname:
      train_file = f
    elif "valid" in fname:
      valid_file = f
    elif "test" in fname:
      test_file = f

  assert train_file, "Training file not found"
  assert valid_file, "Validation file not found"
  assert test_file, "Testing file not found"

  return train_file, valid_file, test_file


@registry.register_problem
class LanguagemodelWikitext103(text_problems.Text2SelfProblem):
  """Wikitext103 dataset token-level."""

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  @property
  def is_generate_per_split(self):
    return True

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    train_file, valid_file, test_file = _maybe_download_corpus(
        tmp_dir, self.vocab_type)

    if dataset_split == problem.DatasetSplit.TRAIN:
      filepath = train_file
      if self.vocab_type == text_problems.VocabType.TOKEN:
        _build_vocab(train_file, data_dir, self.vocab_filename)

    elif dataset_split == problem.DatasetSplit.EVAL:
      filepath = valid_file

    elif dataset_split == problem.DatasetSplit.TEST:
      filepath = test_file

    def _generate_samples():
      with tf.gfile.GFile(filepath, "r") as f:
        for line in f:
          line = " ".join(line.strip().split())
          if line:
            yield {"targets": line}

    return _generate_samples()


@registry.register_problem
class LanguagemodelWikitext103Characters(LanguagemodelWikitext103):
  """Wikitext-103, character-level."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER


@registry.register_problem
class LanguagemodelWikitext103L4k(LanguagemodelWikitext103):
  """Wikitext-103, token-level, with examples up to 4,096 tokens long."""

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    samples_by_line = super(LanguagemodelWikitext103L4k,
                            self).generate_samples(data_dir, tmp_dir,
                                                   dataset_split)

    def _generate_samples():
      tokens = []
      for sample in samples_by_line:
        sample_tokens = sample["targets"].split()
        if len(tokens) + len(sample_tokens) < self.sequence_length:
          tokens.extend(sample_tokens)
        else:
          yield {"targets": " ".join(tokens)}
          tokens = sample_tokens

    return _generate_samples()

  def max_length(self, model_hparams):
    return model_hparams.split_to_length or self.sequence_length

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 4096


@registry.register_problem
class LanguagemodelWikitext103L16k(LanguagemodelWikitext103L4k):
  """Wikitext-103, token-level, with examples up to 16,384 tokens long."""

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 16384
