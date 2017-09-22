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

"""Data generators for PTB data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf


EOS = text_encoder.EOS
PTB_URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"


def _read_words(filename):
  """Reads words from a file."""
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", " %s " % EOS).split()
    else:
      return f.read().decode("utf-8").replace("\n", " %s " % EOS).split()


def _build_vocab(filename, vocab_path, vocab_size):
  """Reads a file to build a vocabulary of `vocab_size` most common words.

   The vocabulary is sorted by occurrence count and has one word per line.
   Originally from:
   https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py

  Args:
    filename: file to read list of words from.
    vocab_path: path where to save the vocabulary.
    vocab_size: size of the vocablulary to generate.
  """
  data = _read_words(filename)
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  words, _ = list(zip(*count_pairs))
  words = words[:vocab_size]
  with open(vocab_path, "w") as f:
    f.write("\n".join(words))


def _get_token_encoder(vocab_dir, vocab_name, filename):
  """Reads from file and returns a `TokenTextEncoder` for the vocabulary."""
  vocab_path = os.path.join(vocab_dir, vocab_name)
  if not tf.gfile.Exists(vocab_path):
    _build_vocab(filename, vocab_path, 10000)
  return text_encoder.TokenTextEncoder(vocab_path)


class PTBProblem(problem.Text2TextProblem):
  """A class for generating PTB data."""

  @property
  def has_inputs(self):
    return False

  @property
  def target_space_id(self):
    if self.is_character_level:
      return problem.SpaceID.EN_CHR
    return problem.SpaceID.EN_TOK

  @property
  def num_shards(self):
    return 10

  @property
  def vocab_name(self):
    return "vocab.lmptb_10k"

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def targeted_vocab_size(self):
    return 10000

  def generator(self, data_dir, tmp_dir, train):
    filename = os.path.basename(PTB_URL)
    compressed_filepath = generator_utils.maybe_download(
        tmp_dir, filename, PTB_URL)
    ptb_files = []
    ptb_char_files = []
    with tarfile.open(compressed_filepath, "r:gz") as tgz:
      files = []
      # Selecting only relevant files.
      for m in tgz.getmembers():
        if "ptb" in m.name and ".txt" in m.name:
          if "char" in m.name:
            ptb_char_files += [m.name]
          else:
            ptb_files += [m.name]
          files += [m]

      tgz.extractall(tmp_dir, members=files)

    if self.is_character_level:
      files = ptb_char_files
    else:
      files = ptb_files

    train_file, valid_file = None, None
    for filename in files:
      if "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"

    if self.is_character_level:
      encoder = text_encoder.ByteTextEncoder()
    else:
      encoder = _get_token_encoder(data_dir, self.vocab_file, train_file)

    if train:
      return self._generator(train_file, encoder)
    return self._generator(valid_file, encoder)

  def _generator(self, filename, encoder):
    with tf.gfile.GFile(filename, "r") as f:
      for line in f:
        line = " ".join(line.replace("\n", " %s " % EOS).split())
        tok = encoder.encode(line)
        if tok:
          yield {"inputs": [0], "targets": tok}


@registry.register_problem
class LanguagemodelPtb10k(PTBProblem):
  """A class for generating PTB data, 10k vocab."""

  @property
  def is_character_level(self):
    return False


@registry.register_problem
class LanguagemodelPtbCharacters(PTBProblem):
  """A class for generating PTB data, character-level."""

  @property
  def is_character_level(self):
    return True
