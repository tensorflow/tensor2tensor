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
from tensor2tensor.data_generators import text_encoder

import tensorflow as tf


EOS = text_encoder.EOS
PTB_URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"


def _read_words(filename):
  """Reads words from a file."""
  with tf.gfile.GFile(filename, "r") as f:
    if sys.version_info[0] >= 3:
      return f.read().replace("\n", " ").split()
    else:
      return f.read().decode("utf-8").replace("\n", " ").split()


def _build_vocab(filename, vocab_path, vocab_size):
  """Reads a file to build a vocabulary of `vocab_size` most common words.

   The vocabulary is sorted by occurence count and has one word per line.
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


def _get_token_encoder(vocab_dir, filename):
  """Reads from file and returns a `TokenTextEncoder` for the vocabulary."""
  vocab_name = "lmptb_10k.vocab"
  vocab_path = os.path.join(vocab_dir, vocab_name)
  _build_vocab(filename, vocab_path, 10000)
  return text_encoder.TokenTextEncoder(vocab_path)


class PTB(object):
  """A class for generating PTB data."""

  def __init__(self, tmp_dir, data_dir, char=False):
    assert not char, "char mode for PTB is not yet implemented"
    self.char = char
    self.data_dir = data_dir

    url = PTB_URL
    filename = os.path.basename(url)
    compressed_filepath = generator_utils.maybe_download(
        tmp_dir, filename, url)
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

    if self.char:
      files = ptb_char_files
    else:
      files = ptb_files
    files = files

    for filename in files:
      if "train" in filename:
        self.train = os.path.join(tmp_dir, filename)
      elif "valid" in filename:
        self.valid = os.path.join(tmp_dir, filename)

    assert hasattr(self, "train"), "Training file not found"
    assert hasattr(self, "valid"), "Validation file not found"
    self.encoder = _get_token_encoder(data_dir, self.train)

  def train_generator(self):
    return self._generator(self.train)

  def valid_generator(self):
    return self._generator(self.valid)

  def _generator(self, filename):
    with tf.gfile.GFile(filename, "r") as f:
      for line in f:
        line = " ".join(line.replace("\n", EOS).split())
        tok = self.encoder.encode(line)
        yield {"inputs": tok[:-1], "targets": tok[1:]}


# Using a object "singleton"
# `train_generator` must be called before
# `valid_generator` in order to work
_ptb = {}


def train_generator(*args, **kwargs):
  """The train data generator to be called."""
  global _ptb
  _ptb = PTB(*args, **kwargs)
  return _ptb.train_generator()


def valid_generator():
  """Validation (aka. dev) data generator."""
  global _ptb  # pylint:disable=global-variable-not-assigned
  return _ptb.valid_generator()
