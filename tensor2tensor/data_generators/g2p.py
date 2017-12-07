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

"""Data generators for g2p (grapheme to phoneme) task."""

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
EOS_ID = text_encoder.EOS_ID
CMUDICT_URL = "https://sourceforge.net/projects/cmusphinx/files/G2P%20Models/phonetisaurus-cmudict-split.tar.gz/download"


def _build_vocab(filename, vocab_path):
  """Reads a file to build a vocabulary with letters and phonemes.

  Args:
    filename: file to read list of words from.
    vocab_path: path where to save the vocabulary.
  """
  vocab = set()
  with tf.gfile.GFile(filename, "r") as f:
    for line in f:
        items = line.split()
        vocab.update(items[0])
        vocab.update(items[1:])
  with open(vocab_path, "w") as f:
    f.write("<pad>\n")
    f.write("<EOS>\n")
    f.write("\n".join(sorted(vocab)))


def _get_token_encoder(vocab_dir, vocab_name, filename):
  """Reads from file and returns a `TokenTextEncoder` for the vocabulary."""
  vocab_path = os.path.join(vocab_dir, vocab_name)
  if not tf.gfile.Exists(vocab_path):
    _build_vocab(filename, vocab_path)
  return text_encoder.TokenTextEncoder(vocab_path)

@registry.register_problem
class G2p(problem.Text2TextProblem):
  """A class for generating G2P data."""

  @property
  def has_inputs(self):
    return True

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def num_shards(self):
    return 1

  @property
  def vocab_name(self):
    return "vocab.g2p"

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def targeted_vocab_size(self):
    return 10000

  @property
  def is_character_level(self):
    return False

  def generator(self, data_dir, tmp_dir, train):
    filename = os.path.basename(CMUDICT_URL)
    compressed_filepath = generator_utils.maybe_download(
        tmp_dir, filename, CMUDICT_URL)

    with tarfile.open(compressed_filepath, "r:gz") as tgz:
      files = [m.name for m in tgz.getmembers()]
      tgz.extractall(tmp_dir)

    train_file, valid_file = None, None
    for filename in files:
      if "train" in filename:
        train_file = os.path.join(tmp_dir, filename)
      elif "test" in filename:
        valid_file = os.path.join(tmp_dir, filename)

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"

    encoder = _get_token_encoder(data_dir, self.vocab_file, train_file)

    if train:
      return self._generator(train_file, encoder)
    return self._generator(valid_file, encoder)

  def _generator(self, filename, encoder):
    with tf.gfile.GFile(filename, "r") as f:
      for line in f:
        items = line.split()
        inputs_string = " ".join(items[0])
        targets_string = " ".join(items[1:])
        inputs = encoder.encode(inputs_string) + [EOS_ID]
        targets = encoder.encode(targets_string) + [EOS_ID]
        yield {"inputs": inputs, "targets": targets}
