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

"""Data generator for Wikipedia title to article dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class LanguagemodelWikiXmlV8kL1k(problem.ChoppedTextProblem):
  """A language model on English Wikipedia.

  XML dump is chopped arbitrarily into sequences of length 1024 tokens,
  without regard to article boundaries.
  """

  def maybe_prepare_text(self, tmp_dir):
    """Download corpus if necessary, decompress, split into multiple text files.

    Args:
      tmp_dir: directory containing dataset.

    Returns:
      list of filenames for local text files.
    """
    compressed_filename = os.path.basename(self.corpus_url)
    compressed_filepath = os.path.join(tmp_dir, compressed_filename)
    decompressed_filepath = compressed_filepath[:-4]
    split_file_prefix = decompressed_filepath + "-part-"
    split_filepattern = split_file_prefix + "?????"
    split_files = sorted(tf.gfile.Glob(split_filepattern))
    if not split_files:
      if not tf.gfile.Exists(decompressed_filepath):
        if not tf.gfile.Exists(compressed_filepath):
          generator_utils.maybe_download(
              tmp_dir, compressed_filepath, self.corpus_url)
        assert not subprocess.call(["bunzip2", compressed_filepath])
      assert tf.gfile.Exists(decompressed_filepath)
      assert not subprocess.call([
          "split", "--line-bytes=4M", "--suffix-length=5",
          "--numeric-suffixes", decompressed_filepath, split_file_prefix])
      split_files = sorted(tf.gfile.Glob(split_filepattern))
    assert split_files
    return split_files

  def train_text_filenames(self, tmp_dir):
    all_files = self.maybe_prepare_text(tmp_dir)
    return [f for i, f in enumerate(all_files) if i % self.dev_fraction != 0]

  def dev_text_filenames(self, tmp_dir):
    all_files = self.maybe_prepare_text(tmp_dir)
    return [f for i, f in enumerate(all_files) if i % self.dev_fraction == 0]

  @property
  def dev_fraction(self):
    return 5000

  @property
  def corpus_url(self):
    return ("https://archive.org/download/enwiki-20171201/"
            "enwiki-20171201-pages-articles.xml.bz2")

  @property
  def vocab_name(self):
    return "vocab.wiki_xml"

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 1024


class LanguagemodelWikiScramble(LanguagemodelWikiXmlV8kL1k):
  """Language modeling on English wikipedia.

  "targets" is a sequence of sequence_length tokens - a fragment of an article.
  "inputs" is a copy of "targets", but with a random scramble_fraction of the
    tokens randomly permuted.

  This dataset is intended to test parallel (non-autoregressive) prediction
  of the target sequence given the input sequence.
  """

  def example_generator(self, encoder, tmp_dir, task_id):
    for x in super(LanguagemodelWikiScramble, self).example_generator(
        encoder, tmp_dir, task_id):
      x["inputs"] = self.scramble(self["targets"])
      yield x

  @property
  def scramble_fraction(self):
    raise NotImplementedError()

  @property
  def has_inputs(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  def scramble(self, seq):
    seq = np.array(seq)
    num_permute = int(self.sequence_length * self.scramble_fraction)
    full_permutation = np.random.permutation(self.sequence_length)
    inverse_full_permutation = np.argsort(full_permutation)
    partial_permutation = np.random.permutation(num_permute)
    seq = seq[full_permutation]
    seq = np.concatenate(
        (seq[:num_permute][partial_permutation], seq[num_permute:]))
    seq = seq[inverse_full_permutation]
    seq = list(seq)
    return seq


@registry.register_problem
class LanguagemodelWikiScrambleL128(LanguagemodelWikiScramble):
  """Sequence length 128, 50% scrambed."""

  @property
  def sequence_length(self):
    return 128

  @property
  def scramble_fraction(self):
    return 0.5


@registry.register_problem
class LanguagemodelWikiScrambleL1k(LanguagemodelWikiScramble):
  """Sequence length 1024, 50% scrambed."""

  @property
  def sequence_length(self):
    return 1024

  @property
  def scramble_fraction(self):
    return 0.5
