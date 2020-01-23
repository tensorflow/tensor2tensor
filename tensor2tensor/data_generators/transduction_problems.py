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

"""A suite of sequence transduction problems.

Each problem generates pairs of tokenized input and output sequences which
represent the effect of the transduction algorithm which must be learned.

These problems are based on the benchmarks outlined in:

Learning to Transduce with Unbounded Memory
Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, Phil Blunsom
https://arxiv.org/abs/1506.02516, 2015

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


MAX_GENERATOR_ATTEMPTS = 100


class TransductionProblem(text_problems.Text2TextProblem):
  """Abstract base clase which all transduction problems inherit from.
  """

  def __init__(self, was_reversed=False, was_copy=False):
    super(TransductionProblem, self).__init__(was_reversed=False,
                                              was_copy=False)
    self.vocab = self.build_vocab()

  @property
  def num_symbols(self):
    """The number of symbols that can be used as part of a sequence."""
    return 128

  def min_sequence_length(self, dataset_split):
    """Determine the minimum sequence length given a dataset_split.

    Args:
      dataset_split: A problem.DatasetSplit.

    Returns:
      The minimum length that a sequence can be for this dataset_split.
    """
    return {
        problem.DatasetSplit.TRAIN: 8,
        problem.DatasetSplit.EVAL: 65,
        problem.DatasetSplit.TEST: 65
    }[dataset_split]

  def max_sequence_length(self, dataset_split):
    """Determine the maximum sequence length given a dataset_split.

    Args:
      dataset_split: A problem.DatasetSplit.

    Returns:
      The maximum length that a sequence can be for this dataset_split.
    """
    return {
        problem.DatasetSplit.TRAIN: 64,
        problem.DatasetSplit.EVAL: 128,
        problem.DatasetSplit.TEST: 128
    }[dataset_split]

  def num_samples(self, dataset_split):
    """Determine the dataset sized given a dataset_split.

    Args:
      dataset_split: A problem.DatasetSplit.

    Returns:
      The desired number of samples for this dataset_split.
    """
    return {
        problem.DatasetSplit.TRAIN: 1000000,
        problem.DatasetSplit.EVAL: 10000,
        problem.DatasetSplit.TEST: 10000
    }[dataset_split]

  @property
  def num_shards(self):
    """Used to split up datasets into multiple files."""
    return 10

  @property
  def is_generate_per_split(self):
    return False

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  def sequence_length(self, dataset_split):
    return random.randint(self.min_sequence_length(dataset_split),
                          self.max_sequence_length(dataset_split))

  def build_vocab(self):
    return ["sym_%d" % i for i in range(1, self.num_symbols + 1)]

  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    if not tf.gfile.Exists(vocab_filename):
      encoder = text_encoder.TokenTextEncoder(None,
                                              vocab_list=sorted(self.vocab))
      encoder.store_to_file(vocab_filename)
    else:
      encoder = text_encoder.TokenTextEncoder(vocab_filename,
                                              replace_oov=self.oov_token)
    return encoder

  def generate_random_sequence(self, dataset_split):
    return [random.choice(self.vocab)
            for _ in range(self.sequence_length(dataset_split))]

  def transpose_sequence(self, input_sequence):
    raise NotImplementedError()

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    for _ in range(self.num_samples(dataset_split)):
      source = self.generate_random_sequence(dataset_split)
      target = self.transpose_sequence(source)
      yield {
          "inputs": " ".join(source),
          "targets": " ".join(target),
      }


@registry.register_problem
class CopySequence(TransductionProblem):
  """Reproduce a sequence exactly as it was input."""

  def transpose_sequence(self, input_sequence):
    return input_sequence


@registry.register_problem
class CopySequenceSmall(CopySequence):
  """Same as CopySequence but with smaller sequences.
  """

  @property
  def num_symbols(self):
    return 64

  def min_sequence_length(self, dataset_split):
    return {
        problem.DatasetSplit.TRAIN: 4,
        problem.DatasetSplit.EVAL: 17,
        problem.DatasetSplit.TEST: 17
    }[dataset_split]

  def max_sequence_length(self, dataset_split):
    return {
        problem.DatasetSplit.TRAIN: 16,
        problem.DatasetSplit.EVAL: 32,
        problem.DatasetSplit.TEST: 32
    }[dataset_split]

  def num_samples(self, dataset_split):
    return {
        problem.DatasetSplit.TRAIN: 100000,
        problem.DatasetSplit.EVAL: 10000,
        problem.DatasetSplit.TEST: 10000
    }[dataset_split]


@registry.register_problem
class ReverseSequence(TransductionProblem):
  """Reverses the order of the sequence.
  """

  def transpose_sequence(self, input_sequence):
    return input_sequence[::-1]


@registry.register_problem
class ReverseSequenceSmall(ReverseSequence):
  """Same as ReverseSequence but with smaller sequences.
  """

  @property
  def num_symbols(self):
    return 64

  def min_sequence_length(self, dataset_split):
    return {
        problem.DatasetSplit.TRAIN: 4,
        problem.DatasetSplit.EVAL: 17,
        problem.DatasetSplit.TEST: 17
    }[dataset_split]

  def max_sequence_length(self, dataset_split):
    return {
        problem.DatasetSplit.TRAIN: 16,
        problem.DatasetSplit.EVAL: 32,
        problem.DatasetSplit.TEST: 32
    }[dataset_split]

  def num_samples(self, dataset_split):
    return {
        problem.DatasetSplit.TRAIN: 100000,
        problem.DatasetSplit.EVAL: 10000,
        problem.DatasetSplit.TEST: 10000
    }[dataset_split]


@registry.register_problem
class FlipBiGramSequence(TransductionProblem):
  """Flip every pair of tokens: 1 2 3 4 -> 2 1 4 3.
  """

  def sequence_length(self, dataset_split):
    """Only generate sequences with even lengths.

    Args:
      dataset_split: A problem.DatasetSplit specifying which dataset the
        sequence is a part of.

    Returns:
      An even number >= min_sequence_length(dataset_split)
        and <= max_sequence_length(dataset_split)
    """
    min_length = self.min_sequence_length(dataset_split)
    min_length += min_length % 2
    max_length = self.max_sequence_length(dataset_split)
    max_length -= max_length % 2
    length = random.randint(min_length, max_length)
    return length - (length % 2)

  def transpose_sequence(self, input_sequence):
    return [input_sequence[i+1] if i%2 == 0 else input_sequence[i-1]
            for i in range(len(input_sequence))]
