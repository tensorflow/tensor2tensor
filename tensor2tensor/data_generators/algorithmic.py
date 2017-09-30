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

"""Algorithmic data generators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import generator_utils as utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry


class AlgorithmicProblem(problem.Problem):
  """Base class for algorithmic problems."""

  @property
  def num_symbols(self):
    raise NotImplementedError()

  def generator(self, nbr_symbols, max_length, nbr_cases):
    """Generates the data."""
    raise NotImplementedError()

  @property
  def train_length(self):
    return 40

  @property
  def dev_length(self):
    return 400

  @property
  def train_size(self):
    return 100000

  @property
  def dev_size(self):
    return 10000

  @property
  def num_shards(self):
    return 10

  def generate_data(self, data_dir, _, task_id=-1):

    def generator_eos(nbr_symbols, max_length, nbr_cases):
      """Shift by NUM_RESERVED_IDS and append EOS token."""
      for case in self.generator(nbr_symbols, max_length, nbr_cases):
        new_case = {}
        for feature in case:
          new_case[feature] = [
              i + text_encoder.NUM_RESERVED_TOKENS for i in case[feature]
          ] + [text_encoder.EOS_ID]
        yield new_case

    utils.generate_dataset_and_shuffle(
        generator_eos(self.num_symbols, self.train_length, self.train_size),
        self.training_filepaths(data_dir, self.num_shards, shuffled=True),
        generator_eos(self.num_symbols, self.dev_length, self.dev_size),
        self.dev_filepaths(data_dir, 1, shuffled=True),
        shuffle=False)

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    vocab_size = self.num_symbols + text_encoder.NUM_RESERVED_TOKENS
    p.input_modality = {"inputs": (registry.Modalities.SYMBOL, vocab_size)}
    p.target_modality = (registry.Modalities.SYMBOL, vocab_size)
    p.input_space_id = problem.SpaceID.DIGIT_0
    p.target_space_id = problem.SpaceID.DIGIT_1


@registry.register_problem
class AlgorithmicIdentityBinary40(AlgorithmicProblem):
  """Problem spec for algorithmic binary identity task."""

  @property
  def num_symbols(self):
    return 2

  def generator(self, nbr_symbols, max_length, nbr_cases):
    """Generator for the identity (copy) task on sequences of symbols.

    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn uniformly at random from [0, nbr_symbols) until
    nbr_cases sequences have been produced.

    Args:
      nbr_symbols: number of symbols to use in each sequence.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.

    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      input-list and target-list are the same.
    """
    for _ in xrange(nbr_cases):
      l = np.random.randint(max_length) + 1
      inputs = [np.random.randint(nbr_symbols) for _ in xrange(l)]
      yield {"inputs": inputs, "targets": inputs}


@registry.register_problem
class AlgorithmicIdentityDecimal40(AlgorithmicIdentityBinary40):
  """Problem spec for algorithmic decimal identity task."""

  @property
  def num_symbols(self):
    return 10


@registry.register_problem
class AlgorithmicShiftDecimal40(AlgorithmicProblem):
  """Problem spec for algorithmic decimal shift task."""

  @property
  def num_symbols(self):
    return 20

  def generator(self, nbr_symbols, max_length, nbr_cases):
    """Generator for the shift task on sequences of symbols.

    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn uniformly at random from [0, nbr_symbols - shift]
    until nbr_cases sequences have been produced (output[i] = input[i] + shift).

    Args:
      nbr_symbols: number of symbols to use in each sequence (input + output).
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.

    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      target-list[i] = input-list[i] + shift.
    """
    shift = 10
    for _ in xrange(nbr_cases):
      l = np.random.randint(max_length) + 1
      inputs = [np.random.randint(nbr_symbols - shift) for _ in xrange(l)]
      yield {"inputs": inputs, "targets": [i + shift for i in inputs]}

  @property
  def dev_length(self):
    return 80


@registry.register_problem
class AlgorithmicReverseBinary40(AlgorithmicProblem):
  """Problem spec for algorithmic binary reversing task."""

  @property
  def num_symbols(self):
    return 2

  def generator(self, nbr_symbols, max_length, nbr_cases):
    """Generator for the reversing task on sequences of symbols.

    The length of the sequence is drawn uniformly at random from [1, max_length]
    and then symbols are drawn uniformly at random from [0, nbr_symbols) until
    nbr_cases sequences have been produced.

    Args:
      nbr_symbols: number of symbols to use in each sequence.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.

    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      target-list is input-list reversed.
    """
    for _ in xrange(nbr_cases):
      l = np.random.randint(max_length) + 1
      inputs = [np.random.randint(nbr_symbols) for _ in xrange(l)]
      yield {"inputs": inputs, "targets": list(reversed(inputs))}


@registry.register_problem
class AlgorithmicReverseDecimal40(AlgorithmicReverseBinary40):
  """Problem spec for algorithmic decimal reversing task."""

  @property
  def num_symbols(self):
    return 10


def zipf_distribution(nbr_symbols, alpha):
  """Helper function: Create a Zipf distribution.

  Args:
    nbr_symbols: number of symbols to use in the distribution.
    alpha: float, Zipf's Law Distribution parameter. Default = 1.5.
      Usually for modelling natural text distribution is in
      the range [1.1-1.6].

  Returns:
    distr_map: list of float, Zipf's distribution over nbr_symbols.

  """
  tmp = np.power(np.arange(1, nbr_symbols + 1), -alpha)
  zeta = np.r_[0.0, np.cumsum(tmp)]
  return [x / zeta[-1] for x in zeta]


def zipf_random_sample(distr_map, sample_len):
  """Helper function: Generate a random Zipf sample of given length.

  Args:
    distr_map: list of float, Zipf's distribution over nbr_symbols.
    sample_len: integer, length of sequence to generate.

  Returns:
    sample: list of integer, Zipf's random sample over nbr_symbols.

  """
  u = np.random.random(sample_len)
  # Random produces values in range [0.0,1.0); even if it is almost
  # improbable(but possible) that it can generate a clear 0.000..0.
  return list(np.searchsorted(distr_map, u))


def reverse_generator_nlplike(nbr_symbols,
                              max_length,
                              nbr_cases,
                              scale_std_dev=100,
                              alpha=1.5):
  """Generator for the reversing nlp-like task on sequences of symbols.

  The length of the sequence is drawn from a Gaussian(Normal) distribution
  at random from [1, max_length] and with std deviation of 1%,
  then symbols are drawn from Zipf's law at random from [0, nbr_symbols) until
  nbr_cases sequences have been produced.

  Args:
    nbr_symbols: integer, number of symbols.
    max_length: integer, maximum length of sequences to generate.
    nbr_cases: the number of cases to generate.
    scale_std_dev: float, Normal distribution's standard deviation scale factor
      used to draw the length of sequence. Default = 1% of the max_length.
    alpha: float, Zipf's Law Distribution parameter. Default = 1.5.
      Usually for modelling natural text distribution is in
      the range [1.1-1.6].

  Yields:
    A dictionary {"inputs": input-list, "targets": target-list} where
    target-list is input-list reversed.
  """
  std_dev = max_length / scale_std_dev
  distr_map = zipf_distribution(nbr_symbols, alpha)
  for _ in xrange(nbr_cases):
    l = int(abs(np.random.normal(loc=max_length / 2, scale=std_dev)) + 1)
    inputs = zipf_random_sample(distr_map, l)
    yield {"inputs": inputs, "targets": list(reversed(inputs))}


@registry.register_problem
class AlgorithmicReverseNlplike8k(AlgorithmicProblem):
  """Problem spec for algorithmic nlp-like reversing task."""

  @property
  def num_symbols(self):
    return 8000

  def generator(self, nbr_symbols, max_length, nbr_cases):
    return reverse_generator_nlplike(nbr_symbols, max_length, nbr_cases, 10,
                                     1.300)

  @property
  def train_length(self):
    return 70

  @property
  def dev_length(self):
    return 70


@registry.register_problem
class AlgorithmicReverseNlplike32k(AlgorithmicReverseNlplike8k):
  """Problem spec for algorithmic nlp-like reversing task, 32k vocab."""

  @property
  def num_symbols(self):
    return 32000

  def generator(self, nbr_symbols, max_length, nbr_cases):
    return reverse_generator_nlplike(nbr_symbols, max_length, nbr_cases, 10,
                                     1.050)


def lower_endian_to_number(l, base):
  """Helper function: convert a list of digits in the given base to a number."""
  return sum([d * (base**i) for i, d in enumerate(l)])


def number_to_lower_endian(n, base):
  """Helper function: convert a number to a list of digits in the given base."""
  if n < base:
    return [n]
  return [n % base] + number_to_lower_endian(n // base, base)


def random_number_lower_endian(length, base):
  """Helper function: generate a random number as a lower-endian digits list."""
  if length == 1:  # Last digit can be 0 only if length is 1.
    return [np.random.randint(base)]
  prefix = [np.random.randint(base) for _ in xrange(length - 1)]
  return prefix + [np.random.randint(base - 1) + 1]  # Last digit is not 0.


@registry.register_problem
class AlgorithmicAdditionBinary40(AlgorithmicProblem):
  """Problem spec for algorithmic binary addition task."""

  @property
  def num_symbols(self):
    return 2

  def generator(self, base, max_length, nbr_cases):
    """Generator for the addition task.

    The length of each number is drawn uniformly at random in [1, max_length/2]
    and then digits are drawn uniformly at random. The numbers are added and
    separated by [base] in the input. Stops at nbr_cases.

    Args:
      base: in which base are the numbers.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.

    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      input-list are the 2 numbers and target-list is the result of adding them.

    Raises:
      ValueError: if max_length is lower than 3.
    """
    if max_length < 3:
      raise ValueError("Maximum length must be at least 3.")
    for _ in xrange(nbr_cases):
      l1 = np.random.randint(max_length // 2) + 1
      l2 = np.random.randint(max_length - l1 - 1) + 1
      n1 = random_number_lower_endian(l1, base)
      n2 = random_number_lower_endian(l2, base)
      result = lower_endian_to_number(n1, base) + lower_endian_to_number(
          n2, base)
      inputs = n1 + [base] + n2
      targets = number_to_lower_endian(result, base)
      yield {"inputs": inputs, "targets": targets}


@registry.register_problem
class AlgorithmicAdditionDecimal40(AlgorithmicAdditionBinary40):
  """Problem spec for algorithmic decimal addition task."""

  @property
  def num_symbols(self):
    return 10


@registry.register_problem
class AlgorithmicMultiplicationBinary40(AlgorithmicProblem):
  """Problem spec for algorithmic binary multiplication task."""

  @property
  def num_symbols(self):
    return 2

  def generator(self, base, max_length, nbr_cases):
    """Generator for the multiplication task.

    The length of each number is drawn uniformly at random in [1, max_length/2]
    and then digits are drawn uniformly at random. The numbers are multiplied
    and separated by [base] in the input. Stops at nbr_cases.

    Args:
      base: in which base are the numbers.
      max_length: integer, maximum length of sequences to generate.
      nbr_cases: the number of cases to generate.

    Yields:
      A dictionary {"inputs": input-list, "targets": target-list} where
      input-list are the 2 numbers and target-list is the result of multiplying
      them.

    Raises:
      ValueError: if max_length is lower than 3.
    """
    if max_length < 3:
      raise ValueError("Maximum length must be at least 3.")
    for _ in xrange(nbr_cases):
      l1 = np.random.randint(max_length // 2) + 1
      l2 = np.random.randint(max_length - l1 - 1) + 1
      n1 = random_number_lower_endian(l1, base)
      n2 = random_number_lower_endian(l2, base)
      result = lower_endian_to_number(n1, base) * lower_endian_to_number(
          n2, base)
      inputs = n1 + [base] + n2
      targets = number_to_lower_endian(result, base)
      yield {"inputs": inputs, "targets": targets}


@registry.register_problem
class AlgorithmicMultiplicationDecimal40(AlgorithmicMultiplicationBinary40):
  """Problem spec for algorithmic decimal multiplication task."""

  @property
  def num_symbols(self):
    return 10


@registry.register_problem
class AlgorithmicReverseBinary40Test(AlgorithmicReverseBinary40):
  """Test Problem with tiny dataset."""

  @property
  def train_length(self):
    return 10

  @property
  def dev_length(self):
    return 10

  @property
  def train_size(self):
    return 1000

  @property
  def dev_size(self):
    return 100

  @property
  def num_shards(self):
    return 1
