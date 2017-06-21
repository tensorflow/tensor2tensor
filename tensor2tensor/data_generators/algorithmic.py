# Copyright 2017 Google Inc.
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


def identity_generator(nbr_symbols, max_length, nbr_cases):
  """Generator for the identity (copy) task on sequences of symbols.

  The length of the sequence is drawn uniformly at random from [1, max_length]
  and then symbols are drawn uniformly at random from [2, nbr_symbols] until
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
    inputs = [np.random.randint(nbr_symbols) + 2 for _ in xrange(l)]
    yield {"inputs": inputs, "targets": inputs + [1]}  # [1] for EOS


def shift_generator(nbr_symbols, shift, max_length, nbr_cases):
  """Generator for the shift task on sequences of symbols.

  The length of the sequence is drawn uniformly at random from [1, max_length]
  and then symbols are drawn uniformly at random from [2, nbr_symbols - shift]
  until nbr_cases sequences have been produced (output[i] = input[i] + shift).

  Args:
    nbr_symbols: number of symbols to use in each sequence (input + output).
    shift: by how much to shift the input.
    max_length: integer, maximum length of sequences to generate.
    nbr_cases: the number of cases to generate.

  Yields:
    A dictionary {"inputs": input-list, "targets": target-list} where
    target-list[i] = input-list[i] + shift.
  """
  for _ in xrange(nbr_cases):
    l = np.random.randint(max_length) + 1
    inputs = [np.random.randint(nbr_symbols - shift) + 2 for _ in xrange(l)]
    yield {"inputs": inputs,
           "targets": [i + shift for i in inputs] + [1]}  # [1] for EOS


def reverse_generator(nbr_symbols, max_length, nbr_cases):
  """Generator for the reversing task on sequences of symbols.

  The length of the sequence is drawn uniformly at random from [1, max_length]
  and then symbols are drawn uniformly at random from [2, nbr_symbols] until
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
    inputs = [np.random.randint(nbr_symbols) + 2 for _ in xrange(l)]
    yield {"inputs": inputs,
           "targets": list(reversed(inputs)) + [1]}  # [1] for EOS


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


def addition_generator(base, max_length, nbr_cases):
  """Generator for the addition task.

  The length of each number is drawn uniformly at random from [1, max_length/2]
  and then digits are drawn uniformly at random. The numbers are added and
  separated by [base+1] in the input. Stops at nbr_cases.

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
    result = lower_endian_to_number(n1, base) + lower_endian_to_number(n2, base)
    # We shift digits by 1 on input and output to leave 0 for padding.
    inputs = [i + 2 for i in n1] + [base + 2] + [i + 2 for i in n2]
    targets = [i + 2 for i in number_to_lower_endian(result, base)]
    yield {"inputs": inputs, "targets": targets + [1]}  # [1] for EOS


def multiplication_generator(base, max_length, nbr_cases):
  """Generator for the multiplication task.

  The length of each number is drawn uniformly at random from [1, max_length/2]
  and then digits are drawn uniformly at random. The numbers are multiplied
  and separated by [base+1] in the input. Stops at nbr_cases.

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
    result = lower_endian_to_number(n1, base) * lower_endian_to_number(n2, base)
    # We shift digits by 1 on input and output to leave 0 for padding.
    inputs = [i + 2 for i in n1] + [base + 2] + [i + 2 for i in n2]
    targets = [i + 2 for i in number_to_lower_endian(result, base)]
    yield {"inputs": inputs, "targets": targets + [1]}  # [1] for EOS
