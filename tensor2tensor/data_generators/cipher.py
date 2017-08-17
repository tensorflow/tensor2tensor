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

"""Cipher data generators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import algorithmic
from tensor2tensor.utils import registry


@registry.register_problem
class AlgorithmicCipherShift5(algorithmic.AlgorithmicProblem):
  """Shift cipher."""

  @property
  def num_symbols(self):
    return 5

  @property
  def distribution(self):
    return [0.4, 0.3, 0.2, 0.08, 0.02]

  @property
  def shift(self):
    return 1

  def generator(self, nbr_symbols, max_length, nbr_cases):
    plain_vocab = range(nbr_symbols)
    indices = generate_plaintext_random(
        plain_vocab, self.distribution, nbr_cases, max_length)
    codes = encipher_shift(indices, plain_vocab, self.shift)
    for plain, code in zip(indices, codes):
      yield {"inputs": plain, "targets": code}

  @property
  def train_length(self):
    return 100

  @property
  def dev_length(self):
    return self.train_length


@registry.register_problem
class AlgorithmicCipherVigenere5(algorithmic.AlgorithmicProblem):
  """Vinegre cipher."""

  @property
  def num_symbols(self):
    return 5

  @property
  def distribution(self):
    return [0.4, 0.3, 0.2, 0.08, 0.02]

  @property
  def key(self):
    return [1, 3]

  def generator(self, nbr_symbols, max_length, nbr_cases):
    plain_vocab = range(nbr_symbols)
    indices = generate_plaintext_random(plain_vocab, self.distribution,
                                        nbr_cases, max_length)
    codes = encipher_vigenere(indices, plain_vocab, self.key)
    for plain, code in zip(indices, codes):
      yield {"inputs": plain, "targets": code}

  @property
  def train_length(self):
    return 200

  @property
  def dev_length(self):
    return self.train_length


@registry.register_problem
class AlgorithmicCipherShift200(AlgorithmicCipherShift5):
  """Shift cipher."""

  @property
  def num_symbols(self):
    return 200

  @property
  def distribution(self):
    vals = range(self.num_symbols)
    val_sum = sum(vals)
    return [v / val_sum for v in vals]


@registry.register_problem
class AlgorithmicCipherVigenere200(AlgorithmicCipherVigenere5):
  """Vinegre cipher."""

  @property
  def num_symbols(self):
    return 200

  @property
  def distribution(self):
    vals = range(self.num_symbols)
    val_sum = sum(vals)
    return [v / val_sum for v in vals]

  @property
  def key(self):
    return [1, 3]


class ShiftEncryptionLayer(object):
  """A single layer for shift."""

  def __init__(self, vocab, shift):
    """Initialize shift layer.

    Args:
      vocab: (list of String) the vocabulary
      shift: (Integer) the amount of shift apply to the alphabet.
        Positive number implies shift to the right, negative number
        implies shift to the left.
    """
    self.shift = shift
    alphabet = vocab
    shifted_alphabet = deque(alphabet)
    shifted_alphabet.rotate(shift)
    self.encrypt = dict(zip(alphabet, list(shifted_alphabet)))
    self.decrypt = dict(zip(list(shifted_alphabet), alphabet))

  def encrypt_character(self, character):
    return self.encrypt[character]

  def decrypt_character(self, character):
    return self.decrypt[character]


def generate_plaintext_random(plain_vocab, distribution, train_samples,
                              length):
  """Generates samples of text from the provided vocabulary.

  Args:
    plain_vocab: vocabulary.
    distribution: distribution.
    train_samples: samples for training.
    length: length.

  Returns:
    train_indices (np.array of Integers): random integers for training.
      shape = [num_samples, length]
    test_indices (np.array of Integers): random integers for testing.
      shape = [num_samples, length]
    plain_vocab   (list of Integers): unique vocabularies.
  """
  if distribution is not None:
    assert len(distribution) == len(plain_vocab)

  train_indices = np.random.choice(
      range(len(plain_vocab)), (train_samples, length), p=distribution)

  return train_indices


def encipher_shift(plaintext, plain_vocab, shift):
  """Encrypt plain text with a single shift layer.

  Args:
    plaintext (list of list of Strings): a list of plain text to encrypt.
    plain_vocab (list of Integer): unique vocabularies being used.
    shift (Integer): number of shift, shift to the right if shift is positive.
  Returns:
    ciphertext (list of Strings): encrypted plain text.
  """
  ciphertext = []
  cipher = ShiftEncryptionLayer(plain_vocab, shift)

  for _, sentence in enumerate(plaintext):
    cipher_sentence = []
    for _, character in enumerate(sentence):
      encrypted_char = cipher.encrypt_character(character)
      cipher_sentence.append(encrypted_char)
    ciphertext.append(cipher_sentence)

  return ciphertext


def encipher_vigenere(plaintext, plain_vocab, key):
  """Encrypt plain text with given key.

  Args:
    plaintext (list of list of Strings): a list of plain text to encrypt.
    plain_vocab (list of Integer): unique vocabularies being used.
    key (list of Integer): key to encrypt cipher using Vigenere table.

  Returns:
    ciphertext (list of Strings): encrypted plain text.
  """
  ciphertext = []
  # generate Vigenere table
  layers = []
  for i in range(len(plain_vocab)):
    layers.append(ShiftEncryptionLayer(plain_vocab, i))

  for i, sentence in enumerate(plaintext):
    cipher_sentence = []
    for j, character in enumerate(sentence):
      key_idx = key[j % len(key)]
      encrypted_char = layers[key_idx].encrypt_character(character)
      cipher_sentence.append(encrypted_char)
    ciphertext.append(cipher_sentence)

  return ciphertext
