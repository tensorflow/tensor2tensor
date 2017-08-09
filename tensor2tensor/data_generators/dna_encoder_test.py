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

"""Tests for tensor2tensor.data_generators.dna_encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import dna_encoder
import tensorflow as tf


class DnaEncoderTest(tf.test.TestCase):

  def test_encode_decode(self):
    original = 'TTCGCGGNNNAACCCAACGCCATCTATGTANNTTGAGTTGTTGAGTTAAA'

    # Encoding should be reversible for any reasonable chunk size.
    for chunk_size in [1, 2, 4, 6, 8]:
      encoder = dna_encoder.DNAEncoder(chunk_size=chunk_size)
      encoded = encoder.encode(original)
      decoded = encoder.decode(encoded)
      self.assertEqual(original, decoded)

  def test_delimited_dna_encoder(self):
    original = 'TTCGCGGNNN,AACCCAACGC,CATCTATGTA,NNTTGAGTTG,TTGAGTTAAA'

    # Encoding should be reversible for any reasonable chunk size.
    for chunk_size in [1, 2, 4, 6, 8]:
      encoder = dna_encoder.DelimitedDNAEncoder(chunk_size=chunk_size)
      encoded = encoder.encode(original)
      decoded = encoder.decode(encoded)
      self.assertEqual(original, decoded)


if __name__ == '__main__':
  tf.test.main()
