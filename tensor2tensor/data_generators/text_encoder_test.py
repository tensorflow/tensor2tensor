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

"""Tests for tensor2tensor.data_generators.text_encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import text_encoder
import tensorflow as tf


class EscapeUnescapeTokenTest(tf.test.TestCase):

  def test_escape_token(self):
    escaped = text_encoder._escape_token(
        u'Foo! Bar.\nunder_score back\\slash',
        set('abcdefghijklmnopqrstuvwxyz .\n') | text_encoder._ESCAPE_CHARS)

    self.assertEqual(
        u'\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_', escaped)

  def test_unescape_token(self):
    unescaped = text_encoder._unescape_token(
        u'\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_')

    self.assertEqual(
        u'Foo! Bar.\nunder_score back\\slash', unescaped)


class SubwordTextEncoderTest(tf.test.TestCase):

  def test_encode_decode(self):
    token_counts = {
        u'this': 9,
        u'sentence': 14,
        u'the': 100,
        u'encoded': 1,
        u'was': 20,
        u'by': 50,
    }
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        50, token_counts, 2, 10)
    encoder.build_from_token_counts(token_counts, min_count=2)

    original = 'This sentence was encoded by the SubwordTextEncoder.'
    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)
    self.assertEqual(original, decoded)


if __name__ == '__main__':
  tf.test.main()
