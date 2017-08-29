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
from __future__ import unicode_literals

import collections
import os
import shutil

# Dependency imports
import mock

from tensor2tensor.data_generators import text_encoder
import tensorflow as tf


class EscapeUnescapeTokenTest(tf.test.TestCase):

  def test_escape_token(self):
    escaped = text_encoder._escape_token(
        'Foo! Bar.\nunder_score back\\slash',
        set('abcdefghijklmnopqrstuvwxyz .\n') | text_encoder._ESCAPE_CHARS)

    self.assertEqual(
        '\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_', escaped)

  def test_unescape_token(self):
    unescaped = text_encoder._unescape_token(
        '\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_')

    self.assertEqual(
        'Foo! Bar.\nunder_score back\\slash', unescaped)


class TokenTextEncoderTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Make sure the test dir exists and is empty."""
    cls.test_temp_dir = os.path.join(tf.test.get_temp_dir(), 'encoder_test')
    shutil.rmtree(cls.test_temp_dir, ignore_errors=True)
    os.mkdir(cls.test_temp_dir)

  def test_save_and_reload(self):
    """Test that saving and reloading doesn't change the vocab.

    Note that this test reads and writes to the filesystem, which necessitates
    that this test size be "large".
    """

    corpus = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'
    vocab_filename = os.path.join(self.test_temp_dir, 'abc.vocab')

    # Make text encoder from a list and store vocab to fake filesystem.
    encoder = text_encoder.TokenTextEncoder(None, vocab_list=corpus.split())
    encoder.store_to_file(vocab_filename)

    # Load back the saved vocab file from the fake_filesystem.
    new_encoder = text_encoder.TokenTextEncoder(vocab_filename)

    self.assertEqual(encoder._id_to_token, new_encoder._id_to_token)
    self.assertEqual(encoder._token_to_id, new_encoder._token_to_id)

  def test_reserved_tokens_in_corpus(self):
    """Test that we handle reserved tokens appearing in the corpus."""
    corpus = 'A B {} D E F {} G {}'.format(text_encoder.EOS,
                                           text_encoder.EOS,
                                           text_encoder.PAD)

    encoder = text_encoder.TokenTextEncoder(None, vocab_list=corpus.split())

    all_tokens = encoder._id_to_token.values()

    # If reserved tokens are removed correctly, then the set of tokens will
    # be unique.
    self.assertEqual(len(all_tokens), len(set(all_tokens)))

  def test_unknown_token_returned_for_oov_term(self):
    "Test that unknown token is returned for term not occuring in corpus."
    unk_token = 'UNK'
    corpus = 'A B C D E F G H ' + unk_token

    encoder = text_encoder.TokenTextEncoder(None, 
                                            vocab_list=corpus.split(), 
                                            unknown_token=unk_token)

    encoded = encoder.encode('H Z')
    self.assertEqual('H ' + unk_token, encoder.decode(encoded))

  def test_unknown_token_not_in_vocab_raises_error(self):
    "Test that when vocabulary does not contain unknown token it fails fast."
    corpus = 'A B C D E F G H'

    with self.assertRaises(Exception):
      text_encoder.TokenTextEncoder(None, 
                                    vocab_list=corpus.split(), 
                                    unknown_token='UNK')


class SubwordTextEncoderTest(tf.test.TestCase):

  def test_encode_decode(self):
    corpus = (
        'This is a corpus of text that provides a bunch of tokens from which '
        'to build a vocabulary. It will be used when strings are encoded '
        'with a TextEncoder subclass. The encoder was coded by a coder.')
    token_counts = collections.Counter(corpus.split(' '))
    alphabet = set(corpus) ^ {' '}

    original = 'This is a coded sentence encoded by the SubwordTextEncoder.'
    token_counts.update(original.split(' '))

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)

    # Encoding should be reversible.
    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)
    self.assertEqual(original, decoded)

    # The substrings coded and coder are frequent enough in the corpus that
    # they should appear in the vocabulary even though they are substrings
    # of other included strings.
    subtoken_strings = {encoder._all_subtoken_strings[i] for i in encoded}
    self.assertIn('encoded_', subtoken_strings)
    self.assertIn('coded_', subtoken_strings)
    self.assertIn('TextEncoder', encoder._all_subtoken_strings)
    self.assertIn('coder', encoder._all_subtoken_strings)

    # Every character in the corpus should be in the encoder's alphabet and
    # its subtoken vocabulary.
    self.assertTrue(alphabet.issubset(encoder._alphabet))
    for a in alphabet:
      self.assertIn(a, encoder._all_subtoken_strings)

  def test_unicode(self):
    corpus = 'Cat emoticons. \U0001F638 \U0001F639 \U0001F63A \U0001F63B'
    token_counts = collections.Counter(corpus.split(' '))

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)

    self.assertIn('\U0001F638', encoder._alphabet)
    self.assertIn('\U0001F63B', encoder._all_subtoken_strings)

  def test_small_vocab(self):
    corpus = 'The quick brown fox jumps over the lazy dog'
    token_counts = collections.Counter(corpus.split(' '))
    alphabet = set(corpus) ^ {' '}

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        10, token_counts, 2, 10)

    # All vocabulary elements are in the alphabet and subtoken strings even
    # if we requested a smaller vocabulary to assure all expected strings
    # are encodable.
    self.assertTrue(alphabet.issubset(encoder._alphabet))
    for a in alphabet:
      self.assertIn(a, encoder._all_subtoken_strings)

  def test_encodable_when_not_in_alphabet(self):
    corpus = 'the quick brown fox jumps over the lazy dog'
    token_counts = collections.Counter(corpus.split(' '))

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)
    original = 'This has UPPER CASE letters that are out of alphabet'

    # Early versions could have an infinite loop when breaking into subtokens
    # if there was any out-of-alphabet characters in the encoded string.
    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)

    self.assertEqual(original, decoded)
    encoded_str = ''.join(encoder._all_subtoken_strings[i] for i in encoded)
    self.assertIn('\\84;', encoded_str)

  @mock.patch.object(text_encoder, '_ESCAPE_CHARS', new=set('\\_;13579'))
  def test_raises_exception_when_not_encodable(self):
    corpus = 'the quick brown fox jumps over the lazy dog'
    token_counts = collections.Counter(corpus.split(' '))

    # Deliberately exclude some required encoding chars from the alphabet
    # and token list, making some strings unencodable.
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)
    original = 'This has UPPER CASE letters that are out of alphabet'

    # Previously there was a bug which produced an infinite loop in this case.
    with self.assertRaises(AssertionError):
      encoder.encode(original)


if __name__ == '__main__':
  tf.test.main()
