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
import io
import os
import random
import shutil
import string

# Dependency imports
import mock
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import text_encoder
import tensorflow as tf


class NativeToUnicodeTest(tf.test.TestCase):

  def test_native_to_unicode(self):
    s = r"foo bar"
    s_unicode = text_encoder.native_to_unicode(s)
    if six.PY2:
      self.assertIsInstance(s_unicode, unicode)
    self.assertEqual(s_unicode, u"foo bar")


class EscapeUnescapeTokenTest(tf.test.TestCase):

  def test_escape_token(self):
    escaped = text_encoder._escape_token(
        "Foo! Bar.\nunder_score back\\slash",
        set("abcdefghijklmnopqrstuvwxyz .\n") | text_encoder._ESCAPE_CHARS)

    self.assertEqual(
        "\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_", escaped)

  def test_unescape_token(self):
    unescaped = text_encoder._unescape_token(
        "\\70;oo\\33; \\66;ar.\\10;under\\uscore back\\\\slash_")

    self.assertEqual(
        "Foo! Bar.\nunder_score back\\slash", unescaped)


class TokenTextEncoderTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Make sure the test dir exists and is empty."""
    cls.test_temp_dir = os.path.join(tf.test.get_temp_dir(), "encoder_test")
    shutil.rmtree(cls.test_temp_dir, ignore_errors=True)
    os.mkdir(cls.test_temp_dir)

  def test_save_and_reload(self):
    """Test that saving and reloading doesn't change the vocab.

    Note that this test reads and writes to the filesystem, which necessitates
    that this test size be "large".
    """

    corpus = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
    vocab_filename = os.path.join(self.test_temp_dir, "abc.vocab")

    # Make text encoder from a list and store vocab to fake filesystem.
    encoder = text_encoder.TokenTextEncoder(None, vocab_list=corpus.split())
    encoder.store_to_file(vocab_filename)

    # Load back the saved vocab file from the fake_filesystem.
    new_encoder = text_encoder.TokenTextEncoder(vocab_filename)

    self.assertEqual(encoder._id_to_token, new_encoder._id_to_token)
    self.assertEqual(encoder._token_to_id, new_encoder._token_to_id)

  def test_reserved_tokens_in_corpus(self):
    """Test that we handle reserved tokens appearing in the corpus."""
    corpus = "A B {} D E F {} G {}".format(text_encoder.EOS,
                                           text_encoder.EOS,
                                           text_encoder.PAD)

    encoder = text_encoder.TokenTextEncoder(None, vocab_list=corpus.split())

    all_tokens = encoder._id_to_token.values()

    # If reserved tokens are removed correctly, then the set of tokens will
    # be unique.
    self.assertEqual(len(all_tokens), len(set(all_tokens)))


class SubwordTextEncoderTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Make sure the test dir exists and is empty."""
    cls.test_temp_dir = os.path.join(tf.test.get_temp_dir(), "encoder_test")
    shutil.rmtree(cls.test_temp_dir, ignore_errors=True)
    os.mkdir(cls.test_temp_dir)

  def test_encode_decode(self):
    corpus = (
        "This is a corpus of text that provides a bunch of tokens from which "
        "to build a vocabulary. It will be used when strings are encoded "
        "with a TextEncoder subclass. The encoder was coded by a coder.")
    token_counts = collections.Counter(corpus.split(" "))
    alphabet = set(corpus) - {" "}

    original = "This is a coded sentence encoded by the SubwordTextEncoder."
    token_counts.update(original.split(" "))

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)

    # Encoding should be reversible.
    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)
    self.assertEqual(original, decoded)

    # The substrings coded and coder are frequent enough in the corpus that
    # they should appear in the vocabulary even though they are substrings
    # of other included strings.
    subtoken_strings = {encoder.all_subtoken_strings[i] for i in encoded}
    self.assertIn("encoded_", subtoken_strings)
    self.assertIn("coded_", subtoken_strings)
    self.assertIn("TextEncoder", encoder.all_subtoken_strings)
    self.assertIn("coder", encoder.all_subtoken_strings)

    # Every character in the corpus should be in the encoders alphabet and
    # its subtoken vocabulary.
    self.assertTrue(alphabet.issubset(encoder._alphabet))
    for a in alphabet:
      self.assertIn(a, encoder.all_subtoken_strings)

  def test_unicode(self):
    corpus = "Cat emoticons. \U0001F638 \U0001F639 \U0001F63A \U0001F63B"
    token_counts = collections.Counter(corpus.split(" "))

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)

    self.assertIn("\U0001F638", encoder._alphabet)
    self.assertIn("\U0001F63B", encoder.all_subtoken_strings)

  def test_small_vocab(self):
    corpus = "The quick brown fox jumps over the lazy dog"
    token_counts = collections.Counter(corpus.split(" "))
    alphabet = set(corpus) - {" "}

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        10, token_counts, 2, 10)

    # All vocabulary elements are in the alphabet and subtoken strings even
    # if we requested a smaller vocabulary to assure all expected strings
    # are encodable.
    self.assertTrue(alphabet.issubset(encoder._alphabet))
    for a in alphabet:
      self.assertIn(a, encoder.all_subtoken_strings)

  def test_long_tokens(self):
    """Subword tokenization should still run efficiently with long tokens.

    To make it run efficiently, we need to use the `max_subtoken_length`
    argument when calling SubwordTextEncoder.build_to_target_size.
    """
    token_length = 4000
    num_tokens = 50
    target_vocab_size = 600
    max_subtoken_length = 10  # Set this to `None` to get problems.
    max_count = 500

    # Generate some long random strings.
    random.seed(0)
    long_tokens = []
    for _ in range(num_tokens):
      long_token = "".join([random.choice(string.ascii_uppercase)
                            for _ in xrange(token_length)])
      long_tokens.append(long_token)

    corpus = " ".join(long_tokens)
    token_counts = collections.Counter(corpus.split(" "))
    alphabet = set(corpus) - {" "}

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        target_vocab_size, token_counts, 1, max_count, num_iterations=1,
        max_subtoken_length=max_subtoken_length)

    # All vocabulary elements are in the alphabet and subtoken strings even
    # if we requested a smaller vocabulary to assure all expected strings
    # are encodable.
    self.assertTrue(alphabet.issubset(encoder._alphabet))
    for a in alphabet:
      self.assertIn(a, encoder.all_subtoken_strings)

  def test_custom_reserved_tokens(self):
    """Test that we can pass custom reserved tokens to SubwordTextEncoder."""
    corpus = "The quick brown fox jumps over the lazy dog"
    token_counts = collections.Counter(corpus.split(" "))

    start_symbol = "<S>"
    end_symbol = "<E>"
    reserved_tokens = text_encoder.RESERVED_TOKENS + [start_symbol,
                                                      end_symbol]
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        10, token_counts, 2, 10, reserved_tokens=reserved_tokens)

    # Make sure that reserved tokens appear in the right places.
    start_id = encoder._subtoken_string_to_id[start_symbol]
    end_id = encoder._subtoken_string_to_id[end_symbol]
    self.assertEqual(start_id, 2)
    self.assertEqual(end_id, 3)

    # Make sure that we haven't messed up the ability to reconstruct.
    reconstructed_corpus = encoder.decode(encoder.encode(corpus))
    self.assertEqual(corpus, reconstructed_corpus)

  def test_encodable_when_not_in_alphabet(self):
    corpus = "the quick brown fox jumps over the lazy dog"
    token_counts = collections.Counter(corpus.split(" "))

    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)
    original = "This has UPPER CASE letters that are out of alphabet"

    # Early versions could have an infinite loop when breaking into subtokens
    # if there was any out-of-alphabet characters in the encoded string.
    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)

    self.assertEqual(original, decoded)
    encoded_str = "".join(encoder.all_subtoken_strings[i] for i in encoded)
    self.assertIn("\\84;", encoded_str)

  @mock.patch.object(text_encoder, "_ESCAPE_CHARS", new=set("\\_;13579"))
  def test_raises_exception_when_not_encodable(self):
    corpus = "the quick brown fox jumps over the lazy dog"
    token_counts = collections.Counter(corpus.split(" "))

    # Deliberately exclude some required encoding chars from the alphabet
    # and token list, making some strings unencodable.
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)
    original = "This has UPPER CASE letters that are out of alphabet"

    # Previously there was a bug which produced an infinite loop in this case.
    with self.assertRaises(AssertionError):
      encoder.encode(original)

  def test_load_from_file(self):
    # Test a vocab file with words not wrapped with single quotes
    encoder = text_encoder.SubwordTextEncoder()
    correct_vocab = ["the", "and", "of"]
    vocab = io.StringIO("the\n"
                        "and\n"
                        "of\n")
    encoder._load_from_file_object(vocab)
    self.assertAllEqual(encoder.all_subtoken_strings, correct_vocab)

    # Test a vocab file with words wrapped in single quotes
    encoder = text_encoder.SubwordTextEncoder()
    vocab = io.StringIO("\"the\"\n"
                        "\"and\"\n"
                        "\"of\"\n")
    encoder._load_from_file_object(vocab)
    self.assertAllEqual(encoder.all_subtoken_strings, correct_vocab)

  def test_reserved_token_chars_not_in_alphabet(self):
    corpus = "dog"
    token_counts = collections.Counter(corpus.split(" "))
    encoder1 = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 100)
    filename = os.path.join(self.test_temp_dir, "out.voc")
    encoder1.store_to_file(filename)
    encoder2 = text_encoder.SubwordTextEncoder(filename=filename)

    self.assertEqual(encoder1._alphabet, encoder2._alphabet)

    for t in text_encoder.RESERVED_TOKENS:
      for c in t:
        # Verify that encoders can encode all reserved token chars.
        encoder1.encode(c)
        encoder2.encode(c)

  def test_save_and_reload(self):
    corpus = "the quick brown fox jumps over the lazy dog"
    token_counts = collections.Counter(corpus.split(" "))

    # Deliberately exclude some required encoding chars from the alphabet
    # and token list, making some strings unencodable.
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)

    filename = os.path.join(self.test_temp_dir, "out.voc")
    encoder.store_to_file(filename)
    new_encoder = text_encoder.SubwordTextEncoder(filename)

    self.assertEqual(encoder._alphabet, new_encoder._alphabet)
    self.assertEqual(encoder.all_subtoken_strings,
                     new_encoder.all_subtoken_strings)
    self.assertEqual(encoder._subtoken_string_to_id,
                     new_encoder._subtoken_string_to_id)
    self.assertEqual(encoder._max_subtoken_len, new_encoder._max_subtoken_len)

  def test_save_and_reload_no_single_quotes(self):
    corpus = "the quick brown fox jumps over the lazy dog"
    token_counts = collections.Counter(corpus.split(" "))

    # Deliberately exclude some required encoding chars from the alphabet
    # and token list, making some strings unencodable.
    encoder = text_encoder.SubwordTextEncoder.build_to_target_size(
        100, token_counts, 2, 10)

    filename = os.path.join(self.test_temp_dir, "out.voc")
    encoder.store_to_file(filename, add_single_quotes=False)
    new_encoder = text_encoder.SubwordTextEncoder(filename)

    self.assertEqual(encoder._alphabet, new_encoder._alphabet)
    self.assertEqual(encoder.all_subtoken_strings,
                     new_encoder.all_subtoken_strings)
    self.assertEqual(encoder._subtoken_string_to_id,
                     new_encoder._subtoken_string_to_id)
    self.assertEqual(encoder._max_subtoken_len, new_encoder._max_subtoken_len)


if __name__ == "__main__":
  tf.test.main()
