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

# coding=utf-8
"""Tests for tensor2tensor.data_generators.tokenizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import tokenizer

import tensorflow as tf


class TokenizerTest(tf.test.TestCase):

  def testEncode(self):
    t = tokenizer.Tokenizer()
    self.assertEqual(
        t.encode(u"Dude - that's so cool."),
        [u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."])
    self.assertEqual(
        t.encode(u"Łukasz est né en 1981."),
        [u"Łukasz", u"est", u"né", u"en", u"1981", u"."])
    self.assertEqual(
        t.encode(u" Spaces at the ends "),
        [u" ", u"Spaces", u"at", u"the", u"ends", u" "])
    self.assertEqual(t.encode(u"802.11b"), [u"802", u".", u"11b"])
    self.assertEqual(t.encode(u"two. \nlines"), [u"two", u". \n", u"lines"])

  def testDecode(self):
    t = tokenizer.Tokenizer()
    self.assertEqual(
        t.decode([u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."]),
        u"Dude - that's so cool.")

  def testInvertibilityOnRandomStrings(self):
    t = tokenizer.Tokenizer()
    random.seed(123)
    for _ in xrange(1000):
      s = u"".join([unichr(random.randint(0, 65535)) for _ in xrange(10)])
      self.assertEqual(s, t.decode(t.encode(s)))


if __name__ == "__main__":
  tf.test.main()
