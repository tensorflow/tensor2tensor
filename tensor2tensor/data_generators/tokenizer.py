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

"""A simple invertible tokenizer.

Converts from a raw string to a list of tokens (strings).

This tokenizer has the following desirable properties:
 - It is invertible.
 - Punctuation is broken away from adjacent letters.
 - A single space between words does not produce an extra token.

The tokenization algorithm is as follows:

0.  We classify the 256 characters into "word characters" and
    "separator characters".  Separator characters are defined as the union of
    string.punctuation and string.whitespace.  All other characters are
    "word characters".

1.  Split the text into a list of tokens, splitting at every boundary of a
    "word character" and a "separator character".  This produces a list which
    alternates between "word tokens" (strings of word characters) and
    "separator tokens" (strings of of separator characters).

2.  Remove every token consisting of a single space, unless it is
    the very first or very last token in the list.  These tokens are now
    implied by the fact that there are two adjacent word tokens.

e.g.  "Dude - that's so cool."
        -> ["Dude", " - ", "that", "'", "s", "so", "cool", "."]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import array
import string

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin


class Tokenizer(object):
  """Vocab for breaking words into wordpieces.
  """

  def __init__(self):
    self._separator_chars = string.punctuation + string.whitespace
    self._separator_char_mask = array.array(
        "l", [chr(i) in self._separator_chars for i in xrange(256)])
    self.token_counts = dict()

  def _increment_token_count(self, token):
    if token in self.token_counts:
      self.token_counts[token] += 1
    else:
      self.token_counts[token] = 1

  def encode(self, raw_text):
    """Encode a raw string as a list of tokens.

    Args:
      raw_text: a string
    Returns:
      a list of stirngs.
    """
    if not raw_text:
      return []
    ret = []
    token_start = 0
    for pos in xrange(1, len(raw_text)):
      if (self._is_separator_char(raw_text[pos]) !=
          self._is_separator_char(raw_text[pos - 1])):
        token = raw_text[token_start:pos]
        if token != " " or token_start == 0:
          ret.append(token)
          self._increment_token_count(token)
        token_start = pos
    final_token = raw_text[token_start:]
    ret.append(final_token)
    self._increment_token_count(final_token)
    return ret

  def decode(self, tokens):
    """Decode a list of tokens to a string.

    Args:
      tokens: a list of stirngs
    Returns:
      a string.
    """
    ret = ""
    for i, token in enumerate(tokens):
      if (i > 0 and self._is_word_char(tokens[i - 1][0]) and
          self._is_word_char(token[0])):
        ret += " "
      ret += token
    return ret

  def _is_separator_char(self, c):
    return self._separator_char_mask[ord(c)]

  def _is_word_char(self, c):
    return not self._is_separator_char(c)
