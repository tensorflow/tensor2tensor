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

from collections import defaultdict
import string
import unicodedata
import sys
import re

# Dependency imports

from six import PY2
from six.moves import xrange  # pylint: disable=redefined-builtin


# Regular expression that matches Unicode whitespace characters
# (including ASCII whitespace) as defined in the Python run-time library
_RE_WHITESPACE = re.compile(r"^\s$", re.UNICODE)


class Tokenizer(object):
  """Vocab for breaking words into Unicode wordpieces.
  """

  _UNICODE_PUNCTUATION = set(unichr(i) for i in xrange(sys.maxunicode)
                             if unicodedata.category(unichr(i)).startswith('P'))
  _UNICODE_WHITESPACE = set(unichr(i) for i in xrange(sys.maxunicode)
                            if _RE_WHITESPACE.match(unichr(i)))
  #_SEPARATOR_CHAR_SET = set(string.punctuation + string.whitespace)
  _SEPARATOR_CHAR_SET = _UNICODE_WHITESPACE | _UNICODE_PUNCTUATION

  def __init__(self):
    self.token_counts = defaultdict(int)

  def encode(self, raw_text):
    """Encode a raw string as a list of tokens.

    Args:
      raw_text: a (Python2 or Python3 native) string
    Returns:
      a list of Unicode strings
    """
    if not raw_text:
      return []
    ret = []
    token_start = 0
    if PY2:
      raw_text = raw_text.decode('utf-8') # Convert to Unicode
    is_sep = [self._is_separator_char(c) for c in raw_text]
    for pos in xrange(1, len(raw_text)):
      if (is_sep[pos] != is_sep[pos-1]):
        token = raw_text[token_start:pos]
        if token != u" " or token_start == 0:
          ret.append(token)
          self.token_counts[token] += 1
        token_start = pos
    final_token = raw_text[token_start:]
    ret.append(final_token)
    self.token_counts[final_token] += 1
    return ret

  def decode(self, tokens):
    """Decode a list of tokens to a string.

    Args:
      tokens: a list of Unicode strings
    Returns:
      a (Python2 or Python3 native) string
    """
    ret = u""
    is_word = [self._is_word_char(t[0]) for t in tokens]
    for i, token in enumerate(tokens):
      if i > 0 and is_word[i - 1] and is_word[i]:
        ret += u" "
      ret += token
    return ret.encode('utf-8') if PY2 else ret

  def _is_separator_char(self, c):
    return c in self._SEPARATOR_CHAR_SET

  def _is_word_char(self, c):
    return c not in self._SEPARATOR_CHAR_SET
