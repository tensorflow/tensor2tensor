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

Converts from a unicode string to a list of tokens
(represented as Unicode strings).

This tokenizer has the following desirable properties:
 - It is invertible.
 - Alphanumeric characters are broken away from non-alphanumeric characters.
 - A single space between words does not produce an extra token.
 - The full Unicode punctuation and separator set is recognized.

The tokenization algorithm is as follows:

1.  Split the text into a list of tokens, splitting at every boundary of an
    alphanumeric character and a non-alphanumeric character.  This produces
    a list which alternates between "alphanumeric tokens"
    (strings of alphanumeric characters) and "non-alphanumeric tokens"
    (strings of of non-alphanumeric characters).

2.  Remove every token consisting of a single space, unless it is
    the very first or very last token in the list.  These tokens are now
    implied by the fact that there are two adjacent alphanumeric tokens.

e.g.  u"Dude - that's so cool."
        -> [u"Dude", u" - ", u"that", u"'", u"s", u"so", u"cool", u"."]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import sys
import unicodedata

# Dependency imports

from six import unichr  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin


class Tokenizer(object):
  """Vocab for breaking words into Unicode wordpieces.
  """

  # This set contains all letter and number characters.
  _ALPHANUMERIC_CHAR_SET = set(
      unichr(i) for i in xrange(sys.maxunicode)
      if (unicodedata.category(unichr(i)).startswith("L") or
          unicodedata.category(unichr(i)).startswith("N")))

  def __init__(self):
    self.token_counts = defaultdict(int)

  def encode(self, text):
    """Encode a unicode string as a list of tokens.

    Args:
      text: a unicode string
    Returns:
      a list of tokens as Unicode strings
    """
    if not text:
      return []
    ret = []
    token_start = 0
    # Classify each character in the input string
    is_alnum = [c in self._ALPHANUMERIC_CHAR_SET for c in text]
    for pos in xrange(1, len(text)):
      if is_alnum[pos] != is_alnum[pos - 1]:
        token = text[token_start:pos]
        if token != u" " or token_start == 0:
          ret.append(token)
          self.token_counts[token] += 1
        token_start = pos
    final_token = text[token_start:]
    ret.append(final_token)
    self.token_counts[final_token] += 1
    return ret

  def decode(self, tokens):
    """Decode a list of tokens to a unicode string.

    Args:
      tokens: a list of Unicode strings
    Returns:
      a unicode string
    """
    ret = u""
    token_is_alnum = [t[0] in self._ALPHANUMERIC_CHAR_SET for t in tokens]
    for i, token in enumerate(tokens):
      if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
        ret += u" "
      ret += token
    return ret
