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

from six import PY2
from six import unichr  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf


# Conversion between Unicode and UTF-8, if required (on Python2)
_native_to_unicode = (lambda s: s.decode("utf-8")) if PY2 else (lambda s: s)


# This set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    unichr(i) for i in xrange(sys.maxunicode)
    if (unicodedata.category(unichr(i)).startswith("L") or
        unicodedata.category(unichr(i)).startswith("N")))


def encode(text):
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
  is_alnum = [c in _ALPHANUMERIC_CHAR_SET for c in text]
  for pos in xrange(1, len(text)):
    if is_alnum[pos] != is_alnum[pos - 1]:
      token = text[token_start:pos]
      if token != u" " or token_start == 0:
        ret.append(token)
      token_start = pos
  final_token = text[token_start:]
  ret.append(final_token)
  return ret


def decode(tokens):
  """Decode a list of tokens to a unicode string.

  Args:
    tokens: a list of Unicode strings
  Returns:
    a unicode string
  """
  ret = u""
  token_is_alnum = [t[0] in _ALPHANUMERIC_CHAR_SET for t in tokens]
  for i, token in enumerate(tokens):
    if i > 0 and token_is_alnum[i - 1] and token_is_alnum[i]:
      ret += u" "
    ret += token
  return ret


def corpus_token_counts(text_filepattern, corpus_max_lines,
                        split_on_newlines=True):
  """Read the corpus and compute a dictionary of token counts.

  Args:
    text_filepattern: a pattern matching one or more files
    corpus_max_lines: an integer - maximum total lines to read.
    split_on_newlines: a boolean.  If true, then split files by lines and strip
      leading and trailing whitespace from each line.

  Returns:
    a dictionary from token to count.
  """
  def read_corpus():
    """Read the corpus."""
    docs = []
    lines_read = 0
    filenames = tf.gfile.Glob(text_filepattern)
    for text_filename in filenames:
      with tf.gfile.Open(text_filename) as f:
        if not split_on_newlines:
          docs.append("")
        for line in f:
          if split_on_newlines:
            # The tokenizer updates token_counts in encode()
            docs.append(line.strip())
          else:
            docs[-1] += line
          lines_read += 1
          if corpus_max_lines > 0 and lines_read > corpus_max_lines:
            return docs
    return docs

  counts = defaultdict(int)
  for doc in read_corpus():
    for tok in encode(_native_to_unicode(doc)):
      counts[tok] += 1
  return counts

