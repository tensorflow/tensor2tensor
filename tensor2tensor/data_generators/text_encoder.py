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

"""Encoders for text data.

* TextEncoder: base class
* ByteTextEncoder: for ascii text
* TokenTextEncoder: with user-supplied vocabulary file
* SubwordTextEncoder: invertible
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import re

# Dependency imports

import six
from six import PY2
from six import unichr  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import tokenizer

import tensorflow as tf


# Reserved tokens for things like padding and EOS symbols.
PAD = "<pad>"
EOS = "<EOS>"
RESERVED_TOKENS = [PAD, EOS]
NUM_RESERVED_TOKENS = len(RESERVED_TOKENS)
PAD_ID = RESERVED_TOKENS.index(PAD)  # Normally 0
EOS_ID = RESERVED_TOKENS.index(EOS)  # Normally 1

if PY2:
  RESERVED_TOKENS_BYTES = RESERVED_TOKENS
else:
  RESERVED_TOKENS_BYTES = [bytes(PAD, "ascii"), bytes(EOS, "ascii")]


# Regular expression for unescaping token strings.
# '\u' is converted to '_'
# '\\' is converted to '\'
# '\213;' is converted to unichr(213)
_UNESCAPE_REGEX = re.compile(u"|".join([r"\\u", r"\\\\", r"\\([0-9]+);"]))


def native_to_unicode_py2(s):
  """Python 2: transform native string to Unicode."""
  if isinstance(s, unicode):
    return s
  return s.decode("utf-8")


# Conversion between Unicode and UTF-8, if required (on Python2)
if PY2:
  native_to_unicode = native_to_unicode_py2
  unicode_to_native = lambda s: s.encode("utf-8")
else:
  # No conversion required on Python3
  native_to_unicode = lambda s: s
  unicode_to_native = lambda s: s


class TextEncoder(object):
  """Base class for converting from ints to/from human readable strings."""

  def __init__(self, num_reserved_ids=NUM_RESERVED_TOKENS):
    self._num_reserved_ids = num_reserved_ids

  @property
  def num_reserved_ids(self):
    return self._num_reserved_ids

  def encode(self, s):
    """Transform a human-readable string into a sequence of int ids.

    The ids should be in the range [num_reserved_ids, vocab_size). Ids [0,
    num_reserved_ids) are reserved.

    EOS is not appended.

    Args:
      s: human-readable string to be converted.

    Returns:
      ids: list of integers
    """
    return [int(w) + self._num_reserved_ids for w in s.split()]

  def decode(self, ids):
    """Transform a sequence of int ids into a human-readable string.

    EOS is not expected in ids.

    Args:
      ids: list of integers to be converted.

    Returns:
      s: human-readable string.
    """
    decoded_ids = []
    for id_ in ids:
      if 0 <= id_ < self._num_reserved_ids:
        decoded_ids.append(RESERVED_TOKENS[int(id_)])
      else:
        decoded_ids.append(id_ - self._num_reserved_ids)
    return " ".join([str(d) for d in decoded_ids])

  @property
  def vocab_size(self):
    raise NotImplementedError()


class ByteTextEncoder(TextEncoder):
  """Encodes each byte to an id. For 8-bit strings only."""

  def encode(self, s):
    numres = self._num_reserved_ids
    if PY2:
      return [ord(c) + numres for c in s]
    # Python3: explicitly convert to UTF-8
    return [c + numres for c in s.encode("utf-8")]

  def decode(self, ids):
    numres = self._num_reserved_ids
    decoded_ids = []
    int2byte = six.int2byte
    for id_ in ids:
      if 0 <= id_ < numres:
        decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
      else:
        decoded_ids.append(int2byte(id_ - numres))
    if PY2:
      return "".join(decoded_ids)
    # Python3: join byte arrays and then decode string
    return b"".join(decoded_ids).decode("utf-8", "replace")

  @property
  def vocab_size(self):
    return 2**8 + self._num_reserved_ids


class TokenTextEncoder(TextEncoder):
  """Encoder based on a user-supplied vocabulary."""

  def __init__(self, vocab_filename, reverse=False,
               num_reserved_ids=NUM_RESERVED_TOKENS):
    """Initialize from a file, one token per line."""
    super(TokenTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
    self._reverse = reverse
    self._load_vocab_from_file(vocab_filename)

  def encode(self, sentence):
    """Converts a space-separated string of tokens to a list of ids."""
    ret = [self._token_to_id[tok] for tok in sentence.strip().split()]
    return ret[::-1] if self._reverse else ret

  def decode(self, ids):
    seq = reversed(ids) if self._reverse else ids
    return " ".join([self._safe_id_to_token(i) for i in seq])

  @property
  def vocab_size(self):
    return len(self._id_to_token)

  def _safe_id_to_token(self, idx):
    return self._id_to_token.get(idx, "ID_%d" % idx)

  def _load_vocab_from_file(self, filename):
    """Load vocab from a file."""
    self._token_to_id = {}
    self._id_to_token = {}

    for idx, tok in enumerate(RESERVED_TOKENS):
      self._token_to_id[tok] = idx
      self._id_to_token[idx] = tok

    token_start_idx = self._num_reserved_ids
    with tf.gfile.Open(filename) as f:
      for i, line in enumerate(f):
        idx = token_start_idx + i
        tok = line.strip()
        self._token_to_id[tok] = idx
        self._id_to_token[idx] = tok


class SubwordTextEncoder(TextEncoder):
  """Class for invertibly encoding text using a limited vocabulary.

  Invertibly encodes a native string as a sequence of subtokens from a limited
  vocabulary.

  A SubwordTextEncoder is built from a corpus (so it is tailored to the text in
  the corpus), and stored to a file. See text_encoder_build_subword.py.

  It can then be loaded and used to encode/decode any text.

  Encoding has four phases:

  1. Tokenize into a list of tokens.  Each token is a unicode string of either
     all alphanumeric characters or all non-alphanumeric characters.  We drop
     tokens consisting of a single space that are between two alphanumeric
     tokens.

  2. Escape each token.  This escapes away special and out-of-vocabulary
     characters, and makes sure that each token ends with an underscore, and
     has no other underscores.

  3. Represent each escaped token as a the concatenation of a list of subtokens
     from the limited vocabulary.  Subtoken selection is done greedily from
     beginning to end.  That is, we construct the list in order, always picking
     the longest subtoken in our vocabulary that matches a prefix of the
     remaining portion of the encoded token.

  4. Concatenate these lists.  This concatenation is invertible due to the
     fact that the trailing underscores indicate when one list is finished.

  """

  def __init__(self, filename=None):
    """Initialize and read from a file, if provided."""
    self._alphabet = set()
    if filename is not None:
      self._load_from_file(filename)
    super(SubwordTextEncoder, self).__init__(num_reserved_ids=None)

  def encode(self, raw_text):
    """Converts a native string to a list of subtoken ids.

    Args:
      raw_text: a native string.
    Returns:
      a list of integers in the range [0, vocab_size)
    """
    return self._tokens_to_subtokens(tokenizer.encode(
        native_to_unicode(raw_text)))

  def decode(self, subtokens):
    """Converts a sequence of subtoken ids to a native string.

    Args:
      subtokens: a list of integers in the range [0, vocab_size)
    Returns:
      a native string
    """
    return unicode_to_native(tokenizer.decode(
        self._subtokens_to_tokens(subtokens)))

  @property
  def vocab_size(self):
    """The subtoken vocabulary size."""
    return len(self._all_subtoken_strings)

  def _tokens_to_subtokens(self, tokens):
    """Converts a list of tokens to a list of subtoken ids.

    Args:
      tokens: a list of strings.
    Returns:
      a list of integers in the range [0, vocab_size)
    """
    ret = []
    for token in tokens:
      ret.extend(self._escaped_token_to_subtokens(self._escape_token(token)))
    return ret

  def _subtokens_to_tokens(self, subtokens):
    """Converts a list of subtoken ids to a list of tokens.

    Args:
      subtokens: a list of integers in the range [0, vocab_size)
    Returns:
      a list of strings.
    """
    concatenated = "".join(
        [self._subtoken_to_subtoken_string(s) for s in subtokens])
    split = concatenated.split("_")
    return [self._unescape_token(t + "_") for t in split if t]

  def _subtoken_to_subtoken_string(self, subtoken):
    """Subtoken_String (string) corresponding to the given subtoken (id)."""
    if 0 <= subtoken < self.vocab_size:
      return self._all_subtoken_strings[subtoken]
    return u""

  def _escaped_token_to_subtokens(self, escaped_token):
    """Converts an escaped token string to a list of subtokens.

    Args:
      escaped_token: an escaped token
    Returns:
      a list of one or more integers.
    """
    ret = []
    pos = 0
    lesc = len(escaped_token)
    while pos < lesc:
      end = min(lesc, pos + self._max_subtoken_len)
      while end > pos:
        subtoken = self._subtoken_string_to_id.get(escaped_token[pos:end], -1)
        if subtoken != -1:
          break
        end -= 1
      assert end > pos
      ret.append(subtoken)
      pos = end

    return ret

  @classmethod
  def build_to_target_size(cls,
                           target_size,
                           token_counts,
                           min_val,
                           max_val,
                           num_iterations=4):
    """Builds a SubwordTextEncoder that has `vocab_size` near `target_size`.

    Uses simple recursive binary search to find a `min_count` value that most
    closely matches the `target_size`.

    Args:
      target_size: desired vocab_size to approximate.
      token_counts: a dictionary of string to int.
      min_val: an integer - lower bound for `min_count`.
      max_val: an integer - upper bound for `min_count`.
      num_iterations: an integer.  how many iterations of refinement.

    Returns:
      a SubwordTextEncoder instance.
    """
    def bisect(min_val, max_val):
      """Bisection to find the right size."""
      present_count = (max_val + min_val) // 2
      tf.logging.info("Trying min_count %d" % present_count)
      subtokenizer = cls()
      subtokenizer.build_from_token_counts(token_counts,
                                           present_count, num_iterations)
      if min_val >= max_val or subtokenizer.vocab_size == target_size:
        return subtokenizer

      if subtokenizer.vocab_size > target_size:
        other_subtokenizer = bisect(present_count + 1, max_val)
      else:
        other_subtokenizer = bisect(min_val, present_count - 1)

      if other_subtokenizer is None:
        return subtokenizer

      if (abs(other_subtokenizer.vocab_size - target_size) <
          abs(subtokenizer.vocab_size - target_size)):
        return other_subtokenizer
      return subtokenizer

    return bisect(min_val, max_val)

  def build_from_token_counts(self,
                              token_counts,
                              min_count,
                              num_iterations=4,
                              num_reserved_ids=NUM_RESERVED_TOKENS):
    """Train a SubwordTextEncoder based on a dictionary of word counts.

    Args:
      token_counts: a dictionary of Unicode strings to int.
      min_count: an integer - discard subtokens with lower counts.
      num_iterations: an integer.  how many iterations of refinement.
      num_reserved_ids: an integer.  how many ids to reserve for special tokens.
    """
    # first determine the alphabet to include all characters with count at
    # least min_count in the dataset.
    char_counts = defaultdict(int)
    for token, count in six.iteritems(token_counts):
      for c in token:
        char_counts[c] += count
    self._alphabet = set()
    for c, count in six.iteritems(char_counts):
      if count >= min_count:
        self._alphabet.add(c)
    # Make sure all characters needed for escaping are included
    for c in u"\\_;0123456789":
      self._alphabet.add(c)

    # We build iteratively.  On each iteration, we segment all the words,
    # then count the resulting potential subtokens, keeping the ones
    # with high enough counts for our new vocabulary.
    if min_count < 1:
      min_count = 1
    for i in xrange(num_iterations):
      tf.logging.info("Iteration {0}".format(i))
      counts = defaultdict(int)
      for token, count in six.iteritems(token_counts):
        escaped_token = self._escape_token(token)
        # we will count all tails of the escaped_token, starting from boundaries
        # determined by our current segmentation.
        if i == 0:
          starts = xrange(len(escaped_token))
        else:
          subtokens = self._escaped_token_to_subtokens(escaped_token)
          pos = 0
          starts = []
          for subtoken in subtokens:
            starts.append(pos)
            pos += len(self._all_subtoken_strings[subtoken])
        for start in starts:
          for end in xrange(start + 1, len(escaped_token) + 1):
            subtoken_string = escaped_token[start:end]
            counts[subtoken_string] += count
      # Make sure all characters needed for escaping are included
      for c in self._alphabet:
        counts[c] += min_count
      # Array of sets of candidate subtoken strings, by length
      len_to_subtoken_strings = []
      for subtoken_string, count in six.iteritems(counts):
        lsub = len(subtoken_string)
        if count >= min_count:
          # Add this subtoken string to its length set
          while len(len_to_subtoken_strings) <= lsub:
            len_to_subtoken_strings.append(set())
          len_to_subtoken_strings[lsub].add(subtoken_string)
      new_subtoken_strings = []
      # consider the candidates longest to shortest, so that if we accept
      # a longer subtoken string, we can decrement the counts of its prefixes.
      for lsub in reversed(range(1, len(len_to_subtoken_strings))):
        subtoken_strings = len_to_subtoken_strings[lsub]
        for subtoken_string in subtoken_strings:
          count = counts[subtoken_string]
          if count >= min_count:
            new_subtoken_strings.append((count, subtoken_string))
            for l in xrange(1, lsub):
              counts[subtoken_string[:l]] -= count
      # Sort in decreasing order by count
      new_subtoken_strings.sort(reverse=True)
      # Now we have a candidate vocabulary
      old_alphabet = self._alphabet
      self._init_from_list([u""] * num_reserved_ids +
                           [p[1] for p in new_subtoken_strings])
      assert old_alphabet == self._alphabet
      tf.logging.info("vocab_size = %d" % self.vocab_size)

    original = "This sentence was encoded by the SubwordTextEncoder."
    encoded = self.encode(original)
    print(encoded)
    print([self._subtoken_to_subtoken_string(s) for s in encoded])
    decoded = self.decode(encoded)
    print(decoded)
    assert decoded == original

  def dump(self):
    """Debugging dump of the current subtoken vocabulary."""
    subtoken_strings = [(i, s)
                        for s, i in six.iteritems(self._subtoken_string_to_id)]
    print(u", ".join(u"{0} : '{1}'".format(i, s)
                     for i, s in sorted(subtoken_strings)))

  def _init_from_list(self, subtoken_strings):
    """Initialize from a list of subtoken strings."""
    self._all_subtoken_strings = subtoken_strings
    # we remember the maximum length of any subtoken to avoid having to
    # check arbitrarily long strings.
    self._max_subtoken_len = max([len(s) for s in subtoken_strings])
    self._subtoken_string_to_id = {
        s: i for i, s in enumerate(subtoken_strings) if s}
    self._alphabet = set([c for c in subtoken_strings if len(c) == 1])

  def _load_from_file(self, filename):
    """Load from a file."""
    subtoken_strings = []
    with tf.gfile.Open(filename) as f:
      for line in f:
        subtoken_strings.append(native_to_unicode(line.strip()[1:-1]))
    self._init_from_list(subtoken_strings)

  def store_to_file(self, filename):
    with tf.gfile.Open(filename, "w") as f:
      for subtoken_string in self._all_subtoken_strings:
        f.write("'" + unicode_to_native(subtoken_string) + "'\n")

  def _escape_token(self, token):
    """Escape away underscores and OOV characters and append '_'.

    This allows the token to be experessed as the concatenation of a list
    of subtokens from the vocabulary.  The underscore acts as a sentinel
    which allows us to invertibly concatenate multiple such lists.

    Args:
      token: a unicode string
    Returns:
      escaped_token: a unicode string
    """
    assert isinstance(token, six.text_type)
    token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u") + u"_"
    ret = u""
    for c in token:
      if c in self._alphabet and c != u"\n":
        ret += c
      else:
        ret += u"\\%d;" % ord(c)
    return ret

  def _unescape_token(self, escaped_token):
    """Inverse of _escape_token().

    Args:
      escaped_token: a unicode string
    Returns:
      token: a unicode string
    """
    def match(m):
      if m.group(1) is not None:
        # Convert '\213;' to unichr(213)
        try:
          return unichr(int(m.group(1)))
        except (ValueError, OverflowError) as _:
          return ""
      # Convert '\u' to '_' and '\\' to '\'
      return u"_" if m.group(0) == u"\\u" else u"\\"
    # Cut off the trailing underscore and apply the regex substitution
    return _UNESCAPE_REGEX.sub(match, escaped_token[:-1])
