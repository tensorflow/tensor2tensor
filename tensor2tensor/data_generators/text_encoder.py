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

# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import tokenizer

import tensorflow as tf

# Reserved tokens for things like padding and EOS symbols.
PAD = '<pad>'
EOS = '<EOS>'
RESERVED_TOKENS = [PAD, EOS]
if six.PY2:
  RESERVED_TOKENS_BYTES = RESERVED_TOKENS
else:
  RESERVED_TOKENS_BYTES = [bytes(PAD, 'ascii'), bytes(EOS, 'ascii')]


class TextEncoder(object):
  """Base class for converting from ints to/from human readable strings."""

  def __init__(self, num_reserved_ids=2):
    self._num_reserved_ids = num_reserved_ids

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
    return ' '.join([str(d) for d in decoded_ids])

  @property
  def vocab_size(self):
    raise NotImplementedError()


class ByteTextEncoder(TextEncoder):
  """Encodes each byte to an id. For 8-bit strings only."""

  def encode(self, s):
    numres = self._num_reserved_ids
    if six.PY2:
      return [ord(c) + numres for c in s]
    # Python3: explicitly convert to UTF-8
    return [c + numres for c in s.encode('utf-8')]

  def decode(self, ids):
    numres = self._num_reserved_ids
    decoded_ids = []
    int2byte = six.int2byte
    for id_ in ids:
      if 0 <= id_ < numres:
        decoded_ids.append(RESERVED_TOKENS_BYTES[int(id_)])
      else:
        decoded_ids.append(int2byte(id_ - numres))
    if six.PY2:
      return ''.join(decoded_ids)
    # Python3: join byte arrays and then decode string
    return b''.join(decoded_ids).decode('utf-8')

  @property
  def vocab_size(self):
    return 2**8 + self._num_reserved_ids


class TokenTextEncoder(TextEncoder):
  """Encoder based on a user-supplied vocabulary."""

  def __init__(self, vocab_filename, reverse=False, num_reserved_ids=2):
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
    return ' '.join([self._safe_id_to_token(i) for i in seq])

  @property
  def vocab_size(self):
    return len(self._id_to_token)

  def _safe_id_to_token(self, idx):
    return self._id_to_token.get(idx, 'ID_%d' % idx)

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
  """Class for breaking tokens into subtokens.

  Invertibly encodes a string as a sequence of subtokens from a limited
  vocabulary.

  A SubwordTextEncoder is built from a corpus (so it is tailored to the text in
  the corpus), and stored to a file. See text_encoder_build_subword.py.

  It can then be loaded and used to encode/decode any text.
  """

  def __init__(self, filename=None, num_reserved_ids=2):
    self._tokenizer = tokenizer.Tokenizer()
    if filename is not None:
      # Read from a file.
      self._load_from_file(filename)

    super(SubwordTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)

  def encode(self, raw_text):
    """Converts a string to a list of subtoken ids.

    Args:
      raw_text: a string.
    Returns:
      a list of integers in the range [0, vocab_size)
    """
    return self._tokens_to_subtokens(self._tokenizer.encode(raw_text))

  def decode(self, subtokens):
    """Converts a sequence of subtoken ids to a string.

    Args:
      subtokens: a list of integers in the range [0, vocab_size)
    Returns:
      a string
    """
    return self._tokenizer.decode(self._subtokens_to_tokens(subtokens))

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
    concatenated = ''.join(
        [self.subtoken_to_subtoken_string(s) for s in subtokens])
    split = concatenated.split('_')
    return [self._unescape_token(t + '_') for t in split if t]

  def subtoken_to_subtoken_string(self, subtoken):
    """Subtoken_String (string) corresponding to the given subtoken (id)."""
    if 0 <= subtoken < self.vocab_size:
      subtoken_string = self._all_subtoken_strings[subtoken]
      if subtoken_string:
        return subtoken_string
    if 0 <= subtoken < self._num_reserved_ids:
      return '%s_' % RESERVED_TOKENS[subtoken]
    return 'ID%d_' % subtoken

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
      end = lesc
      while end > pos:
        subtoken = self._subtoken_string_to_id.get(escaped_token[pos:end], -1)
        if subtoken != -1:
          break
        end -= 1
      if end > pos:
        ret.append(subtoken)
        pos = end
      else:
        # No subtoken in the vocabulary matches escaped_token[pos].
        # This can happen if the token contains a Unicode character
        # that did not occur in the vocabulary training set.
        # The id self.vocab_size - 1 is decoded as Unicode uFFFD,
        # REPLACEMENT_CHARACTER.
        ret.append(self.vocab_size - 1)
        # Ensure that the outer loop continues
        pos += 1
    return ret

  @classmethod
  def alphabet(cls, token_counts):
    """Return the set of Unicode characters that appear in the tokens"""
    alphabet_set = set()
    for token in six.iterkeys(token_counts):
      alphabet_set |= set(token)
    return alphabet_set

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
      store_filename: a string - where to write the vocabulary.
      min_val: an integer - lower bound for `min_count`.
      max_val: an integer - upper bound for `min_count`.
      num_iterations: an integer.  how many iterations of refinement.

    Returns:
      a SubwordTextEncoder instance.
    """

    # Calculate the alphabet, i.e. the set of all Unicode characters
    # that appear in the tokens
    alphabet_set = cls.alphabet(token_counts)
    tf.logging.info('Alphabet contains %d characters' % len(alphabet_set))

    def bisect(min_val, max_val):
      present_count = (max_val + min_val) // 2
      tf.logging.info('Trying min_count %d' % present_count)
      subtokenizer = cls()
      subtokenizer.build_from_token_counts(token_counts, alphabet_set,
                                           present_count, num_iterations)

      if min_val >= max_val or subtokenizer.vocab_size == target_size:
        return subtokenizer
      if subtokenizer.vocab_size > target_size:
        other_subtokenizer = bisect(present_count + 1, max_val)
      else:
        other_subtokenizer = bisect(min_val, present_count - 1)
      if (abs(other_subtokenizer.vocab_size - target_size) <
          abs(subtokenizer.vocab_size - target_size)):
        return other_subtokenizer
      else:
        return subtokenizer

    return bisect(min_val, max_val)

  def build_from_token_counts(self,
                              token_counts,
                              alphabet_set,
                              min_count,
                              num_iterations=4):
    """Train a SubwordTextEncoder based on a dictionary of word counts.

    Args:
      token_counts: a dictionary of Unicode strings to int.
      alphabet_set: the set of Unicode characters that appear in the tokens.
      min_count: an integer - discard subtokens with lower counts.
      num_iterations: an integer.  how many iterations of refinement.
    """
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
      # Array of sets of candidate subtoken strings, by length
      len_to_subtoken_strings = []
      for subtoken_string, count in six.iteritems(counts):
        lsub = len(subtoken_string)
        # All subtoken strings of length 1 are automatically included
        # later, so we don't need to consider them here
        if count < min_count or lsub <= 1:
          continue
        # Add this subtoken string to its length set
        while len(len_to_subtoken_strings) <= lsub:
          len_to_subtoken_strings.append(set())
        len_to_subtoken_strings[lsub].add(subtoken_string)
      new_subtoken_strings = []
      # consider the candidates longest to shortest, so that if we accept
      # a longer subtoken string, we can decrement the counts of its prefixes.
      for subtoken_strings in reversed(len_to_subtoken_strings[2:]):
        for subtoken_string in subtoken_strings:
          count = counts[subtoken_string]
          if count < min_count:
            continue
          new_subtoken_strings.append((count, subtoken_string))
          for l in xrange(1, len(subtoken_string)):
            counts[subtoken_string[:l]] -= count
      # Sort what we've got so far in decreasing order by count
      new_subtoken_strings.sort(reverse = True)
      # Add the alphabet set at the end of the vocabulary list
      for char in alphabet_set:
        new_subtoken_strings.append((0, char))
      # Also include the Unicode REPLACEMENT CHARACTER to use
      # when encountering previously unseen Unicode characters
      # in the input (i.e. input external to the tokenizer training
      # set, which may thus contain characters not in the alphabet_set).
      # This must be the last entry in the subtoken vocabulary list.
      new_subtoken_strings.append((0, u'\uFFFD'))
      # Now we have a candidate vocabulary
      self._init_from_list([u''] * self._num_reserved_ids +
                           [p[1] for p in new_subtoken_strings])
      tf.logging.info('vocab_size = %d' % self.vocab_size)

    #original = 'This sentence was encoded by the SubwordTextEncoder.'
    #encoded = self.encode(original)
    #print(encoded)
    #print([self.subtoken_to_subtoken_string(s) for s in encoded])
    #decoded = self.decode(encoded)
    #print(decoded)
    #assert decoded == original

  def dump(self):
    """ Debugging dump of the current subtoken vocabulary """
    subtoken_strings = [(i, s) for s, i in six.iteritems(self._subtoken_string_to_id)]
    print(u", ".join(u"{0} : '{1}'".format(i, s) for i, s in sorted(subtoken_strings)))

  def _init_from_list(self, subtoken_strings):
    """Initialize from a list of subtoken strings."""
    self._all_subtoken_strings = subtoken_strings
    self._subtoken_string_to_id = { s : i for i, s in enumerate(subtoken_strings) if s }

  def _load_from_file(self, filename):
    """Load from a file."""
    subtoken_strings = []
    with tf.gfile.Open(filename) as f:
      for line in f:
        if six.PY2:
          subtoken_strings.append(line.strip()[1:-1].decode('utf-8'))
        else:
          subtoken_strings.append(line.strip()[1:-1])
    self._init_from_list(subtoken_strings)

  def store_to_file(self, filename):
    with tf.gfile.Open(filename, 'w') as f:
      for subtoken_string in self._all_subtoken_strings:
        if six.PY2:
          f.write('\'' + subtoken_string.encode('utf-8') + '\'\n')
        else:
          f.write('\'' + subtoken_string + '\'\n')

  def _escape_token(self, token):
    r"""Translate '\'->'\\' and '_'->'\u', then append '_'.

    Args:
      token: a string
    Returns:
      escaped_token: a string
    """
    return token.replace('\\', '\\\\').replace('_', '\\u') + '_'

  def _unescape_token(self, escaped_token):
    r"""Remove '_' from end, then translate '\\'->'\' and '\u'->'_'.

    Args:
      escaped_token: a string
    Returns:
      token: a string
    """
    assert escaped_token[-1] == '_'
    return escaped_token[:-1].replace('\\u', '_').replace('\\\\', '\\')

  @classmethod
  def get_token_counts(cls, text_filepattern, corpus_max_lines):
    """Read the corpus and compute a dictionary of token counts."""
    tok = tokenizer.Tokenizer()
    lines_read = 0
    filenames = tf.gfile.Glob(text_filepattern)
    for text_filename in filenames:
      with tf.gfile.Open(text_filename) as f:
        for line in f:
          # The tokenizer updates token_counts in encode()
          tok.encode(line.strip())
          lines_read += 1
          if corpus_max_lines > 0 and lines_read > corpus_max_lines:
            return tok.token_counts
    return tok.token_counts
