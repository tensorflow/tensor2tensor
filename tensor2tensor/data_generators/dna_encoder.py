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

"""Encoders for DNA data.

* DNAEncoder: ACTG strings to ints and back
* DelimitedDNAEncoder: for delimited subsequences
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import text_encoder


class DNAEncoder(text_encoder.TextEncoder):
  """ACTG strings to ints and back. Optionally chunks bases into single ids.

  To use a different character set, subclass and set BASES to the char set. UNK
  and PAD must not appear in the char set, but can also be reset.

  Uses 'N' as an unknown base.
  """
  BASES = list("ACTG")
  UNK = "N"
  PAD = "0"

  def __init__(self,
               chunk_size=1,
               num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS):
    super(DNAEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
    # Build a vocabulary of chunks of size chunk_size
    self._chunk_size = chunk_size
    tokens = self._tokens()
    tokens.sort()
    ids = range(self._num_reserved_ids, len(tokens) + self._num_reserved_ids)
    self._ids_to_tokens = dict(zip(ids, tokens))
    self._tokens_to_ids = dict(zip(tokens, ids))

  def _tokens(self):
    chunks = []
    for size in range(1, self._chunk_size + 1):
      c = itertools.product(self.BASES + [self.UNK], repeat=size)
      num_pad = self._chunk_size - size
      padding = (self.PAD,) * num_pad
      c = [el + padding for el in c]
      chunks.extend(c)
    return chunks

  @property
  def vocab_size(self):
    return len(self._ids_to_tokens) + self._num_reserved_ids

  def encode(self, s):
    bases = list(s)
    extra = len(bases) % self._chunk_size
    if extra > 0:
      pad = [self.PAD] * (self._chunk_size - extra)
      bases.extend(pad)
    assert (len(bases) % self._chunk_size) == 0
    num_chunks = len(bases) // self._chunk_size
    ids = []
    for chunk_idx in xrange(num_chunks):
      start_idx = chunk_idx * self._chunk_size
      end_idx = start_idx + self._chunk_size
      chunk = tuple(bases[start_idx:end_idx])
      if chunk not in self._tokens_to_ids:
        raise ValueError("Unrecognized token %s" % chunk)
      ids.append(self._tokens_to_ids[chunk])
    return ids

  def decode(self, ids):
    bases = []
    for idx in ids:
      if idx >= self._num_reserved_ids:
        chunk = self._ids_to_tokens[idx]
        if self.PAD in chunk:
          chunk = chunk[:chunk.index(self.PAD)]
      else:
        chunk = [text_encoder.RESERVED_TOKENS[idx]]
      bases.extend(chunk)
    return "".join(bases)


class DelimitedDNAEncoder(DNAEncoder):
  """DNAEncoder for delimiter separated subsequences.

  Uses ',' as default delimiter.
  """

  def __init__(self, delimiter=",", **kwargs):
    self._delimiter = delimiter
    self._delimiter_key = tuple(self._delimiter)
    super(DelimitedDNAEncoder, self).__init__(**kwargs)

  @property
  def delimiter(self):
    return self._delimiter

  def _tokens(self):
    return super(DelimitedDNAEncoder, self)._tokens() + [self._delimiter_key]

  def encode(self, delimited_string):
    ids = []
    for s in delimited_string.split(self.delimiter):
      ids.extend(super(DelimitedDNAEncoder, self).encode(s))
      ids.append(self._tokens_to_ids[self._delimiter_key])
    return ids[:-1]
