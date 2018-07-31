# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Data generator for pointer-generator for word transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow as tf


@registry.register_problem
class Text2textCopyableTokens(text_problems.Text2textTmpdirTokens):
  """Allows training a variant of Text2textTmpdirTokens that supports copying.

  Handling the case where the input contains OOV tokens. Store a temporary vocab
  ID for source OOV, so that the decoder can directly copy from the input.
  Uses TokenTextEncoderOov as the vocab encoder.
  """

  def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
    vocab_filename = os.path.join(data_dir, self.vocab_filename)
    encoder = TokenTextEncoderOov(
        vocab_filename, replace_oov=self.oov_token)
    return encoder

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    return self.text2text_generate_encoded_oovs(
        generator, encoder, has_inputs=self.has_inputs)

  def text2text_generate_encoded_oovs(self,
                                      sample_generator,
                                      vocab,
                                      targets_vocab=None,
                                      has_inputs=True):
    """Encode Text2Text samples from the generator with the vocab."""
    targets_vocab = targets_vocab or vocab
    for sample in sample_generator:
      if has_inputs:
        (sample["inputs"], sample["inputs_extend"], source_oovs,
         _) = vocab.encode(sample["inputs"])
        sample["inputs"].append(text_encoder.EOS_ID)
        sample["inputs_extend"].append(text_encoder.EOS_ID)
      # need to pass the source OOV tokens to the target encoder
      sample["targets"], sample["targets_extend"] = targets_vocab.encode_target(
          sample["targets"], source_oovs)
      sample["targets"].append(text_encoder.EOS_ID)
      sample["targets_extend"].append(text_encoder.EOS_ID)
      yield sample

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "inputs_extend": tf.VarLenFeature(tf.int64),
        "targets": tf.VarLenFeature(tf.int64),
        "targets_extend": tf.VarLenFeature(tf.int64)
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


class TokenTextEncoderOov(text_encoder.TokenTextEncoder):
  """Encoder based on a user-supplied vocabulary (file or list).

  This encoder extends over TokenTextEncoder by additionally assigning distinct
  temporary IDs to OOV tokens appearing in the source sequence. This facilitates
  decoding with the pointer-generator mechanism using word-based tokenization.

  NOTE: TokenTextEncoderOov does not conform to the TextEncoder API; it changes
  the signature of encode and decode.
  """

  def encode(self, s):
    """Converts a space-separated string of tokens to lists of ids.

    Also store temporary vocabulary IDs for source OOV tokens. OOVs are
    represented by their temporary OOV number. E.g., if the vocabulary size
    is 50k and the source has 3 OOVs, then these temporary OOV numbers will
    be 50000, 50001, 50002.

    Args:
      s: human-readable string to be converted.

    Returns:
      ids: list of integers
      ids_extend: list of integers including extended temporary vocab IDs for
      source OOVs.
      oovs: A dict storing source OOV words, used for the decoder to copy. The
      key is OOV word, and the value is the order they appear in the source,
      starting from 0.
      source_oov_id_to_token: a list of source OOV tokens, in the same order as
      they appear in the source.
    """
    sentence = s
    tokens = sentence.strip().split()
    ids = []
    ids_extend = []
    oovs = {}
    for t in tokens:
      if t in self._token_to_id:
        ids.append(self._token_to_id[t])
        ids_extend.append(self._token_to_id[t])
      else:
        next_oov_id = len(oovs)
        oov_num = oovs.get(t, next_oov_id)
        if oov_num == next_oov_id:
          oovs[t] = oov_num
        ids_extend.append(self.vocab_size + oov_num)
        ids.append(self._token_to_id[self._replace_oov])
    source_oov_id_to_token = [""] * len(oovs)
    for oov in oovs:
      source_oov_id_to_token[oovs[oov]] = oov
    if self._reverse:
      return ids[::-1], ids_extend[::-1], oovs, source_oov_id_to_token
    else:
      return ids, ids_extend, oovs, source_oov_id_to_token

  def encode_target(self, target, source_oovs):
    """Converts a space-separated string of tokens to lists of ids.

    Also store a version of extened vocabulary IDs.
    For target OOVs that are in the source, encode them using the temporary
    vocab IDs.
    For target OOVs not in the source, encode them as <UNK>

    Args:
      target: target string
      source_oovs: source OOV words stored in dict, key is the word, value is
      the order in which they appear in the source starting from 0

    Returns:
      ids: list of integers
      ids_extend: list of integers including extended vocabulary IDs.
    """
    tokens = target.strip().split()
    ids = []
    ids_extend = []
    for t in tokens:
      if t in self._token_to_id:
        i = self._token_to_id[t]
        ids.append(i)
        ids_extend.append(i)
      else:
        ids.append(self._token_to_id[self._replace_oov])
        if t in source_oovs:
          vocab_idx = self.vocab_size + source_oovs[t]
          ids_extend.append(vocab_idx)
        else:
          ids_extend.append(self._token_to_id[self._replace_oov])
    if self._reverse:
      return ids[::-1], ids_extend[::-1]
    else:
      return ids, ids_extend

  def decode_oov(self, ids, source_oov):
    return " ".join(self.decode_list_oov(ids, source_oov))

  def decode_list_oov(self, ids, source_oov_id_to_token):
    """decode ids back to tokens, considering OOVs temporary IDs.

    Args:
      ids: vocab ids. Could possibly include source temporary OOV ID starting
      from vocab_size.
      source_oov_id_to_token: a list of source OOV tokens, with the order the
      same as they appear in the source.

    Returns:
      decoded tokens, possibly including source OOV tokens.

    """
    seq = reversed(ids) if self._reverse else ids
    tokens = []
    for cur_id in seq:
      if cur_id in self._id_to_token:
        tokens.append(self._id_to_token[cur_id])
      else:
        tokens.append(source_oov_id_to_token[cur_id - self.vocab_size])
    return tokens
