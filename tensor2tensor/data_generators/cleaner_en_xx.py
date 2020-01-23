# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

# encoding=UTF-8
"""An unsophisticated data cleaner for en-.. sentence translation pairs.

This pattern-based English-... cleaner aims fairly aggressively for clean
sentence-like pairs. It discards pairs if the English member has signs of
non-sentence noise or origin, e.g., lacks expected punctuation or has suspicious
character sequences. It also simplistically detects and corrects some missing
sentence breaks. It makes minimal assumptions about the other language, mainly
that its sentences can end in one of '.!?' and that its sentences can start
with an ASCII capital letter.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import itertools
import re

from tensor2tensor.data_generators import text_encoder

import tensorflow.compat.v1 as tf


_RE_GOOD_S_START = re.compile(r'^["“”]?[A-Z]')
_RE_GOOD_S_END = re.compile(r'\w[.?!]["”]?$', re.UNICODE)

_RE_LABEL_COLON = re.compile(r'^\w+\.?( \w+)?: ', re.UNICODE)
_RE_DIGIT_SPACE_DIGIT = re.compile(r'\d +\d', re.UNICODE)
_RE_ALL_CAP_WORDS = re.compile(r'^[A-Z]\S*(\s+[A-Z]\S+)+\s*$')

_RE_DQ_ONE = re.compile(r'^[^"“”]*["“”][^"“”]*$')
_RE_DQ_INITIAL = re.compile(r'^["“”]([^"“”]+)$')
_RE_DQ_FINAL = re.compile(r'^[^"“”]+["“”]$')
_RE_DQ_LINE = re.compile(r'^["“”].*["“”]$')

_RE_DQ_MANY = re.compile(r'(["“”].*){3,}')
_RE_SQ_MANY = re.compile(r'''(['‘’][^st].*){3,}''')
_RE_CHARS_QQ = re.compile(r'''["“”'‘’]\s*["“”'‘’]''')
_RE_SPACE_PUNCT_SPACE = re.compile(r'''\s["“”'‘’,:;]\s''')

_RE_COPYRIGHT = re.compile(r'©|^Copyright|^\(C\)')
_RE_UNMATCHED_PAREN_LEFT = re.compile(r'[(][^)]*$')
_RE_UNMATCHED_PAREN_RIGHT = re.compile(r'^[^(]*[)]')
_RE_TAGLINE_CITY = re.compile(r'^[A-Z]{2,}(\s+[A-Z]+)*\s+-')
_RE_CHARS_UPPER_UNDERSCORE = re.compile(r'^[A-Z]+[a-z]*_')


def paracrawl_v3_pairs(paracrawl_file):
  """Generates raw (English, other) pairs from a ParaCrawl V3.0 data file.

  Args:
    paracrawl_file: A ParaCrawl V3.0 en-.. data file.
  Yields:
    Pairs of (sentence_en, sentence_xx), as Unicode strings.
  Raises:
    StopIteration: If the file ends while this method is in the middle of
        creating a translation pair.
  """
  raw_sentences = _raw_sentences(paracrawl_file)
  for s_en in raw_sentences:
    try:
      s_xx = next(raw_sentences)
      if s_en and s_xx:  # Prevent empty string examples.
        yield s_en, s_xx
    except StopIteration:
      tf.logging.error(
          'Unmatched final sentence while reading in sentence pairs: [%s]',
          s_en)


def _raw_sentences(paracrawl_file):
  """Generates Unicode strings, one for each <seg> in a ParaCrawl data file.

  Also decodes some of the most common HTML entities found in ParaCrawl data.

  Args:
    paracrawl_file: A ParaCrawl V3.0 en-.. data file.
  Yields:
    One Unicode string for each <seg> element in the ParaCrawl data file.
  """
  for line_utf8 in paracrawl_file:
    line_uni = line_utf8.decode('UTF-8')
    text_match = re.match(r' +<seg>(.*)</seg>$', line_uni)
    if text_match:
      txt = text_match.group(1)
      txt = re.sub(r'&amp;', r'&', txt)
      txt = re.sub(r'& ?amp;', r'&', txt)
      txt = re.sub(r'& ?apos;', r"'", txt)
      txt = re.sub(r'& ?quot;', r'"', txt)
      txt = re.sub(r'& ?lt;', r'<', txt)
      txt = re.sub(r'& ?gt;', r'>', txt)
      yield txt


def clean_en_xx_pairs(en_xx_pairs):
  """Generates a cleaned-up stream of (English, other) translation pairs.

  Cleaning includes both filtering and simplistic sentence splitting, with
  minimal assumptions on the non-English pair member: (1) All filtering is
  done based on the English member of the pair, and (2) sentence splitting
  assumes only that sentences can end with one of '.!?' and begin with an
  ASCII uppercase letter. Input pairs that would get split into different
  numbers of sentences (e.g., three English sentences vs. two German ones) are
  discarded.

  Args:
    en_xx_pairs: A stream (iterable) of Unicode string pairs. Each item in the
        stream should be a (sentence_en, sentence_xx) pair.
  Yields:
    Cleaned-up (sentence_en, sentence_xx) pairs.
  """
  for s1, s2 in en_xx_pairs:
    if _regex_filter(s1):
      continue
    s1_list, s2_list = _split_sentences(s1, s2)
    if len(s1_list) != len(s2_list):
      continue  # discard this pair
    elif len(s1_list) == 1:
      yield s1, s2
    else:
      for s1_subsentence, s2_subsentence in itertools.izip(s1_list, s2_list):
        if _regex_filter(s1_subsentence):
          continue
        yield s1_subsentence, s2_subsentence


def _regex_filter(sentence):
  return (not _is_match(sentence, _RE_GOOD_S_START)
          or not _is_match(sentence, _RE_GOOD_S_END)
          or _is_match(sentence, _RE_LABEL_COLON)
          or _is_match(sentence, _RE_DIGIT_SPACE_DIGIT)
          or _is_match(sentence, _RE_DQ_ONE)
          or _is_match(sentence, _RE_DQ_INITIAL)
          or _is_match(sentence, _RE_DQ_FINAL)
          or _is_match(sentence, _RE_DQ_LINE)
          or _is_match(sentence, _RE_DQ_MANY)
          or _is_match(sentence, _RE_SQ_MANY)
          or _is_match(sentence, _RE_CHARS_QQ)
          or _is_match(sentence, _RE_SPACE_PUNCT_SPACE)
          or _is_match(sentence, _RE_COPYRIGHT)
          or _is_match(sentence, _RE_UNMATCHED_PAREN_LEFT)
          or _is_match(sentence, _RE_UNMATCHED_PAREN_RIGHT)
          or _is_match(sentence, _RE_TAGLINE_CITY)
          or _is_match(sentence, _RE_CHARS_UPPER_UNDERSCORE))


def _is_match(sentence, regex):
  return regex.search(sentence)


def _split_sentences(s1, s2):
  s1 = text_encoder.native_to_unicode(s1)
  s2 = text_encoder.native_to_unicode(s2)
  s1 = re.sub(r'(\w[A-Z]|[0-9a-z])([.!?]) ([A-Z])', r'\1\2__|__\3', s1)
  s2 = re.sub(r'([^0-9][.!?]) ([A-Z])', r'\1__|__\2', s2)
  s1_subsentences = s1.split('__|__')
  s2_subsentences = s2.split('__|__')
  return s1_subsentences, s2_subsentences
