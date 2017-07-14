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

"""Data generator for Wikipedia title to article dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bz2
from collections import defaultdict
import os

# Dependency imports

import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer

import tensorflow as tf


# End-of-sentence marker (should correspond to the position of EOS in the
# RESERVED_TOKENS list in text_encoder.py)
EOS = 1


def _maybe_download_corpus(tmp_dir):
  """Download corpus if necessary.

  Args:
    tmp_dir: directory containing dataset.

  Returns:
    filepath of the downloaded corpus file.
  """
  corpus_url = ("https://dumps.wikimedia.org/enwiki/20170620/"
                "enwiki-20170620-pages-articles-multistream.xml.bz2")
  corpus_filename = os.path.basename(corpus_url)
  corpus_filepath = os.path.join(tmp_dir, corpus_filename)
  if not os.path.exists(corpus_filepath):
    generator_utils.maybe_download(tmp_dir, corpus_filename, corpus_url)
  return corpus_filepath


def page_generator(tmp_dir, max_docs=None):
  doc = u""
  count = 0
  corpus_filepath = _maybe_download_corpus(tmp_dir)
  for line in bz2.BZ2File(corpus_filepath, "r"):
    line = unicode(line, "utf-8") if six.PY2 else line.decode("utf-8")
    if not doc and line != u"  <page>\n":
      continue
    doc += line
    if line == u"  </page>\n":
      yield doc
      doc = u""
      count += 1
      if max_docs and count >= max_docs:
        break


def _page_title(page):
  start_pos = page.find(u"<title>")
  end_pos = page.find(u"</title>")
  assert start_pos != -1
  assert end_pos != -1
  start_pos += len(u"<title>")
  return page[start_pos:end_pos]


def _get_or_build_subword_text_encoder(tmp_dir):
  """Builds a SubwordTextEncoder based on the corpus.

  Args:
    tmp_dir: a string

  Returns:
    a SubwordTextEncoder.
  """
  filename = os.path.join(tmp_dir, "wiki_32k.subword_text_encoder")
  if tf.gfile.Exists(filename):
    return text_encoder.SubwordTextEncoder(filename)
  token_counts = defaultdict(int)
  for page in page_generator(tmp_dir, max_docs=1000):
    tokens = tokenizer.encode(page)
    tokens = set(tokens)
    for tok in tokens:
      token_counts[tok] += 1
  new_token_counts = defaultdict(int)
  for token, count in six.iteritems(token_counts):
    if count >= 3:
      new_token_counts[token] = count
  ret = text_encoder.SubwordTextEncoder()
  ret.build_from_token_counts(new_token_counts, min_count=10)
  ret.store_to_file(filename)
  return ret


def generator(tmp_dir, train):
  """Generator for lm1b sentences.

  Args:
    tmp_dir: a string.
    train: a boolean.

  Yields:
    A dictionary {"inputs": [<subword ids>], "targets": [<subword ids>]}
  """
  assert train
  encoder = _get_or_build_subword_text_encoder(tmp_dir)
  for page in page_generator(tmp_dir):
    title = _page_title(page)
    encoded = encoder.encode(page) + [EOS]
    encoded_title = encoder.encode(title) + [EOS]
    yield {"inputs": encoded_title, "targets": encoded}
