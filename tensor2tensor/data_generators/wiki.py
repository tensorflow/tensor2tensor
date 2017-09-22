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

"""Data generator for Wikipedia title to article dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

import bz2file

import numpy as np

import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


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
  if not tf.gfile.Exists(corpus_filepath):
    generator_utils.maybe_download(tmp_dir, corpus_filename, corpus_url)
  return corpus_filepath


def page_generator(tmp_dir, max_docs=None):
  doc = u""
  count = 0
  corpus_filepath = _maybe_download_corpus(tmp_dir)
  for line in bz2file.BZ2File(corpus_filepath, "r", buffering=1000000):
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


@registry.register_problem
class LanguagemodelWikiFull32k(problem.Text2TextProblem):
  """A language model on full English Wikipedia."""

  @property
  def is_character_level(self):
    return False

  @property
  def has_inputs(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def num_shards(self):
    return 1000

  @property
  def vocab_name(self):
    return "vocab.wiki"

  @property
  def use_subword_tokenizer(self):
    return True

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def use_train_shards_for_dev(self):
    return True

  def generator(self, data_dir, tmp_dir, _):
    encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        page_generator(tmp_dir, max_docs=10000))
    for page in page_generator(tmp_dir):
      title = _page_title(page)
      encoded = encoder.encode(page) + [EOS]
      encoded_title = encoder.encode(title) + [EOS]
      yield {"inputs": encoded_title, "targets": encoded}


class LanguagemodelWikiScramble(problem.Text2TextProblem):
  """Language modeling on English wikipedia.

  "targets" is a sequence of sequence_length tokens - a fragment of an article.
  "inputs" is a copy of "targets", but with a random scramble_fraction of the
    tokens randomly permuted.

  This dataset is intended to test parallel (non-autoregressive) prediction
  of the target sequence given the input sequence.
  """

  @property
  def sequence_length(self):
    raise NotImplementedError()

  @property
  def scramble_fraction(self):
    raise NotImplementedError()

  @property
  def is_character_level(self):
    return False

  @property
  def has_inputs(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def num_shards(self):
    return 1000

  @property
  def vocab_name(self):
    return "vocab.wiki"

  @property
  def use_subword_tokenizer(self):
    return True

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def use_train_shards_for_dev(self):
    return True

  @property
  def max_cases(self):
    return (2 ** 30) / self.sequence_length

  def scramble(self, seq):
    seq = np.array(seq)
    num_permute = int(self.sequence_length * self.scramble_fraction)
    full_permutation = np.random.permutation(self.sequence_length)
    inverse_full_permutation = np.argsort(full_permutation)
    partial_permutation = np.random.permutation(num_permute)
    seq = seq[full_permutation]
    seq = np.concatenate(
        (seq[:num_permute][partial_permutation], seq[num_permute:]))
    seq = seq[inverse_full_permutation]
    seq = list(seq)
    return seq

  def generator(self, data_dir, tmp_dir, _):
    encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        page_generator(tmp_dir, max_docs=1000))
    case_num = 0
    for page in page_generator(tmp_dir):
      encoded = encoder.encode(page)
      for i in xrange(len(encoded) // self.sequence_length):
        case_num += 1
        if self.max_cases and case_num > self.max_cases:
          return
        targets = encoded[
            i * self.sequence_length:(i + 1) * self.sequence_length]
        inputs = self.scramble(targets)
        yield {"inputs": inputs, "targets": targets}

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.NEG_LOG_PERPLEXITY
    ]


@registry.register_problem
class LanguagemodelWikiScramble128(LanguagemodelWikiScramble):
  """Sequence length 128, 50% scrambed."""

  @property
  def sequence_length(self):
    return 128

  @property
  def scramble_fraction(self):
    return 0.5


@registry.register_problem
class LanguagemodelWikiScramble1k50(LanguagemodelWikiScramble):
  """Sequence length 1024, 50% scrambed."""

  @property
  def sequence_length(self):
    return 1024

  @property
  def scramble_fraction(self):
    return 0.5


@registry.register_problem
class LanguagemodelWikiScramble8k50(LanguagemodelWikiScramble):
  """Sequence length 8192, 50% scrambed."""

  @property
  def sequence_length(self):
    return 8192

  @property
  def scramble_fraction(self):
    return 0.5
