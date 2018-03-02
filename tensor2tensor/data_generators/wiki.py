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

"""Data generator for Wikipedia title to article dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class LanguagemodelWikiXmlV8kL1k(text_problems.ChoppedTextProblem):
  """A language model on English Wikipedia.

  XML dump is chopped arbitrarily into sequences of length 1024 tokens,
  without regard to article boundaries.
  """

  def maybe_prepare_text(self, tmp_dir):
    """Download corpus if necessary, decompress, split into multiple text files.

    Args:
      tmp_dir: directory containing dataset.

    Returns:
      list of filepaths for local text files.
    """
    compressed_filename = os.path.basename(self.corpus_url)
    compressed_filepath = os.path.join(tmp_dir, compressed_filename)
    decompressed_filepath = compressed_filepath[:-4]
    split_file_prefix = decompressed_filepath + "-part-"
    split_filepattern = split_file_prefix + "?????"
    split_files = sorted(tf.gfile.Glob(split_filepattern))
    if not split_files:
      if not tf.gfile.Exists(decompressed_filepath):
        if not tf.gfile.Exists(compressed_filepath):
          generator_utils.maybe_download(
              tmp_dir, compressed_filepath, self.corpus_url)
        assert not subprocess.call(["bunzip2", compressed_filepath])
      assert tf.gfile.Exists(decompressed_filepath)
      assert not subprocess.call([
          "split", "--line-bytes=4M", "--suffix-length=5",
          "--numeric-suffixes", decompressed_filepath, split_file_prefix])
      split_files = sorted(tf.gfile.Glob(split_filepattern))
    assert split_files
    return split_files

  def train_text_filepaths(self, tmp_dir):
    all_files = self.maybe_prepare_text(tmp_dir)
    return [f for i, f in enumerate(all_files) if i % self.dev_fraction != 0]

  def dev_text_filepaths(self, tmp_dir):
    all_files = self.maybe_prepare_text(tmp_dir)
    return [f for i, f in enumerate(all_files) if i % self.dev_fraction == 0]

  @property
  def dev_fraction(self):
    return 5000

  @property
  def corpus_url(self):
    return ("https://archive.org/download/enwiki-20171201/"
            "enwiki-20171201-pages-articles.xml.bz2")

  @property
  def vocab_filename(self):
    return "vocab.wiki_xml.%d" % self.approx_vocab_size

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 1024

  @property
  def max_chars_for_vocab(self):
    """Number of characters of training data to use for generating vocab."""
    # magic number for backwards compatibility
    return 41800829


@registry.register_problem
class LanguagemodelWikiXmlV8kL4k(LanguagemodelWikiXmlV8kL1k):
  """A language model on English Wikipedia.

  XML dump is chopped arbitrarily into sequences of length 4096 tokens,
  without regard to article boundaries.
  """

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 4096


class LanguagemodelWikiScramble(LanguagemodelWikiXmlV8kL1k):
  """Language modeling on English wikipedia.

  "targets" is a sequence of sequence_length tokens - a fragment of an article.
  "inputs" is a copy of "targets", but with a random scramble_fraction of the
    tokens randomly permuted.

  This dataset is intended to test parallel (non-autoregressive) prediction
  of the target sequence given the input sequence.
  """

  def example_generator(self, encoder, tmp_dir, task_id):
    for x in super(LanguagemodelWikiScramble, self).example_generator(
        encoder, tmp_dir, task_id):
      x["inputs"] = self.scramble(x["targets"])
      yield x

  @property
  def scramble_fraction(self):
    raise NotImplementedError()

  @property
  def has_inputs(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def remainder_policy(self):
    """What to do with leftover tokens."""
    return "drop"

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


@registry.register_problem
class LanguagemodelWikiScrambleL128(LanguagemodelWikiScramble):
  """Sequence length 128, 50% scrambed."""

  @property
  def sequence_length(self):
    return 128

  @property
  def scramble_fraction(self):
    return 0.5


@registry.register_problem
class LanguagemodelWikiScrambleL1k(LanguagemodelWikiScramble):
  """Sequence length 1024, 50% scrambed."""

  @property
  def sequence_length(self):
    return 1024

  @property
  def scramble_fraction(self):
    return 0.5


@registry.register_problem
class LanguagemodelWikiNorefV8kL1k(LanguagemodelWikiXmlV8kL1k):
  """A language model on English Wikipedia.

  References and internal links are removed from the raw XML.

  Special pages (non-articles) are dropped.

  This more closely resemples plain text, though there are still some xml
  elements, like tables.

  Each article is prefixed by a line containing the title and length in
  characters - e.g.
  title: "Price of Tea in China" length: 12345
  During inference time, you can forward generate starting with such a header
  in order to obtain a randomly generated article with a given title and
  (approximate) length.

  Result is chopped arbitrarily into sequences of length 1024 tokens,
  without regard to article boundaries.
  """

  @property
  def vocab_filename(self):
    return "vocab.wiki_noref.%d" % self.approx_vocab_size

  def filepath_to_unicode_text(self, filepath):
    """Overriddes the base class to clean up the xml dump before tokenizing."""
    dump = problem.to_unicode_ignore_erros(tf.gfile.Open(filepath).read())
    pages = _dump_to_pages(dump)
    ret = u""
    for p in pages:
      title = _page_to_title(p)
      text = _page_to_text(p)
      text = _remove_triple_quotes(
          _remove_double_brackets(_remove_references(text)))
      if u":" in title:
        # not a regular article
        continue
      if len(text) <= 140:
        # Probably a redirect or something like that.  Skip it.
        continue
      ret += u"title: \"%s\" length: %d\n%s\n" % (title, len(text), text)
    return ret

  @property
  def max_chars_for_vocab(self):
    """Number of characters of training data to use for generating vocab."""
    # magic number for backwards compatibility
    return 21240483


def _dump_to_pages(dump):
  """Extract pages from an xml dump.

  Args:
    dump: a unicode string
  Returns:
    a list of unicode strings
  """
  pos = 0
  ret = []
  start_tag = u"<page>\n"
  end_tag = u"</page>\n"
  while True:
    start_pos = dump.find(start_tag, pos)
    if start_pos == -1:
      break
    start_pos += len(start_tag)
    end_pos = dump.find(end_tag, start_pos)
    if end_pos == -1:
      break
    ret.append(dump[start_pos:end_pos])
    pos = end_pos + len(end_tag)
  return ret


def _page_to_title(page):
  """Extract the title from a page.

  Args:
    page: a unicode string
  Returns:
    a unicode string
  """
  # print("page=%s" % page)
  start_tag = u"<title>"
  end_tag = u"</title>"
  start_pos = page.find(start_tag)
  end_pos = page.find(end_tag)
  assert start_pos != -1
  assert end_pos != -1
  start_pos += len(start_tag)
  return page[start_pos:end_pos]


def _page_to_text(page):
  """Extract the text from a page.

  Args:
    page: a unicode string
  Returns:
    a unicode string
  """
  # text start tag looks like "<text ..otherstuff>"
  start_pos = page.find(u"<text")
  assert start_pos != -1
  end_tag_pos = page.find(u">", start_pos)
  assert end_tag_pos != -1
  end_tag_pos += len(u">")
  end_pos = page.find(u"</text>")
  if end_pos == -1:
    return u""
  return page[end_tag_pos:end_pos]


def _find_and_replace(text, start_string, end_string, replace_fn):
  """Remove everything found between instances of start_string and end_string.

  Replace each such instance with replace_fn(removed_text)

  e.g. _find_and_replace(u"the [[fat]] cat [[sat]]", u"[[", u"]]", lambda x: x)
    = u"the fat cat sat"

  Args:
    text: a unicode string
    start_string: a unicode string
    end_string: a unicode string
    replace_fn: a unary function from unicode string to unicode string

  Returns:
    a string
  """
  ret = u""
  current_pos = 0
  while True:
    start_pos = text.find(start_string, current_pos)
    if start_pos == -1:
      ret += text[current_pos:]
      break
    ret += text[current_pos:start_pos]
    end_pos = text.find(end_string, start_pos + len(start_string))
    if end_pos == -1:
      break
    ret += replace_fn(text[start_pos + len(start_string):end_pos])
    current_pos = end_pos + len(end_string)
  return ret


def _remove_references(text):
  """Strip out references from wikipedia xml."""
  return _find_and_replace(text, u"&lt;ref", u"&lt;/ref&gt;", lambda s: "")


def _remove_triple_quotes(text):
  """Strip out triple quotes from wikipedia xml."""
  return _find_and_replace(text, u"'''", u"'''", lambda s: s)


def _remove_double_brackets(text):
  """Remove double brackets (internal links) but leave the viewable text.

  Args:
    text: a unicode string
  Returns:
    a unicode string
  """
  def replacement_fn(s):
    if u":" in s:
      # this is probably a category or something like that.
      return ""
    # keep the part after the bar.
    bar_pos = s.find(u"|")
    if bar_pos == -1:
      return s
    return s[bar_pos + 1:]
  return _find_and_replace(text, u"[[", u"]]", replacement_fn)


@registry.register_problem
class LanguagemodelWikiNorefV8kL16k(LanguagemodelWikiNorefV8kL1k):
  """A language model on English Wikipedia.

  References removed.  Chopped into segments of 16k tokens.
  """

  @property
  def sequence_length(self):
    """Length of each example (in tokens)."""
    return 2**14
