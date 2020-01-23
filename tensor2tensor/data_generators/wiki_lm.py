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

"""Data generators for untokenized wikipedia LM dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


def concat_generator(filename, up_threshold, low_threshold=10):
  """Generate concatenated lines from file upto up_threshold characters."""
  txt = ""
  for line in tf.gfile.Open(filename):
    line = line.strip()
    if len(txt) + len(line) + 1 >= up_threshold:
      ret = txt
      txt = ""
      # We don't yield very short long parts to prevent noisy examples.
      if len(ret) > low_threshold and len(ret) < up_threshold:
        yield {"targets": ret}

    if not txt:
      txt = line
    else:
      txt = " ".join([txt, line])


def mix_generators(generator_list):
  """Given python generators, generate from one, then from another, etc."""
  i = 0
  l = len(generator_list)
  stopiters_seen = 0
  while stopiters_seen <= l:
    try:
      yield six.next(generator_list[i % l])
      i += 1
      stopiters_seen = 0
    except StopIteration:
      i += 1
      stopiters_seen += 1


# File names and Google drive ids for the training/eval/test Wikipedia data.
_EN_TRAIN_NAME_ID = ("enwiki_train.txt.gz", "1-l02fI15ieMIZk8EnXhzhsvuEYRoznZ8")
_EN_EVAL_NAME_ID = ("enwiki_eval.txt.gz", "1odhDxWKtAPKXwxRw1KCrmlrVewxdXYq7")
_EN_TEST_NAME_ID = ("enwiki_test.txt.gz", "1i1Bg6XqvdRl1LuOiIWbg7ww8Y02Ip5VK")

_DE_TRAIN_NAME_ID = ("dewiki_train.txt.gz", "1FzEwoPonw9xlwX34vLPFInUF8F4X5yJy")
_DE_EVAL_NAME_ID = ("dewiki_eval.txt.gz", "1EKwRRPHyWny0RJ-aqSGMcNfjAlzFl51B")
_DE_TEST_NAME_ID = ("dewiki_test.txt.gz", "1Kr13Y7y_OD3JtUM9riXpFQP9UiHDkcFY")

_FR_TRAIN_NAME_ID = ("frwiki_train.txt.gz", "1etUIEZxMQKORwLGkssE5wlfCxxkeo8WV")
_FR_EVAL_NAME_ID = ("frwiki_eval.txt.gz", "13qrR5ZnHRgIMdcURVpixKL9gTO23GcPc")
_FR_TEST_NAME_ID = ("frwiki_test.txt.gz", "1mQpHRkAV9KXt68de69RwR8dkDi8EEusV")

_RO_TRAIN_NAME_ID = ("rowiki_train.txt.gz", "1wUJTEAlQeDcAwFnBxa8PzE-DCiXSU_W7")
_RO_EVAL_NAME_ID = ("rowiki_eval.txt.gz", "1uIPy2ZgkyArPy_gnsILENjgv4QQmSKtx")
_RO_TEST_NAME_ID = ("rowiki_test.txt.gz", "1kphjN4jXTbw8HyRYKaRE2zY4D7Fr-p7-")


@registry.register_problem
class LanguagemodelEnWiki32k(text_problems.Text2SelfProblem):
  """A language model on the untokenized wikipedia corpus, English."""

  train_names_ids = [_EN_TRAIN_NAME_ID]
  eval_names_ids = [_EN_EVAL_NAME_ID]
  test_names_ids = [_EN_TEST_NAME_ID]

  @property
  def approx_vocab_size(self):
    return 32000

  @property
  def max_samples_for_vocab(self):
    return 128000

  @property
  def combine_characters_threshold(self):
    """Threshold for upto how many characters to combine in examples."""
    return 512*8  # So we should have 512 tokens on average, maybe more.

  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples."""
    if dataset_split == problem.DatasetSplit.TRAIN:
      file_names_ids = self.train_names_ids
    elif dataset_split == problem.DatasetSplit.TEST:
      file_names_ids = self.test_names_ids
    else:
      file_names_ids = self.eval_names_ids

    wiki_generators = []
    for (fname, fid) in file_names_ids:
      url = "https://drive.google.com/uc?export=download&id=" + fid
      download_path = generator_utils.maybe_download_from_drive(
          tmp_dir, fname, url)
      wiki_file = os.path.join(tmp_dir, fname[:-3])
      if not tf.gfile.Exists(wiki_file):
        generator_utils.gunzip_file(download_path, wiki_file)
      wiki_generators.append(
          concat_generator(wiki_file, self.combine_characters_threshold))

    for example in mix_generators(wiki_generators):
      yield example


@registry.register_problem
class LanguagemodelEnWiki64k(LanguagemodelEnWiki32k):
  """As above, with 64k vocabulary."""

  @property
  def approx_vocab_size(self):
    return 64000


@registry.register_problem
class LanguagemodelEnWiki64kShorter(LanguagemodelEnWiki64k):
  """With 64k vocabulary and shorter truncation lengths."""

  @property
  def combine_characters_threshold(self):
    """Threshold for upto how many characters to combine in examples."""
    return 384*8

  @property
  def use_vocab_from_other_problem(self):
    return LanguagemodelEnWiki64k()


@registry.register_problem
class LanguagemodelDeWiki32k(LanguagemodelEnWiki32k):
  """A language model on the untokenized wikipedia corpus, German."""

  train_names_ids = [_DE_TRAIN_NAME_ID]
  eval_names_ids = [_DE_EVAL_NAME_ID]
  test_names_ids = [_DE_TEST_NAME_ID]


@registry.register_problem
class LanguagemodelDeWiki64k(LanguagemodelDeWiki32k):
  """As above, with 64k vocabulary."""

  @property
  def approx_vocab_size(self):
    return 64000


@registry.register_problem
class LanguagemodelFrWiki32k(LanguagemodelEnWiki32k):
  """A language model on the untokenized wikipedia corpus, French."""

  train_names_ids = [_FR_TRAIN_NAME_ID]
  eval_names_ids = [_FR_EVAL_NAME_ID]
  test_names_ids = [_FR_TEST_NAME_ID]


@registry.register_problem
class LanguagemodelFrWiki64k(LanguagemodelFrWiki32k):
  """As above, with 64k vocabulary."""

  @property
  def approx_vocab_size(self):
    return 64000


@registry.register_problem
class LanguagemodelRoWiki32k(LanguagemodelEnWiki32k):
  """A language model on the untokenized wikipedia corpus, Romanian."""

  train_names_ids = [_RO_TRAIN_NAME_ID]
  eval_names_ids = [_RO_EVAL_NAME_ID]
  test_names_ids = [_RO_TEST_NAME_ID]


@registry.register_problem
class LanguagemodelRoWiki64k(LanguagemodelRoWiki32k):
  """As above, with 64k vocabulary."""

  @property
  def approx_vocab_size(self):
    return 64000


@registry.register_problem
class LanguagemodelDeEnFrRoWiki64k(LanguagemodelEnWiki32k):
  """A language model on untokenized Wikipedia, 4 languages together."""

  train_names_ids = [_DE_TRAIN_NAME_ID, _FR_TRAIN_NAME_ID,
                     _EN_TRAIN_NAME_ID, _RO_TRAIN_NAME_ID]
  eval_names_ids = [_DE_EVAL_NAME_ID, _FR_EVAL_NAME_ID,
                    _EN_EVAL_NAME_ID, _RO_EVAL_NAME_ID]
  test_names_ids = [_DE_TEST_NAME_ID, _FR_TEST_NAME_ID,
                    _EN_TEST_NAME_ID, _RO_TEST_NAME_ID]

  @property
  def approx_vocab_size(self):
    return 64000

  @property
  def max_samples_for_vocab(self):
    return 256000  # Samples are intertwined, take more to cover 4 languages.


@registry.register_problem
class LanguagemodelDeEnFrRoWiki64kFitbPacked1k(
    LanguagemodelDeEnFrRoWiki64k):
  """4 languages fill-in-the-blanks text-to-text problem."""

  @property
  def use_vocab_from_other_problem(self):
    return LanguagemodelDeEnFrRoWiki64k()

  @property
  def has_inputs(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    for example in super(
        LanguagemodelDeEnFrRoWiki64kFitbPacked1k, self).generate_samples(
            data_dir, tmp_dir, dataset_split):
      a, b = generator_utils.random_deinterleave(example["targets"])
      yield {"inputs": a, "targets": b}

  @property
  def num_training_examples(self):
    return 3597800

  @property
  def packed_length(self):
    return 1024

  @property
  def inputs_prefix(self):
    return "wiki fill "

  @property
  def targets_prefix(self):
    return "wiki fill "
