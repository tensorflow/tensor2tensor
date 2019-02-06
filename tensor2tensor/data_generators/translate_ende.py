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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry


_ENDE_TRAIN_DATASETS = [
    [
        "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz",  # pylint: disable=line-too-long
        ("training-parallel-nc-v13/news-commentary-v13.de-en.en",
         "training-parallel-nc-v13/news-commentary-v13.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.de-en.en", "commoncrawl.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.de-en.en", "training/europarl-v7.de-en.de")
    ],
]
_ENDE_EVAL_DATASETS = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.de")
    ],
]


@registry.register_problem
class TranslateEndeWmt8k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  @property
  def additional_training_datasets(self):
    """Allow subclasses to add training datasets."""
    return []

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    train_datasets = _ENDE_TRAIN_DATASETS + self.additional_training_datasets
    return train_datasets if train else _ENDE_EVAL_DATASETS


@registry.register_problem
class TranslateEndeWmt32k(TranslateEndeWmt8k):

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768


@registry.register_problem
class TranslateEndeWmtParacrawlBicleaner32k(TranslateEndeWmt32k):
  """WMT en-de corpus with extra data from Paracrawl, cleaned with Bicleaner."""

  @property
  def use_vocab_from_other_problem(self):
    return TranslateEndeWmt32k()

  @property
  def additional_training_datasets(self):
    paracrawl = "https://s3.amazonaws.com/web-language-models/paracrawl/"
    return [(paracrawl + "release3/en-de.bicleaner07.tmx.gz",
             ("tmx", "en-de.bicleaner07.tmx.gz"))]


@registry.register_problem
class TranslateEndeWmt32kPacked(TranslateEndeWmt32k):

  @property
  def packed_length(self):
    return 256

  @property
  def use_vocab_from_other_problem(self):
    return TranslateEndeWmt32k()


@registry.register_problem
class TranslateEndeWmt8kPacked(TranslateEndeWmt8k):

  @property
  def packed_length(self):
    return 256

  @property
  def use_vocab_from_other_problem(self):
    return TranslateEndeWmt8k()


@registry.register_problem
class TranslateEndeWmtCharacters(TranslateEndeWmt8k):
  """Problem spec for WMT En-De translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER


@registry.register_problem
class TranslateEndeWmtMulti64k(TranslateEndeWmt8k):
  """Translation with muli-lingual vocabulary."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()


@registry.register_problem
class TranslateEndeWmtMulti64kPacked1k(TranslateEndeWmtMulti64k):
  """Translation with muli-lingual vocabulary."""

  @property
  def packed_length(self):
    return 1024
