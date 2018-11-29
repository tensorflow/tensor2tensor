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


_ENRO_TRAIN_DATASETS = [
    [
        "http://www.statmt.org/europarl/v7/ro-en.tgz",
        ("europarl-v7.ro-en.en", "europarl-v7.ro-en.ro")
    ],
]
_ENRO_TEST_DATASETS = [
    [
        ("http://data.statmt.org/wmt16/translation-task/"
         "dev-romanian-updated.tgz"),
        ("dev/newsdev2016-roen-ref.en.sgm", "dev/newsdev2016-roen-src.ro.sgm")
    ],
]


@registry.register_problem
class TranslateEnroWmt8k(translate.TranslateProblem):
  """Problem spec for WMT En-Ro translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENRO_TRAIN_DATASETS if train else _ENRO_TEST_DATASETS


@registry.register_problem
class TranslateEnroWmt32k(TranslateEnroWmt8k):

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768


@registry.register_problem
class TranslateEnroWmtCharacters(TranslateEnroWmt8k):
  """Problem spec for WMT En-Ro translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER


@registry.register_problem
class TranslateEnroWmtMulti64k(TranslateEnroWmt8k):
  """Translation with muli-lingual vocabulary."""

  @property
  def vocab_filename(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k().vocab_filename
