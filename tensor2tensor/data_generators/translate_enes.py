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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_ENES_TRAIN_DATASETS = [
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.es-en.en", "commoncrawl.es-en.es")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.es-en.en", "training/europarl-v7.es-en.es")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-un.tgz",
        ("un/undoc.2000.es-en.en", "un/undoc.2000.es-en.es")
    ],
    [
        "https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-es.zipporah0-dedup-clean.tgz",
        ("paracrawl-release1.en-es.zipporah0-dedup-clean.en",
         "paracrawl-release1.en-es.zipporah0-dedup-clean.es")
    ]
]
_ENES_TEST_DATASETS = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.es")
    ],
]


@registry.register_problem
class TranslateEnesWmt32k(translate.TranslateProblem):
  """En-es translation trained on WMT corpus."""

  @property
  def additional_training_datasets(self):
    """Allow subclasses to add training datasets."""
    return []

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    train_datasets = _ENES_TRAIN_DATASETS + self.additional_training_datasets
    return train_datasets if train else _ENES_TEST_DATASETS

  def vocab_data_files(self):
    return _ENES_TRAIN_DATASETS


@registry.register_problem
class TranslateEnesWmtClean32k(TranslateEnesWmt32k):
  """En-es translation trained on WMT with further cleaning."""

  @property
  def use_vocab_from_other_problem(self):
    return TranslateEnesWmt32k()

  @property
  def datatypes_to_clean(self):
    return ["txt"]


@registry.register_problem
class TranslateEnesWmt32kPacked(TranslateEnesWmt32k):

  @property
  def packed_length(self):
    return 256

  @property
  def use_vocab_from_other_problem(self):
    return TranslateEnesWmt32k()


@registry.register_problem
class TranslateEnesWmt8k(TranslateEnesWmt32k):
  """Problem spec for WMT En-Es translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192


@registry.register_problem
class TranslateEnesWmt8kPacked(TranslateEnesWmt8k):

  @property
  def packed_length(self):
    return 256

  @property
  def use_vocab_from_other_problem(self):
    return TranslateEnesWmt8k()


@registry.register_problem
class TranslateEnesWmtCharacters(TranslateEnesWmt8k):
  """Problem spec for WMT En-Es translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER
