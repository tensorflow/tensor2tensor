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
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_ENFR_TRAIN_SMALL_DATA = [
    [
        "https://s3.amazonaws.com/opennmt-trainingdata/baseline-1M-enfr.tgz",
        ("baseline-1M-enfr/baseline-1M_train.en",
         "baseline-1M-enfr/baseline-1M_train.fr")
    ],
]
_ENFR_TEST_SMALL_DATA = [
    [
        "https://s3.amazonaws.com/opennmt-trainingdata/baseline-1M-enfr.tgz",
        ("baseline-1M-enfr/baseline-1M_valid.en",
         "baseline-1M-enfr/baseline-1M_valid.fr")
    ],
]
_ENFR_TRAIN_LARGE_DATA = [
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.fr-en.en", "commoncrawl.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.fr-en.en", "training/europarl-v7.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz",
        ("training/news-commentary-v9.fr-en.en",
         "training/news-commentary-v9.fr-en.fr")
    ],
    [
        "http://www.statmt.org/wmt10/training-giga-fren.tar",
        ("giga-fren.release2.fixed.en.gz",
         "giga-fren.release2.fixed.fr.gz")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-un.tgz",
        ("un/undoc.2000.fr-en.en", "un/undoc.2000.fr-en.fr")
    ],
]
_ENFR_TEST_LARGE_DATA = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.fr")
    ],
]


@registry.register_problem
class TranslateEnfrWmtSmall8k(translate.TranslateProblem):
  """Problem spec for WMT En-Fr translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  @property
  def vocab_filename(self):
    return "vocab.enfr.%d" % self.approx_vocab_size

  @property
  def use_small_dataset(self):
    return True

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    if self.use_small_dataset:
      datasets = _ENFR_TRAIN_SMALL_DATA if train else _ENFR_TEST_SMALL_DATA
    else:
      datasets = _ENFR_TRAIN_LARGE_DATA if train else _ENFR_TEST_LARGE_DATA
    return datasets

  def vocab_data_files(self):
    return _ENFR_TRAIN_SMALL_DATA


@registry.register_problem
class TranslateEnfrWmtSmall32k(TranslateEnfrWmtSmall8k):

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768


@registry.register_problem
class TranslateEnfrWmt8k(TranslateEnfrWmtSmall8k):

  @property
  def use_small_dataset(self):
    return False


@registry.register_problem
class TranslateEnfrWmt32k(TranslateEnfrWmtSmall32k):

  @property
  def use_small_dataset(self):
    return False


@registry.register_problem
class TranslateEnfrWmt32kPacked(TranslateEnfrWmt32k):

  @property
  def packed_length(self):
    return 256


@registry.register_problem
class TranslateEnfrWmtSmallCharacters(translate.TranslateProblem):
  """Problem spec for WMT En-Fr translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  @property
  def use_small_dataset(self):
    return True

  @property
  def vocab_filename(self):
    return "vocab.enfr.%d" % self.approx_vocab_size

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    if self.use_small_dataset:
      datasets = _ENFR_TRAIN_SMALL_DATA if train else _ENFR_TEST_SMALL_DATA
    else:
      datasets = _ENFR_TRAIN_LARGE_DATA if train else _ENFR_TEST_LARGE_DATA
    return datasets


@registry.register_problem
class TranslateEnfrWmtCharacters(TranslateEnfrWmtSmallCharacters):

  @property
  def use_small_dataset(self):
    return False
