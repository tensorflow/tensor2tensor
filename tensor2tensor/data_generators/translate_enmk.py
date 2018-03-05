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

# Dependency imports

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# For English-Macedonian the SETimes corpus
# from http://nlp.ffzg.hr/resources/corpora/setimes/ is used.
# The original dataset has 207,777 parallel sentences.
# For training the first 205,777 sentences are used.
_ENMK_TRAIN_DATASETS = [[
    "https://github.com/stefan-it/nmt-mk-en/raw/master/data/setimes.mk-en.train.tgz",  # pylint: disable=line-too-long
    ("train.en", "train.mk")
]]

# For development 1000 parallel sentences are used.
_ENMK_TEST_DATASETS = [[
    "https://github.com/stefan-it/nmt-mk-en/raw/master/data/setimes.mk-en.dev.tgz",  # pylint: disable=line-too-long
    ("dev.en", "dev.mk")
]]


# See this PR on github for some results with Transformer on these Problems.
# https://github.com/tensorflow/tensor2tensor/pull/626


@registry.register_problem
class TranslateEnmkSetimes32k(translate.TranslateProblem):
  """Problem spec for SETimes En-Mk translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_filename(self):
    return "vocab.enmk.%d" % self.approx_vocab_size

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENMK_TRAIN_DATASETS if train else _ENMK_TEST_DATASETS


@registry.register_problem
class TranslateEnmkSetimesCharacters(translate.TranslateProblem):
  """Problem spec for SETimes En-Mk translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENMK_TRAIN_DATASETS if train else _ENMK_TEST_DATASETS
