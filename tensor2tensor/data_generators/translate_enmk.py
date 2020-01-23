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

# For English-Macedonian the SETimes corpus
# from http://nlp.ffzg.hr/resources/corpora/setimes/ is used.
_ENMK_TRAIN_DATASETS = [[
    "http://nlp.ffzg.hr/data/corpora/setimes/setimes.en-mk.txt.tgz",
    ("setimes.en-mk.en.txt", "setimes.en-mk.mk.txt")
]]

# For development the MULTEXT-East "1984" corpus from
# https://www.clarin.si/repository/xmlui/handle/11356/1043 is used.
# 4,986 parallel sentences are used for evaluation.
_ENMK_DEV_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-mk/raw/master/data/MTE-1984-dev.enmk.tgz",  # pylint: disable=line-too-long
    ("MTE1984-dev.en", "MTE1984-dev.mk")
]]


# See this PR on github for some results with Transformer on these Problems.
# https://github.com/tensorflow/tensor2tensor/pull/738


@registry.register_problem
class TranslateEnmkSetimes32k(translate.TranslateProblem):
  """Problem spec for SETimes En-Mk translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENMK_TRAIN_DATASETS if train else _ENMK_DEV_DATASETS


@registry.register_problem
class TranslateEnmkSetimesCharacters(translate.TranslateProblem):
  """Problem spec for SETimes En-Mk translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENMK_TRAIN_DATASETS if train else _ENMK_DEV_DATASETS
