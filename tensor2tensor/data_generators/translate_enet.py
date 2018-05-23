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
"""Data generators for En-Et translation."""

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

# For English-Estonian the WMT18 data is used
# The complete corpus has ~ 2,18M sentences
_ENET_TRAIN_DATASETS = [
    [
        "http://data.statmt.org/wmt18/translation-task/training-parallel-ep-v8.tgz",  # pylint: disable=line-too-long
        ("training/europarl-v8.et-en.en", "training/europarl-v8.et-en.et")
    ],
    [
        "https://s3.amazonaws.com/web-language-models/paracrawl/release1/paracrawl-release1.en-et.zipporah0-dedup-clean.tgz",  # pylint: disable=line-too-long
        ("paracrawl-release1.en-et.zipporah0-dedup-clean.en",
         "paracrawl-release1.en-et.zipporah0-dedup-clean.et")
    ],
    [
        "http://data.statmt.org/wmt18/translation-task/rapid2016.tgz",
        ("rapid2016.en-et.en", "rapid2016.en-et.et")
    ],
]

# For development 2,000 parallel sentences are used
_ENET_TEST_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-et/raw/master/data/newsdev2018-enet.tar.gz",  # pylint: disable=line-too-long
    ("newsdev2018-enet-src.en", "newsdev2018-enet-ref.et")
]]


@registry.register_problem
class TranslateEnetWmt32k(translate.TranslateProblem):
  """Problem spec for WMT18 En-Et translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_filename(self):
    return "vocab.enet.%d" % self.approx_vocab_size

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENET_TRAIN_DATASETS if train else _ENET_TEST_DATASETS


@registry.register_problem
class TranslateEnetWmtCharacters(translate.TranslateProblem):
  """Problem spec for WMT18 En-Et translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENET_TRAIN_DATASETS if train else _ENET_TEST_DATASETS
