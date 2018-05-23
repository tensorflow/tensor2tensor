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

_ENCS_TRAIN_DATASETS = [
    [("https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/"
      "11234/1-1458/data-plaintext-format.tar"),
     ("tsv", 3, 2, "data.plaintext-format/*train.gz")],
    [
        "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",  # pylint: disable=line-too-long
        ("training/news-commentary-v12.cs-en.en",
         "training/news-commentary-v12.cs-en.cs")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.cs-en.en", "commoncrawl.cs-en.cs")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.cs-en.en", "training/europarl-v7.cs-en.cs")
    ],
]
_ENCS_TEST_DATASETS = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.cs")
    ],
]


@registry.register_problem
class TranslateEncsWmt32k(translate.TranslateProblem):
  """Problem spec for WMT English-Czech translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_filename(self):
    return "vocab.encs.%d" % self.approx_vocab_size

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENCS_TRAIN_DATASETS if train else _ENCS_TEST_DATASETS

  def vocab_data_files(self):
    datasets = self.source_data_files(problem.DatasetSplit.TRAIN)
    vocab_datasets = []
    if datasets[0][0].endswith("data-plaintext-format.tar"):
      vocab_datasets.append([
          datasets[0][0], [
              "%s-compiled-train.lang1" % self.name,
              "%s-compiled-train.lang2" % self.name
          ]
      ])
      datasets = datasets[1:]
    vocab_datasets += [[item[0], [item[1][0], item[1][1]]] for item in datasets]
    return vocab_datasets


@registry.register_problem
class TranslateEncsWmtCharacters(translate.TranslateProblem):
  """Problem spec for WMT En-Cs character-based translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    datasets = _ENCS_TRAIN_DATASETS if train else _ENCS_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "wmt_encs_chr_%s" % tag)
    return text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                data_path + ".lang2")
