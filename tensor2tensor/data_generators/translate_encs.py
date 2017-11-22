# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
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
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_name(self):
    return "vocab.encs"

  def generator(self, data_dir, tmp_dir, train):
    datasets = _ENCS_TRAIN_DATASETS if train else _ENCS_TEST_DATASETS
    tag = "train" if train else "dev"
    vocab_datasets = []
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "wmt_encs_tok_%s" % tag)
    # CzEng contains 100 gz files with tab-separated columns, so let's expect
    # it is the first dataset in datasets and use the newly created *.lang{1,2}
    # files for vocab construction.
    if datasets[0][0].endswith("data-plaintext-format.tar"):
      vocab_datasets.append([
          datasets[0][0],
          ["wmt_encs_tok_%s.lang1" % tag,
           "wmt_encs_tok_%s.lang2" % tag]
      ])
      datasets = datasets[1:]
    vocab_datasets += [[item[0], [item[1][0], item[1][1]]] for item in datasets]
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size,
        vocab_datasets)
    return translate.token_generator(data_path + ".lang1", data_path + ".lang2",
                                     symbolizer_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.CS_TOK


@registry.register_problem
class TranslateEncsWmtCharacters(translate.TranslateProblem):
  """Problem spec for WMT En-Cs character-based translation."""

  @property
  def is_character_level(self):
    return True

  def generator(self, data_dir, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _ENCS_TRAIN_DATASETS if train else _ENCS_TEST_DATASETS
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets,
                                       "wmt_encs_chr_%s" % tag)
    return translate.character_generator(
        data_path + ".lang1", data_path + ".lang2", character_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.CS_CHR
