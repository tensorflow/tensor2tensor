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


from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry


EOS = text_encoder.EOS_ID

_URL = "https://github.com/LauraMartinus/ukuxhumana/blob/master/data/en_tn"

_ENTN_TRAIN_DATASETS = [[
    _URL + "/eng_tswane.train.tar.gz?raw=true",
    ("entn_parallel.train.en", "entn_parallel.train.tn")
]]

_ENTN_TEST_DATASETS = [[
    _URL + "/eng_tswane.dev.tar.gz?raw=true",
    ("entn_parallel.dev.en", "entn_parallel.dev.tn")
]]


@registry.register_problem
class TranslateEntnRma(translate.TranslateProblem):
  """Problem spec for English-Setswana translation.

  Uses the RMA Autshumato dataset.
  """

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  @property
  def vocab_filename(self):
    return "vocab.entn.%d" % self.approx_vocab_size

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENTN_TRAIN_DATASETS if train else _ENTN_TEST_DATASETS
