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
"""
Quora Question Pairs paraphrase problem publicly released on Kaggle.
Challenge topic : "Can you identify question pairs that have the same intent?"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry


_COLUMN_SEPARATOR = '<SEN_SEP>'

_SENT1_INDEX = 3
_SENT2_INDEX = 4
_QQP_LABEL_INDEX = 5
_QQP_TEXT_INDICES = [_SENT1_INDEX, _SENT2_INDEX]

_TRAIN_SHARDS = 10
_DEV_SHARDS = 1
_SUBWORD_VOCAB_SIZE = 8000
_QQP_NUM_CLASSES = 2

_QQP_TRAIN_DATASET = {
  "url": "https://drive.google.com/uc?export=download&id=1dnck-CCIyx8y2xg1vwFzcwXieZJB7ERC",
  "compressed_file_name": "qqp_train.tgz",
  "file_name": "train.csv"
}

_QQP_QUOTE_CHAR = '"'


@registry.register_problem
class ParaphraseQuoraQuestionPairs(text_problems.Text2ClassProblem):

  @property
  def approx_vocab_size(self):
    return _SUBWORD_VOCAB_SIZE

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    url = _QQP_TRAIN_DATASET["url"]
    compressed_filename = _QQP_TRAIN_DATASET["compressed_file_name"]
    decompressed_filename = _QQP_TRAIN_DATASET["file_name"]

    path_to_file = os.path.join(tmp_dir, decompressed_filename)
    compressed_filepath = generator_utils.maybe_download_from_drive(tmp_dir,
                                                                    compressed_filename, url)

    with tarfile.open(compressed_filepath, 'r:gz') as corpus_tar:
      corpus_tar.extractall(tmp_dir)

    if self.vocab_type == text_problems.VocabType.SUBWORD:
      generator_utils.get_or_generate_vocab_from_csv(data_dir,
                                                     self.vocab_filename,
                                                     self.approx_vocab_size,
                                                     path_to_file,
                                                     column_indices=_QQP_TEXT_INDICES,
                                                     skip_first_line=True,
                                                     column_separator=_COLUMN_SEPARATOR,
                                                     quotechar=_QQP_QUOTE_CHAR)

    return text_problems.text2class_csv_iterator(path_to_file,
                                                 column_indices=_QQP_TEXT_INDICES,
                                                 label_index=_QQP_LABEL_INDEX,
                                                 skip_first_line=True,
                                                 column_separator=_COLUMN_SEPARATOR,
                                                 quotechar=_QQP_QUOTE_CHAR)

  @property
  def num_classes(self):
    return _QQP_NUM_CLASSES

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": _TRAIN_SHARDS,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": _DEV_SHARDS,
    }]

  @property
  def is_generate_per_split(self):
    return False

