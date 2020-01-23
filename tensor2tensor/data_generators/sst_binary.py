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

"""Stanford Sentiment Treebank Binary Classification Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

EOS = text_encoder.EOS


@registry.register_problem
class SentimentSSTBinary(text_problems.Text2ClassProblem):
  """Stanford Sentiment Treebank binary classification problems."""

  # Link to data from GLUE: https://gluebenchmark.com/tasks
  _SST2_URL = ("https://firebasestorage.googleapis.com/v0/b/"
               "mtl-sentence-representations.appspot.com/o/"
               "data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-"
               "44a2-b9b4-cf6337f84ac8")

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**14

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    # Note this binary classification is different from usual MNLI.
    return ["neg", "pos"]

  def _maybe_download_corpora(self, tmp_dir):
    sst_binary_filename = "SST-2.zip"
    sst_binary_finalpath = os.path.join(tmp_dir, "SST-2")
    if not tf.gfile.Exists(sst_binary_finalpath):
      zip_filepath = generator_utils.maybe_download(
          tmp_dir, sst_binary_filename, self._SST2_URL)
      zip_ref = zipfile.ZipFile(zip_filepath, "r")
      zip_ref.extractall(tmp_dir)
      zip_ref.close()

    return sst_binary_finalpath

  def example_generator(self, filename):
    for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
      if idx == 0: continue  # skip header
      line = text_encoder.to_unicode_utf8(line.strip())
      sent, label = line.split("\t")
      yield {
          "inputs": sent,
          "label": int(label)
      }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    sst_binary_dir = self._maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = "train.tsv"
    else:
      filesplit = "dev.tsv"

    filename = os.path.join(sst_binary_dir, filesplit)
    for example in self.example_generator(filename):
      yield example


@registry.register_problem
class SentimentSSTBinaryCharacters(SentimentSSTBinary):
  """Binary Stanford Sentiment Treebank problems, character level"""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_CHR_SENT
