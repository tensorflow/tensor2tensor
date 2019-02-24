# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Data generators for the MSR Paraphrase Corpus."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow as tf

EOS = text_encoder.EOS


@registry.register_problem
class MSRParaphraseCorpus(text_problems.TextConcat2ClassProblem):
  """MSR Paraphrase Identification problems."""

  # Link to data from GLUE: https://gluebenchmark.com/tasks
  DEV_IDS = ("https://firebasestorage.googleapis.com/v0/b/"
             "mtl-sentence-representations.appspot.com/o/"
             "data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-"
             "48f4-b431-7480817f1adc")
  MRPC_TRAIN = ("https://s3.amazonaws.com/senteval/senteval_data/"
                "msr_paraphrase_train.txt")
  MRPC_TEST = ("https://s3.amazonaws.com/senteval/senteval_data/"
               "msr_paraphrase_test.txt")
  DATA_DIR = "MRPC"

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
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**13  # 8k vocab suffices for this small dataset.

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    return ["not_paraphrase", "paraphrase"]

  def _maybe_download_corpora(self, tmp_dir):
    mrpc_dir = os.path.join(tmp_dir, self.DATA_DIR)
    tf.gfile.MakeDirs(mrpc_dir)
    mrpc_train_finalpath = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
    mrpc_test_finalpath = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
    mrpc_dev_ids_finalpath = os.path.join(mrpc_dir, "dev_ids.tsv")

    def download_file(tdir, filepath, url):
      if not tf.gfile.Exists(filepath):
        generator_utils.maybe_download(tdir, filepath, url)

    download_file(mrpc_dir, mrpc_train_finalpath, self.MRPC_TRAIN)
    download_file(mrpc_dir, mrpc_test_finalpath, self.MRPC_TEST)
    download_file(mrpc_dir, mrpc_dev_ids_finalpath, self.DEV_IDS)

    return mrpc_dir

  def example_generator(self, filename, dev_ids, dataset_split):
    for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
      if idx == 0: continue  # skip header
      line = text_encoder.to_unicode_utf8(line.strip())
      l, id1, id2, s1, s2 = line.split("\t")
      is_dev = [id1, id2] in dev_ids
      if dataset_split == problem.DatasetSplit.TRAIN and is_dev:
        continue
      if dataset_split == problem.DatasetSplit.EVAL and not is_dev:
        continue
      inputs = [[s1, s2], [s2, s1]]
      for inp in inputs:
        yield {
            "inputs": inp,
            "label": int(l)
        }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    mrpc_dir = self._maybe_download_corpora(tmp_dir)
    if dataset_split != problem.DatasetSplit.TEST:
      filesplit = "msr_paraphrase_train.txt"
    else:
      filesplit = "msr_paraphrase_test.txt"
    dev_ids = []
    if dataset_split != problem.DatasetSplit.TEST:
      for row in tf.gfile.Open(os.path.join(mrpc_dir, "dev_ids.tsv")):
        dev_ids.append(row.strip().split("\t"))

    filename = os.path.join(mrpc_dir, filesplit)
    for example in self.example_generator(filename, dev_ids, dataset_split):
      yield example


@registry.register_problem
class MSRParaphraseCorpusCharacters(MSRParaphraseCorpus):
  """MSR Paraphrase Identification problems, character level"""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_SIM
