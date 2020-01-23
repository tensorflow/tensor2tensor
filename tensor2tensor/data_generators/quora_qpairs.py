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

"""Data generators for the Quora Question Pairs dataset."""

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
class QuoraQuestionPairs(text_problems.TextConcat2ClassProblem):
  """Quora duplicate question pairs binary classification problems."""

  # Link to data from GLUE: https://gluebenchmark.com/tasks
  _QQP_URL = ("https://firebasestorage.googleapis.com/v0/b/"
              "mtl-sentence-representations.appspot.com/o/"
              "data%2FQQP.zip?alt=media&token=700c6acf-160d-"
              "4d89-81d1-de4191d02cb5")

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 100,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**15

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    return ["not_duplicate", "duplicate"]

  def _maybe_download_corpora(self, tmp_dir):
    qqp_filename = "QQP.zip"
    qqp_finalpath = os.path.join(tmp_dir, "QQP")
    if not tf.gfile.Exists(qqp_finalpath):
      zip_filepath = generator_utils.maybe_download(
          tmp_dir, qqp_filename, self._QQP_URL)
      zip_ref = zipfile.ZipFile(zip_filepath, "r")
      zip_ref.extractall(tmp_dir)
      zip_ref.close()

    return qqp_finalpath

  def example_generator(self, filename):
    skipped = 0
    for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
      if idx == 0: continue  # skip header
      line = text_encoder.to_unicode_utf8(line.strip())
      split_line = line.split("\t")
      if len(split_line) < 6:
        skipped += 1
        tf.logging.info("Skipping %d" % skipped)
        continue
      s1, s2, l = split_line[3:]
      # A neat data augmentation trick from Radford et al. (2018)
      # https://blog.openai.com/language-unsupervised/
      inputs = [[s1, s2], [s2, s1]]
      for inp in inputs:
        yield {
            "inputs": inp,
            "label": int(l)
        }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    qqp_dir = self._maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = "train.tsv"
    else:
      filesplit = "dev.tsv"

    filename = os.path.join(qqp_dir, filesplit)
    for example in self.example_generator(filename):
      yield example


@registry.register_problem
class QuoraQuestionPairsCharacters(QuoraQuestionPairs):
  """Quora duplicate question pairs classification problems, character level"""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_SIM
