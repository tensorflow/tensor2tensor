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
"""Data generators for MultiNLI."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import lm1b
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow as tf

EOS = text_encoder.EOS


@registry.register_problem
class MultiNLI(text_problems.TextConcat2ClassProblem):
  """MultiNLI classification problems."""

  # Link to data from GLUE: https://gluebenchmark.com/tasks
  _MNLI_URL = ("https://firebasestorage.googleapis.com/v0/b/"
               "mtl-sentence-representations.appspot.com/o/"
               "data%2FMNLI.zip?alt=media&token=50329ea1-e339-"
               "40e2-809c-10c40afff3ce")

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
    return 3

  def class_labels(self, data_dir):
    del data_dir
    # Note this binary classification is different from usual MNLI.
    return ["contradiction", "entailment", "neutral"]

  def _maybe_download_corpora(self, tmp_dir):
    mnli_filename = "MNLI.zip"
    mnli_finalpath = os.path.join(tmp_dir, "MNLI")
    if not tf.gfile.Exists(mnli_finalpath):
      zip_filepath = generator_utils.maybe_download(
          tmp_dir, mnli_filename, self._MNLI_URL)
      zip_ref = zipfile.ZipFile(zip_filepath, "r")
      zip_ref.extractall(tmp_dir)
      zip_ref.close()

    return mnli_finalpath

  def example_generator(self, filename):
    label_list = self.class_labels(data_dir=None)
    for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
      if idx == 0: continue  # skip header
      if six.PY2:
        line = unicode(line.strip(), "utf-8")
      else:
        line = line.strip().decode("utf-8")
      split_line = line.split("\t")
      # Works for both splits even though dev has some extra human labels.
      s1, s2 = split_line[8:10]
      l = label_list.index(split_line[-1])
      inputs = [s1, s2]
      yield {
          "inputs": inputs,
          "label": l
      }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    mnli_dir = self._maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = ["train.tsv"]
    else:
      filesplit = ["dev_matched.tsv", "dev_mismatched.tsv"]

    for fs in filesplit:
      filename = os.path.join(mnli_dir, fs)
      for example in self.example_generator(filename):
        yield example


@registry.register_problem
class MultiNLICharacters(MultiNLI):
  """MultiNLI classification problems, character level"""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.THREE_CL_NLI


@registry.register_problem
class MultiNLISharedVocab(MultiNLI):
  """MultiNLI classification problems with the LM1b vocabulary"""

  @property
  def vocab_filename(self):
    return lm1b.LanguagemodelLm1b32k().vocab_filename
