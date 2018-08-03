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
"""Data generators for the Winograd NLI dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import six
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow as tf

EOS = text_encoder.EOS


@registry.register_problem
class WinogradNLI(text_problems.TextConcat2ClassProblem):
  """Winograd NLI classification problems."""

  # Link to data from GLUE: https://gluebenchmark.com/tasks
  _WNLI_URL = ("https://firebasestorage.googleapis.com/v0/b/"
               "mtl-sentence-representations.appspot.com/o/"
               "data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-"
               "4bd7-99a5-5e00222e0faf")

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**13  # 8k vocab suffices for this small dataset.

  @property
  def vocab_filename(self):
    return "vocab.wnli.%d" % self.approx_vocab_size

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    # Note this binary classification is different from usual MNLI.
    return ["not_entailment", "entailment"]

  def _maybe_download_corpora(self, tmp_dir):
    wnli_filename = "WNLI.zip"
    wnli_finalpath = os.path.join(tmp_dir, "WNLI")
    if not tf.gfile.Exists(wnli_finalpath):
      zip_filepath = generator_utils.maybe_download(
          tmp_dir, wnli_filename, self._WNLI_URL)
      zip_ref = zipfile.ZipFile(zip_filepath, "r")
      zip_ref.extractall(tmp_dir)
      zip_ref.close()

    return wnli_finalpath

  def example_generator(self, filename):
    for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
      if idx == 0: continue  # skip header
      if six.PY2:
        line = unicode(line.strip(), "utf-8")
      else:
        line = line.strip().decode("utf-8")
      _, s1, s2, l = line.split("\t")
      inputs = [s1, s2]
      yield {
          "inputs": inputs,
          "label": int(l)
      }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    wnli_dir = self._maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = "train.tsv"
    else:
      filesplit = "dev.tsv"

    filename = os.path.join(wnli_dir, filesplit)
    for example in self.example_generator(filename):
      yield example


@registry.register_problem
class WinogradNLICharacters(WinogradNLI):
  """Winograd NLI classification problems, character level"""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_NLI
