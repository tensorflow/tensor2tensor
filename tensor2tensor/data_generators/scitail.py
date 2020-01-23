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

"""Data generators for SciTail."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import lm1b
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

EOS = text_encoder.EOS


@registry.register_problem
class SciTail(text_problems.TextConcat2ClassProblem):
  """SciTail classification problems."""

  # Data from allen institute for AI.
  _SCITAIL_URL = ("http://data.allenai.org.s3.amazonaws.com/"
                  "downloads/SciTailV1.1.zip")

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
    return 2**13

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    # Note this binary classification is different from usual SNLI.
    return ["neutral", "entails"]

  def _maybe_download_corpora(self, tmp_dir):
    scitail_filename = "SciTailV1.1.zip"
    scitail_finalpath = os.path.join(tmp_dir, "SciTailV1.1")
    if not tf.gfile.Exists(scitail_finalpath):
      zip_filepath = generator_utils.maybe_download(
          tmp_dir, scitail_filename, self._SCITAIL_URL)
      zip_ref = zipfile.ZipFile(zip_filepath, "r")
      zip_ref.extractall(tmp_dir)
      zip_ref.close()

    return scitail_finalpath

  def example_generator(self, filename):
    label_list = self.class_labels(data_dir=None)
    for line in tf.gfile.Open(filename, "rb"):
      line = text_encoder.to_unicode_utf8(line.strip())
      split_line = line.split("\t")
      s1, s2 = split_line[:2]
      l = label_list.index(split_line[2])
      inputs = [s1, s2]
      yield {
          "inputs": inputs,
          "label": l
      }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    scitail_dir = self._maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = "tsv_format/scitail_1.0_train.tsv"
    else:
      filesplit = "tsv_format/scitail_1.0_dev.tsv"

    filename = os.path.join(scitail_dir, filesplit)
    for example in self.example_generator(filename):
      yield example


@registry.register_problem
class SciTailCharacters(SciTail):
  """SciTail classification problems, character level"""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_NLI


@registry.register_problem
class SciTailSharedVocab(SciTail):
  """SciTail classification problems with the LM1b vocabulary"""

  @property
  def vocab_filename(self):
    return lm1b.LanguagemodelLm1b32k().vocab_filename
