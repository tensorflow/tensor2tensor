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

"""Data generators for CoNLL dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf


@registry.register_problem
class Conll2002Ner(text_problems.Text2textTmpdir):
  """Base class for CoNLL2002 problems."""

  def source_data_files(self, dataset_split):
    """Files to be passed to generate_samples."""
    raise NotImplementedError()

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir

    url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/conll2002.zip"  # pylint: disable=line-too-long
    compressed_filename = os.path.basename(url)
    compressed_filepath = os.path.join(tmp_dir, compressed_filename)
    generator_utils.maybe_download(tmp_dir, compressed_filename, url)

    compressed_dir = compressed_filepath.strip(".zip")

    filenames = self.source_data_files(dataset_split)
    for filename in filenames:
      filepath = os.path.join(compressed_dir, filename)
      if not tf.gfile.Exists(filepath):
        with zipfile.ZipFile(compressed_filepath, "r") as corpus_zip:
          corpus_zip.extractall(tmp_dir)
      with tf.gfile.GFile(filepath, mode="r") as cur_file:
        words, tags = [], []
        for line in cur_file:
          line_split = line.strip().split()
          if not line_split:
            yield {
                "inputs": str.join(" ", words),
                "targets": str.join(" ", tags)
            }
            words, tags = [], []
            continue
          words.append(line_split[0])
          tags.append(line_split[2])
        if words:
          yield {"inputs": str.join(" ", words), "targets": str.join(" ", tags)}


@registry.register_problem
class Conll2002EsNer(Conll2002Ner):
  """Problem spec for CoNLL2002 Spanish named entity task."""
  TRAIN_FILES = ["esp.train"]
  EVAL_FILES = ["esp.testa", "esp.testb"]

  def source_data_files(self, dataset_split):
    is_training = dataset_split == problem.DatasetSplit.TRAIN
    return self.TRAIN_FILES if is_training else self.EVAL_FILES


@registry.register_problem
class Conll2002NlNer(Conll2002Ner):
  """Problem spec for CoNLL2002 Dutch named entity task."""
  TRAIN_FILES = ["ned.train"]
  EVAL_FILES = ["ned.testa", "ned.testb"]

  def source_data_files(self, dataset_split):
    is_training = dataset_split == problem.DatasetSplit.TRAIN
    return self.TRAIN_FILES if is_training else self.EVAL_FILES
