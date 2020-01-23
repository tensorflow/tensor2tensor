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

"""Data generators for StanfordNLI."""

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
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

EOS = text_encoder.EOS


@registry.register_problem
class StanfordNLI(text_problems.TextConcat2ClassProblem):
  """StanfordNLI classification problems."""

  # Link to data from GLUE: https://gluebenchmark.com/tasks
  _SNLI_URL = ("https://nlp.stanford.edu/projects/snli/snli_1.0.zip")

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
    # Note this binary classification is different from usual SNLI.
    return ["contradiction", "entailment", "neutral"]

  def _maybe_download_corpora(self, tmp_dir):
    snli_filename = "SNLI.zip"
    snli_finalpath = os.path.join(tmp_dir, "snli_1.0")
    if not tf.gfile.Exists(snli_finalpath):
      zip_filepath = generator_utils.maybe_download(
          tmp_dir, snli_filename, self._SNLI_URL)
      zip_ref = zipfile.ZipFile(zip_filepath, "r")
      zip_ref.extractall(tmp_dir)
      zip_ref.close()

    return snli_finalpath

  def example_generator(self, filename):
    label_list = self.class_labels(data_dir=None)
    for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
      if idx == 0: continue  # skip header
      line = text_encoder.to_unicode_utf8(line.strip())
      split_line = line.split("\t")
      # Works for both splits even though dev has some extra human labels.
      s1, s2 = split_line[5:7]
      if split_line[0] == "-":
        continue
      l = label_list.index(split_line[0])
      inputs = [s1, s2]
      yield {
          "inputs": inputs,
          "label": l
      }

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    snli_dir = self._maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = "snli_1.0_train.txt"
    else:
      filesplit = "snli_1.0_dev.txt"

    filename = os.path.join(snli_dir, filesplit)
    for example in self.example_generator(filename):
      yield example


@registry.register_problem
class StanfordNLICharacters(StanfordNLI):
  """StanfordNLI classification problems, character level"""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.THREE_CL_NLI


@registry.register_problem
class StanfordNLISharedVocab(StanfordNLI):
  """StanfordNLI classification problems with the LM1b vocabulary"""

  @property
  def vocab_filename(self):
    return lm1b.LanguagemodelLm1b32k().vocab_filename


@registry.register_problem
class StanfordNLIWikiLMSharedVocab(StanfordNLI):
  """StanfordNLI classification problems with the Wiki vocabulary"""

  @property
  def vocab_filename(self):
    return wiki_lm.LanguagemodelEnWiki32k().vocab_filename


@registry.register_problem
class StanfordNLIWikiLMSharedVocab64k(StanfordNLIWikiLMSharedVocab):
  """StanfordNLI classification problems with the Wiki vocabulary"""

  @property
  def vocab_filename(self):
    return wiki_lm.LanguagemodelEnWiki64k().vocab_filename
