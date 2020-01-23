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

"""Data generators for MultiNLI."""

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

# Link to data from GLUE: https://gluebenchmark.com/tasks
_MNLI_URL = ("https://firebasestorage.googleapis.com/v0/b/"
             "mtl-sentence-representations.appspot.com/o/"
             "data%2FMNLI.zip?alt=media&token=50329ea1-e339-"
             "40e2-809c-10c40afff3ce")


def _maybe_download_corpora(tmp_dir):
  """Download corpora for multinli.

  Args:
    tmp_dir: a string
  Returns:
    a string
  """
  mnli_filename = "MNLI.zip"
  mnli_finalpath = os.path.join(tmp_dir, "MNLI")
  if not tf.gfile.Exists(mnli_finalpath):
    zip_filepath = generator_utils.maybe_download(
        tmp_dir, mnli_filename, _MNLI_URL)
    zip_ref = zipfile.ZipFile(zip_filepath, "r")
    zip_ref.extractall(tmp_dir)
    zip_ref.close()

  return mnli_finalpath


def _example_generator(filename):
  """Generate mnli examples.

  Args:
    filename: a string
  Yields:
    dictionaries containing "premise", "hypothesis" and "label" strings
  """
  for idx, line in enumerate(tf.gfile.Open(filename, "rb")):
    if idx == 0: continue  # skip header
    line = text_encoder.to_unicode_utf8(line.strip())
    split_line = line.split("\t")
    # Works for both splits even though dev has some extra human labels.
    yield {
        "premise": split_line[8],
        "hypothesis": split_line[9],
        "label": split_line[-1]
    }


@registry.register_problem
class MultiNLI(text_problems.TextConcat2ClassProblem):
  """MultiNLI classification problems."""

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

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    mnli_dir = _maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = ["train.tsv"]
    else:
      # Using dev matched as the default for eval. Can also switch this to
      # dev_mismatched.tsv
      filesplit = ["dev_matched.tsv"]
    label_list = self.class_labels(data_dir=None)
    for fs in filesplit:
      filename = os.path.join(mnli_dir, fs)
      for example in _example_generator(filename):
        yield {
            "inputs": [example["premise"], example["hypothesis"]],
            "label": label_list.index(example["label"])
        }


@registry.register_problem
class MultiNLIText2text(text_problems.Text2TextProblem):
  """MultiNLI classification problems."""

  @property
  def is_generate_per_split(self):
    return True

  @property
  def approx_vocab_size(self):
    return 2**15

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    mnli_dir = _maybe_download_corpora(tmp_dir)
    if dataset_split == problem.DatasetSplit.TRAIN:
      filesplit = ["train.tsv"]
    else:
      # Using dev matched as the default for eval. Can also switch this to
      # dev_mismatched.tsv
      filesplit = ["dev_matched.tsv"]
    for fs in filesplit:
      filename = os.path.join(mnli_dir, fs)
      for example in _example_generator(filename):
        yield {
            "inputs": "multinli premise: %s hypothesis: %s" % (
                example["premise"], example["hypothesis"]),
            "targets": example["label"]
        }


@registry.register_problem
class MultiNLIText2textMulti64kPacked1k(MultiNLIText2text):
  """MultiNLI classification problems with the multi-lingual vocabulary."""

  @property
  def packed_length(self):
    return 1024

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()

  @property
  def num_training_examples(self):
    return 18300


@registry.register_problem
class MultiNLICharacters(MultiNLI):
  """MultiNLI classification problems, character level."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.THREE_CL_NLI


@registry.register_problem
class MultiNLISharedVocab(MultiNLI):
  """MultiNLI classification problems with the LM1b vocabulary."""

  @property
  def use_vocab_from_other_problem(self):
    return lm1b.LanguagemodelLm1b32k()


@registry.register_problem
class MultiNLIWikiLMSharedVocab(MultiNLI):
  """MultiNLI classification problems with the Wiki vocabulary."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelEnWiki32k()


@registry.register_problem
class MultiNLIWikiLMSharedVocab64k(MultiNLIWikiLMSharedVocab):
  """MultiNLI classification problems with the Wiki vocabulary."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelEnWiki64k()


@registry.register_problem
class MultiNLIWikiLMMultiVocab64k(MultiNLIWikiLMSharedVocab):
  """MultiNLI classification problems with the multi-lingual vocabulary."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()
