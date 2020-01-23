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

"""Data generators for enwik8 data-set."""

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


def _maybe_download_corpus(tmp_dir):
  """Download and unpack the corpus.

  Args:
    tmp_dir: directory containing dataset.

  Returns:
    path to entire corpus as a text file.
  """
  corpus_url = "http://mattmahoney.net/dc/enwik8.zip"
  corpus_filename = os.path.basename(corpus_url)
  compressed_filepath = generator_utils.maybe_download(
      tmp_dir, corpus_filename, corpus_url)

  zip_ref = zipfile.ZipFile(compressed_filepath, "r")
  zip_ref.extractall(tmp_dir)
  zip_ref.close()

  return os.path.join(tmp_dir, "enwik8")


@registry.register_problem
class Enwik8L65k(text_problems.Text2SelfProblem):
  """Enwiki8, with examples up to 65,536 characters long."""

  DUPE_FACTOR = 4

  @property
  def is_generate_per_split(self):
    return True

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_CHR

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 16,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  def max_length(self, model_hparams):
    return self.sequence_length

  @property
  def sequence_length(self):
    """Length of each example (number of characters)."""
    return 65536

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    filepath = _maybe_download_corpus(tmp_dir)
    with tf.io.gfile.GFile(filepath) as f:
      data = f.read()

    tf.logging.info("Length of enwik8 = %d", len(data))

    num_test_chars = 5000000

    if dataset_split == problem.DatasetSplit.TRAIN:
      part = data[: -2 * num_test_chars]
    elif dataset_split == problem.DatasetSplit.EVAL:
      part = data[-2 * num_test_chars: -num_test_chars]
    elif dataset_split == problem.DatasetSplit.TEST:
      part = data[-num_test_chars:]
    else:
      raise ValueError("Undefined dataset_split")

    tf.logging.info("Length of split '%s' = %d", dataset_split, len(part))

    # TODO(kitaev): Better handling of evaluation data, to ensure that there is
    # always context available.
    if dataset_split == problem.DatasetSplit.TRAIN:
      offset = self.sequence_length // self.DUPE_FACTOR
      for start in range(0, len(part), offset):
        yield {"targets": part[start:start+self.sequence_length]}
    else:
      for start in range(0, len(part), self.sequence_length):
        yield {"targets": part[start:start+self.sequence_length]}

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    vocab = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:
      sample["targets"] = vocab.encode(sample["targets"])
      yield sample
