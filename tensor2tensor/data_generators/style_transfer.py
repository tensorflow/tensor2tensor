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
"""Base classes for text-based language style transfer problems.

* StyleTransferProblem: abstract class for style transfer problems.
* StyleTransferShakespeare: specific problem implementation that enriches
  language with Shakespeare-like style.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf

logger = tf.logging

"""
Modern-Shakespeare corpus is consisted of:
- 18,395 parallel sentences for training (train set),
- 1,218 parallel sentences for evaluation (dev set),
- 1,462 parallel sentence for testing (test set).
"""

_SHAKESPEARE_MODERN_TRAIN_DATASET = [[
    "https://github.com/tlatkowski/st/raw/master/shakespeare.train.tgz",
    ("train.original", "train.modern")
]]

_SHAKESPEARE_MODERN_DEV_DATASET = [[
    "https://github.com/tlatkowski/st/raw/master/shakespeare.dev.tgz",
    ("dev.original", "dev.modern")
]]

_TRAIN_SHARDS = 1
_DEV_SHARDS = 1
_SUBWORD_VOCAB_SIZE = 8000


class StyleTransferProblem(text_problems.Text2TextProblem):
  """Base class for transferring styles problems"""

  @property
  def target(self):
    raise NotImplementedError()

  @property
  def source(self):
    raise NotImplementedError()

  def dataset_url(self, dataset_split):
    raise NotImplementedError()

  def vocab_data_files(self):
    """Files to be passed to get_or_generate_vocab."""
    return self.dataset_url(problem.DatasetSplit.TRAIN)

  @property
  def approx_vocab_size(self):
    return _SUBWORD_VOCAB_SIZE

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": _TRAIN_SHARDS,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": _DEV_SHARDS,
    }]

  @property
  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    dataset = self.dataset_url(dataset_split)

    tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"

    url = dataset[0][0]
    compressed_filename = os.path.basename(url)
    compressed_filepath = os.path.join(tmp_dir, compressed_filename)
    generator_utils.maybe_download(tmp_dir, compressed_filename, url)

    mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
    with tarfile.open(compressed_filepath, mode) as corpus_tar:
      corpus_tar.extractall(tmp_dir)

    if self.vocab_type == text_problems.VocabType.SUBWORD:
      generator_utils.get_or_generate_vocab(
          data_dir, tmp_dir, self.vocab_filename, self.approx_vocab_size,
          self.vocab_data_files())

    source_file = os.path.join(tmp_dir, tag + ".modern")
    target_file = os.path.join(tmp_dir, tag + ".original")
    return text_problems.text2text_txt_iterator(source_file,
                                                target_file)


@registry.register_problem
class StyleTransferShakespeareToModern(StyleTransferProblem):
  """Transferring style from Shakespeare original English to modern one"""

  @property
  def target(self):
    return ".modern"

  @property
  def source(self):
    return ".original"

  def dataset_url(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    if train:
      return _SHAKESPEARE_MODERN_TRAIN_DATASET
    return _SHAKESPEARE_MODERN_DEV_DATASET


@registry.register_problem
class StyleTransferModernToShakespeare(StyleTransferProblem):
  """Transferring style from modern English to Shakespeare original English"""

  @property
  def target(self):
    return ".original"

  @property
  def source(self):
    return ".modern"

  def dataset_url(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    if train:
      return _SHAKESPEARE_MODERN_TRAIN_DATASET
    return _SHAKESPEARE_MODERN_DEV_DATASET
