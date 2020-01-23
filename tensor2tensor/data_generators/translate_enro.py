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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry


_ENRO_TRAIN_DATASETS = [
    [
        "http://www.statmt.org/europarl/v7/ro-en.tgz",
        ("europarl-v7.ro-en.en", "europarl-v7.ro-en.ro")
    ],
    [
        "http://opus.nlpl.eu/download.php?f=SETIMES/v2/moses/en-ro.txt.zip",
        ("SETIMES.en-ro.en", "SETIMES.en-ro.ro")
    ]
]
_ENRO_TEST_DATASETS = [
    [
        ("http://data.statmt.org/wmt16/translation-task/"
         "dev-romanian-updated.tgz"),
        ("dev/newsdev2016-roen-ref.en.sgm", "dev/newsdev2016-roen-src.ro.sgm")
    ],
]


@registry.register_problem
class TranslateEnroWmt8k(translate.TranslateProblem):
  """Problem spec for WMT En-Ro translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENRO_TRAIN_DATASETS if train else _ENRO_TEST_DATASETS


@registry.register_problem
class TranslateEnroWmt32k(TranslateEnroWmt8k):

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768


@registry.register_problem
class TranslateEnroWmtCharacters(TranslateEnroWmt8k):
  """Problem spec for WMT En-Ro translation."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER


@registry.register_problem
class TranslateEnroWmtMulti64k(TranslateEnroWmt8k):
  """Translation with muli-lingual vocabulary."""

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()


@registry.register_problem
class TranslateEnroWmtMultiSmall64k(TranslateEnroWmt8k):
  """Translation with muli-lingual vocabulary, small (6K) training data."""

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 16,  # It's a small dataset, TPUs like at least a few shards.
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def use_vocab_from_other_problem(self):
    return wiki_lm.LanguagemodelDeEnFrRoWiki64k()

  @property
  def how_many_examples_to_sample(self):
    return 6000

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate just the first 6k samples for training."""
    # If not training, do the same as before.
    if dataset_split != problem.DatasetSplit.TRAIN:
      for x in super(TranslateEnroWmtMultiSmall64k, self).generate_samples(
          data_dir, tmp_dir, dataset_split):
        yield x
      raise StopIteration
    # Now we assume we're training.
    counter = 0
    # The size of this data-set in total is around 614K, we want to sample so
    # that in expectation we take the requested number of samples in 1 go.
    sample_prob = self.how_many_examples_to_sample / float(614000)
    # Let's sample.
    for x in super(TranslateEnroWmtMultiSmall64k, self).generate_samples(
        data_dir, tmp_dir, dataset_split):
      if random.random() > sample_prob:
        continue
      counter += 1
      if counter > self.how_many_examples_to_sample:
        raise StopIteration
      yield x
    # We do it again if we don't have enough samples.
    if counter < self.how_many_examples_to_sample:
      for x in super(TranslateEnroWmtMultiSmall64k, self).generate_samples(
          data_dir, tmp_dir, dataset_split):
        if random.random() > sample_prob:
          continue
        counter += 1
        if counter > self.how_many_examples_to_sample:
          raise StopIteration
        yield x


@registry.register_problem
class TranslateEnroWmtMultiTiny64k(TranslateEnroWmtMultiSmall64k):
  """Translation with muli-lingual vocabulary, tiny (600) training data."""

  @property
  def how_many_examples_to_sample(self):
    return 600


@registry.register_problem
class TranslateEnroWmtMultiTiny64kPacked1k(TranslateEnroWmtMultiTiny64k):
  """Translation with muli-lingual vocabulary."""

  @property
  def packed_length(self):
    return 1024

  @property
  def num_training_examples(self):
    return 32

  @property
  def inputs_prefix(self):
    return "translate English Romanian "

  @property
  def targets_prefix(self):
    return "translate Romanian English "
