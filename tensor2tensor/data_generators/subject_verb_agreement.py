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

"""Data generators for subject-verb agreement dataset.

https://arxiv.org/pdf/1611.01368.pdf

Based on he main paper, predicting verb's number can be done in two setups:
- Language Modeling
- Binary Classification

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import gzip
import os
import random
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

_FILE_NAME = 'agr_50_mostcommon_10K'
_TAR = _FILE_NAME + '.tsv.gz'
_URL = 'http://tallinzen.net/media/rnn_agreement/' + _TAR
_LABEL_DICT = {'VBZ': 0, 'VBP': 1}


def _build_vocab(examples, example_field, vocab_dir, vocab_name):
  """Build a vocabulary from examples.

  Args:
    examples: a dict containing all the examples.
    example_field: field of example from which the vocabulary is built.
    vocab_dir: directory where to save the vocabulary.
    vocab_name: vocab file name.

  Returns:
    text encoder.
  """
  vocab_path = os.path.join(vocab_dir, vocab_name)
  if not tf.gfile.Exists(vocab_path):
    data = []
    for e in examples:
      data.extend(e[example_field].split())
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    encoder = text_encoder.TokenTextEncoder(None, vocab_list=words)
    encoder.store_to_file(vocab_path)
  else:
    encoder = text_encoder.TokenTextEncoder(vocab_path)
  return encoder


def load_examples(tmp_dir, prop_train=0.09, prop_val=0.01):
  """Loads exampls from the tsv file.

  Args:
    tmp_dir: temp directory.
    prop_train: proportion of the train data
    prop_val: proportion of the validation data

  Returns:
    All examples in the dataset pluse train, test, and development splits.

  """

  infile = generator_utils.maybe_download(tmp_dir, _TAR, _URL)
  tf.logging.info('Loading examples')

  all_examples = []
  for i, d in enumerate(csv.DictReader(gzip.open(infile), delimiter='\t')):
    if i % 100000 == 0:
      tf.logging.info('%d examples have been loaded....' % i)
    ex = {x: int(y) if y.isdigit() else y for x, y in d.items()}
    all_examples.append(ex)

  random.seed(1)
  random.shuffle(all_examples)
  n_train = int(len(all_examples) * prop_train)
  n_val = n_train + int(len(all_examples) * prop_val)
  train = all_examples[:n_train]
  val = all_examples[n_train:n_val]
  test = []
  for e in all_examples[n_val:]:
    if e['n_intervening'] == e['n_diff_intervening']:
      test.append(e)

  return all_examples, train, val, test


@registry.register_problem
class SvaNumberPrediction(text_problems.Text2ClassProblem):
  """Subject verb agreement as verb number predicion (binary classification)."""

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.

    This is the setup of the main paper. 10% train/ 90% eval

    Returns:
      A dict containing splits information.

    """
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.EVAL,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.TEST,
        'shards': 10,
    }]

  @property
  def train_proportion(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return 0.09

  @property
  def validation_proportion(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return 0.01

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    """Class labels."""
    del data_dir
    return ['VBZ', 'VBP']

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of text and label pairs.

    Each yielded dict will be a single example. The inputs should be raw text.
    The label should be an int in [0, self.num_classes).

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).

    Returns:
      sample generator.
    """
    example_filed = 'sentence'
    examples_for_vocab, train, val, test = load_examples(
        tmp_dir, self.train_proportion, self.validation_proportion)
    _build_vocab(
        examples_for_vocab, example_filed, data_dir, self.vocab_filename)
    if dataset_split == problem.DatasetSplit.TRAIN:
      examples = train

    elif dataset_split == problem.DatasetSplit.EVAL:
      examples = val

    elif dataset_split == problem.DatasetSplit.TEST:
      examples = test

    def _generate_samples():
      for example in examples:
        index = int(example['verb_index']) - 1
        inputs = example[example_filed].split()[:index]
        yield {
            'inputs': ' '.join(inputs),
            'label': _LABEL_DICT[example['verb_pos']]
        }

    return _generate_samples()

  def eval_metrics(self):
    """Specify the set of evaluation metrics for this problem.

    Returns:
      List of evaluation metrics of interest.
    """
    # TODO(dehghani): Implement accuracy of the target word as a t2t metric.
    return [metrics.Metrics.ACC]


@registry.register_problem
class SvaLanguageModeling(text_problems.Text2SelfProblem):
  """Subject verb agreement as language modeling task."""

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.

    This is the setup of the main paper. 10% train/ 90% eval

    Returns:
      A dict containing splits information.

    """
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.EVAL,
        'shards': 1,
    }, {
        'split': problem.DatasetSplit.TEST,
        'shards': 10,
    }]

  @property
  def train_proportion(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return 0.09

  @property
  def validation_proportion(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return 0.01

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generates samples.

    Args:
      data_dir: data directory
      tmp_dir: temp directory
      dataset_split: dataset split

    Returns:
      sample generator.

    """

    example_filed = 'sentence'
    examples_for_vocab, train, val, test = load_examples(
        tmp_dir, self.train_proportion, self.validation_proportion)
    _build_vocab(
        examples_for_vocab, example_filed, data_dir, self.vocab_filename)
    if dataset_split == problem.DatasetSplit.TRAIN:
      examples = train

    elif dataset_split == problem.DatasetSplit.EVAL:
      examples = val

    elif dataset_split == problem.DatasetSplit.TEST:
      examples = test

    def _generate_samples():
      for example in examples:
        index = int(example['verb_index']) - 1
        targets = example[example_filed].split()[:index + 1]
        yield {'targets': ' '.join(targets)}

    return _generate_samples()
