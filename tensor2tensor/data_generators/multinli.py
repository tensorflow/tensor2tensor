# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Data generators for MultiNLI (https://www.nyu.edu/projects/bowman/multinli/).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import zipfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

EOS = text_encoder.EOS_ID


class MultinliProblem(problem.Problem):
  """Base class for MultiNLI classification problems."""

  _ZIP = 'multinli_1.0.zip'
  _URL = 'https://www.nyu.edu/projects/bowman/multinli/' + _ZIP
  _LABEL_DICT = {'contradiction': 0,
                 'entailment': 1,
                 'neutral': 2}
  _LABELS = {'contradiction', 'entailment', 'neutral'}

  @property
  def num_shards(self):
    return 10

  @property
  def vocab_file(self):
    if self._matched:
      return 'multinli_matched.vocab'
    else:
      return 'multinli_mismatched.vocab'

  @property
  def targeted_vocab_size(self):
    return 2**14

  @property
  def _matched(self):
    raise NotImplementedError()

  @property
  def _train_file(self):
    return 'multinli_1.0/multinli_1.0_train.jsonl'

  @property
  def _dev_file(self):
    if self._matched:
      return 'multinli_1.0/multinli_1.0_dev_matched.jsonl'
    else:
      return 'multinli_1.0/multinli_1.0_dev_mismatched.jsonl'

  def _examples(self, data_dir, tmp_dir, train):
    file_path = generator_utils.maybe_download(tmp_dir, self._ZIP, self._URL)
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(tmp_dir)
    zip_ref.close()

    data_file = self._train_file if train else self._dev_file
    examples = []
    with tf.gfile.GFile(os.path.join(tmp_dir, data_file), mode='r') as f:
      for line in f:
        record = json.loads(line)
        try:
          label_str = record['gold_label'].encode('ascii')
          if label_str != '-':
            label = self._LABEL_DICT[label_str]
            sentence1 = record['sentence1'].encode('ascii')
            sentence2 = record['sentence2'].encode('ascii')
            examples.append({'sentence1': sentence1,
                             'sentence2': sentence2,
                             'label': label})
        except UnicodeEncodeError:
          pass

    return examples

  def _inputs_and_targets(self, encoder, examples):
    for e in examples:
      enc_s1 = encoder.encode(e['sentence1'])
      enc_s2 = encoder.encode(e['sentence2'])

      yield {
          'inputs': enc_s1 + [EOS] + enc_s2 + [EOS],
          'targets': [e['label']]
      }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(data_dir, 1, shuffled=False)

    train_examples = self._examples(data_dir, tmp_dir, train=True)
    dev_examples = self._examples(data_dir, tmp_dir, train=False)

    encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        (e['sentence1'] + ' ' + e['sentence2']
         for e in train_examples + dev_examples)
        )

    generator_utils.generate_dataset_and_shuffle(
        self._inputs_and_targets(encoder, train_examples), train_paths,
        self._inputs_and_targets(encoder, dev_examples), dev_paths)

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    source_vocab_size = self._encoders['inputs'].vocab_size
    p.input_modality = {
        'inputs': (registry.Modalities.SYMBOL, source_vocab_size)
    }
    p.target_modality = (registry.Modalities.CLASS_LABEL, 3)
    p.input_space_id = problem.SpaceID.EN_TOK
    p.target_space_id = problem.SpaceID.GENERIC

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    return {
        'inputs': encoder,
        'targets': text_encoder.ClassLabelEncoder(self._LABELS),
    }

  def example_reading_spec(self):
    data_fields = {
        'inputs': tf.VarLenFeature(tf.int64),
        'targets': tf.FixedLenFeature([1], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def eval_metrics(self):
    return [metrics.Metrics.ACC]


@registry.register_problem
class MultinliMatched(MultinliProblem):
  """MultiNLI with matched dev set."""

  @property
  def _matched(self):
    return True


@registry.register_problem
class MultinliMismatched(MultinliProblem):
  """MultiNLI with mismatched dev set."""

  @property
  def _matched(self):
    return False
