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
"""Data generators for SquaAD (https://rajpurkar.github.io/SQuAD-explorer/).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class Squad(text_problems.QuestionAndContext2TextProblem):
  """Base class for SquAD question answering problem."""

  _DEV_SET = 'dev-v1.1.json'
  _URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset'
  _TRAINING_SET = 'train-v1.1.json'

  @property
  def dataset_splits(self):
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': 10,
    }, {
        'split': problem.DatasetSplit.EVAL,
        'shards': 1,
    }]

  @property
  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    url = self._URL
    file_name = (self._TRAINING_SET if dataset_split ==
                 problem.DatasetSplit.TRAIN else self._DEV_SET)
    squad_file = generator_utils.maybe_download(tmp_dir,
                                                file_name,
                                                os.path.join(url, file_name))
    with tf.gfile.GFile(squad_file, mode='r') as fp:
      squad = json.load(fp)

    version = squad['version']
    for article in squad['data']:
      if 'title' in article:
        title = article['title'].strip()
      else:
        title = 'no title'
      for paragraph in article['paragraphs']:
        context = paragraph['context'].strip()
        for qa in paragraph['qas']:
          question = qa['question'].strip()
          id_ = qa['id']

          answer_starts = [answer['answer_start'] for answer in qa['answers']]
          answers = [answer['text'].strip() for answer in qa['answers']]

          # Features currently used are 'context', 'question', and 'answers'.
          # Others are extracted here for the ease of future expansions.
          example = {
              'version': version,
              'title': title,
              'context': context,
              'question': question,
              'id': id_,
              'answer_starts': answer_starts,
              'answers': answers,
              'num_answers': len(answers),
              'is_supervised': True,
          }
          yield {
              'inputs': example['question'],
              # TODO(ddohan, wgaj): Figure out a way of extracting all answers.
              'targets': example['answers'][0],
              'context': example['context']
          }


@registry.register_problem
class SquadConcat(Squad):
  """Squad with question and context concatenated together in inputs."""

  def dataset_filename(self):
    return 'squad'

  def preprocess_example(self, example, unused_mode, unused_model_hparams):
    sep = tf.convert_to_tensor([self.QUESTION_SEPARATOR_ID],
                               dtype=example['inputs'].dtype)
    example['inputs'] = tf.concat(
        [example['inputs'], sep, example['context']], 0)
    return example

  def hparams(self, defaults, unused_model_hparams):
    (super(SquadConcat, self)
     .hparams(defaults, unused_model_hparams))
    p = defaults
    del p.input_modality['context']


@registry.register_problem
class SquadConcatPositioned(SquadConcat):
  """SquadConcat with targets in format of answer position + answer length."""

  def generate_targets(self, targets, context):
    targets = targets[:-1]  # skip last terminal symbol.
    targets_new = []
    i = 0
    while i < len(context) - len(targets):
      if context[i: i + len(targets)] == targets:
        # emit answer's position and length.
        targets_new.append(i)
        targets_new.append(len(targets))
      i += 1
    return targets_new

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    samples = (super(SquadConcatPositioned, self)
               .generate_encoded_samples(data_dir, tmp_dir, dataset_split))
    for sample in samples:
      sample['targets'] = self.generate_targets(sample['targets'],
                                                sample['context'])
      if sample['targets']:
        yield sample
