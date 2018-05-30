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
"""Multi time series forecasting problem."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


@registry.register_problem
class TimeSeriesToyProblem(problem.Problem):
  """Base Problem for multi timeseries for datasets."""

  def __init__(self,
               was_reversed=False,
               was_copy=False,
               num_train_shards=9,
               num_eval_shards=1,
               num_samples=100):
    super(TimeSeriesToyProblem, self).__init__(was_reversed, was_copy)
    self._num_train_shards = num_train_shards
    self._num_eval_shards = num_eval_shards
    self._num_samples = num_samples

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        'inputs': text_encoder.RealEncoder(),
        'targets': text_encoder.RealEncoder()
    }

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        'split': problem.DatasetSplit.TRAIN,
        'shards': self._num_train_shards,
    }, {
        'split': problem.DatasetSplit.EVAL,
        'shards': self._num_eval_shards,
    }]

  def eval_metrics(self):
    eval_metrics = [metrics.Metrics.RMSE]

    return eval_metrics

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split

    series_1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # This generates _num_samples instances of each possible split of series_1;
    # inputs & targets are of variable size.
    for x in range(self._num_samples):
      split_index = random.randint(1, 9)
      inputs, targets = series_1[:split_index], series_1[split_index:]
      example_keys = ['inputs', 'targets']
      ex_dict = dict(zip(example_keys, [inputs, targets]))
      print('Inputs & Targets #', x, ':', ex_dict)
      yield ex_dict

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {'inputs': (registry.Modalities.REAL, 1)}
    p.target_modality = (registry.Modalities.REAL, 1)
    p.input_space_id = problem.SpaceID.REAL
    p.target_space_id = problem.SpaceID.REAL

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    split_paths = [(split['split'], filepath_fns[split['split']](
        data_dir, split['shards'], shuffled=False))
                   for split in self.dataset_splits]

    all_paths = []
    for _, paths in split_paths:
      all_paths.extend(paths)

    if self.is_generate_per_split:
      for split, paths in split_paths:
        generator_utils.generate_files(
            self.generate_samples(data_dir, tmp_dir, split), paths)
    else:
      generator_utils.generate_files(
          self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN),
          all_paths)

    generator_utils.shuffle_dataset(all_paths)

  def example_reading_spec(self):
    data_fields = {
        'inputs': tf.VarLenFeature(tf.float32),
        'targets': tf.VarLenFeature(tf.float32),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)
