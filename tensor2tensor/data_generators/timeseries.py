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

import numpy as np
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf


class TimeseriesProblem(problem.Problem):
  """Base Problem for multi timeseries datasets."""

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.RealEncoder(),
        "targets": text_encoder.RealEncoder()
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
        "split": problem.DatasetSplit.TRAIN,
        "shards": self.num_train_shards,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": self.num_eval_shards,
    }]

  @property
  def num_train_shards(self):
    """Number of training shards."""
    return 9

  @property
  def num_eval_shards(self):
    """Number of eval shards."""
    return 1

  @property
  def num_series(self):
    """Number of timeseries."""
    raise NotImplementedError()

  @property
  def num_input_timestamps(self):
    """Number of timestamps to include in the input."""
    raise NotImplementedError()

  @property
  def num_target_timestamps(self):
    """Number of timestamps to include in the target."""
    raise NotImplementedError()

  def timeseries_dataset(self):
    """Multi-timeseries data [ timestamps , self.num_series ] ."""
    raise NotImplementedError()

  def eval_metrics(self):
    eval_metrics = [metrics.Metrics.RMSE]
    return eval_metrics

  @property
  def normalizing_constant(self):
    """Constant by which all data will be multiplied to be more normalized."""
    return 1.0  # Adjust so that your loss is around 1 or 10 or 100, not 1e+9.

  def preprocess_example(self, example, unused_mode, unused_hparams):
    # Time series are flat on disk, we un-flatten them back here.
    flat_inputs = example["inputs"]
    flat_targets = example["targets"]
    c = self.normalizing_constant
    # Tensor2Tensor models expect [height, width, depth] examples, here we
    # use height for time and set width to 1 and num_series is our depth.
    example["inputs"] = tf.reshape(
        flat_inputs, [self.num_input_timestamps, 1, self.num_series]) * c
    example["targets"] = tf.reshape(
        flat_targets, [self.num_target_timestamps, 1, self.num_series]) * c
    return example

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split

    series = self.timeseries_dataset()
    num_timestamps = len(series)

    # Generate samples with num_input_timestamps for "inputs" and
    # num_target_timestamps in the "targets".
    for split_index in range(self.num_input_timestamps,
                             num_timestamps - self.num_target_timestamps + 1):
      inputs = series[split_index -
                      self.num_input_timestamps:split_index, :].tolist()
      targets = series[split_index:split_index +
                       self.num_target_timestamps, :].tolist()
      # We need to flatten the lists on disk for tf,Example to work.
      flat_inputs = [item for sublist in inputs for item in sublist]
      flat_targets = [item for sublist in targets for item in sublist]
      example_keys = ["inputs", "targets"]
      ex_dict = dict(zip(example_keys, [flat_inputs, flat_targets]))
      yield ex_dict

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": (registry.Modalities.REAL, self.num_series)}
    p.target_modality = (registry.Modalities.REAL, self.num_series)
    p.input_space_id = problem.SpaceID.REAL
    p.target_space_id = problem.SpaceID.REAL

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    filepath_fns = {
        problem.DatasetSplit.TRAIN: self.training_filepaths,
        problem.DatasetSplit.EVAL: self.dev_filepaths,
        problem.DatasetSplit.TEST: self.test_filepaths,
    }

    split_paths = [(split["split"], filepath_fns[split["split"]](
        data_dir, split["shards"], shuffled=False))
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
        "inputs": tf.VarLenFeature(tf.float32),
        "targets": tf.VarLenFeature(tf.float32),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


@registry.register_problem
class TimeseriesToyProblem(TimeseriesProblem):
  """Timeseries problem with a toy dataset."""

  @property
  def num_train_shards(self):
    """Number of training shards."""
    return 1

  @property
  def num_eval_shards(self):
    """Number of eval shards."""
    return 1

  @property
  def num_series(self):
    """Number of timeseries."""
    return 2

  @property
  def num_input_timestamps(self):
    """Number of timestamps to include in the input."""
    return 2

  @property
  def num_target_timestamps(self):
    """Number of timestamps to include in the target."""
    return 2

  def timeseries_dataset(self):
    series = [[float(i + n) for n in range(self.num_series)] for i in range(10)]

    return np.array(series)
