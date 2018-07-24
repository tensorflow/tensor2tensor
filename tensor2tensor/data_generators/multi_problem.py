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
"""Base class for combining multiple problems for multitask learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import discretization
import tensorflow as tf


class MultiProblem(problem.Problem):
  """MultiProblem base class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(MultiProblem, self).__init__(was_reversed, was_copy)
    self.task_list = []

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    assert len(self.task_list) > 1

    for task in self.task_list:
      assert task.vocab_type == text_problems.VocabType.CHARACTER
      task.generate_data(data_dir, tmp_dir, task_id)

  def add_task_id(self, task, example):
    """Convert example to code switching mode by adding a task id."""
    if hasattr(task, "class_labels"):
      # TODO(urvashik): handle the case where num_labels > 9
      example["targets"] = tf.cast(discretization.int_to_bit(
          example["targets"], 1, base=10) + 50, tf.int64)
      example["targets"] = tf.squeeze(example["targets"], axis=[-1])

    if task.has_inputs:
      inputs = example.pop("inputs")
      concat_list = [inputs, [task.task_id], example["targets"]]
    else:
      concat_list = [[task.task_id], example["targets"]]

    example["targets"] = tf.concat(concat_list, 0)
    return example

  def filepattern(self, data_dir, mode, shard=None):
    print("Generating multi problem filepattern")
    return [task.filepattern(data_dir, mode, shard) for task in self.task_list]

  def get_hparams(self, model_hparams=None):
    if self._hparams is not None:
      return self._hparams

    self._hparams = self.task_list[0].get_hparams()

    return self._hparams

  def flatten_zip(self, *args):
    flattened = tf.data.Dataset.from_tensors(args[0])
    for ex in args[1:]:
      flattened.concatenate(tf.data.Dataset.from_tensors(ex))

    return flattened

  def dataset(self,
              mode,
              data_dir=None,
              num_threads=None,
              output_buffer_size=None,
              shuffle_files=None,
              hparams=None,
              preprocess=True,
              dataset_split=None,
              shard=None,
              partition_id=0,
              num_partitions=1,
              max_records=-1):

    datasets = []
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    for task in self.task_list:
      task_dataset = task.dataset(mode, data_dir, num_threads,
                                  output_buffer_size, shuffle_files,
                                  hparams, preprocess, dataset_split,
                                  shard, partition_id, num_partitions,
                                  max_records).repeat()
      # pylint: disable=cell-var-from-loop
      task_dataset = task_dataset.map(lambda x: self.add_task_id(task, x))
      datasets.append(task_dataset)

    self.get_hparams()

    if is_training:
      single_mtl_dataset = tf.data.Dataset.zip(tuple(datasets)).flat_map(
          self.flatten_zip)
    else:
      single_mtl_dataset = datasets[0]
      for data in datasets[1:]:
        single_mtl_dataset = single_mtl_dataset.concatenate(data)

    return single_mtl_dataset
