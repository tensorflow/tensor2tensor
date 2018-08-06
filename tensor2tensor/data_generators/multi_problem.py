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
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import discretization
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
import tensorflow as tf


class MultiProblem(problem.Problem):
  """MultiProblem base class."""

  def __init__(self, was_reversed=False, was_copy=False):
    super(MultiProblem, self).__init__(was_reversed, was_copy)
    self.task_list = []

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    assert len(self.task_list) > 1

    for task in self.task_list:
      task.generate_data(data_dir, tmp_dir, task_id)

  def add_task_id(self, task, example, encoder):
    """Convert example to code switching mode by adding a task id."""
    if hasattr(task, "class_labels"):
      if self.vocab_type == text_problems.VocabType.CHARACTER:
        # TODO(urvashik): handle the case where num_labels > 9
        example["targets"] = tf.cast(discretization.int_to_bit(
            example["targets"], 1, base=10) + 50, tf.int64)
        example["targets"] = tf.squeeze(example["targets"], axis=[-1])
      elif self.vocab_type == text_problems.VocabType.SUBWORD:
        offset = encoder.vocab_size + len(self.task_list)
        # An additional +1 because of 0-indexing
        example["targets"] = offset + example["targets"] + 1

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

    self._hparams = self.task_list[0].get_hparams(model_hparams)
    # increase the vocab size in order to account for task ids
    vocab_size_inc = len(self.task_list)
    vocab_size_inc += self.get_max_num_classes()
    vocab_size = self._hparams.vocabulary["targets"].vocab_size
    self._hparams.target_modality = (registry.Modalities.SYMBOL,
                                     vocab_size + vocab_size_inc)

    return self._hparams

  def flatten_zip(self, *args):
    """A list of examples to a dataset containing mixed examples.

    Given a list of `n` dataset examples, flatten them by converting
    each element into a dataset and concatenating them to convert into a
    single dataset.

    Args:
      *args: A list containing one example each from `n` different datasets.

    Returns:
      flattened: A new dataset containing the examples from the list as part
        of a single dataset.
    """

    flattened = tf.data.Dataset.from_tensors(args[0])
    for ex in args[1:]:
      flattened = flattened.concatenate(tf.data.Dataset.from_tensors(ex))

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

    # A list of datasets corresponding to the tasks in the task_list object
    # that need to be mixed.
    datasets = []
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    primary_task = self.task_list[0]
    if primary_task.has_inputs:
      raise ValueError("Only support language models as primary problem which "
                       "supplies the vocabulary and the hparams.")
    enc = primary_task.feature_encoders(data_dir=data_dir)["targets"]

    for idx, task in enumerate(self.task_list):
      task_dataset = task.dataset(mode, data_dir, num_threads,
                                  output_buffer_size, shuffle_files,
                                  hparams, preprocess, dataset_split,
                                  shard, partition_id, num_partitions,
                                  max_records)

      if idx == 0:
        self.update_task_ids(enc)

      if is_training:
        task_dataset = task_dataset.repeat()
      # pylint: disable=cell-var-from-loop
      task_dataset = task_dataset.map(lambda x: self.add_task_id(task, x, enc))
      datasets.append(task_dataset)

    # Setup the problem hparams by setting them to the LM task hparams.
    self.get_hparams()

    single_mtl_dataset = tf.data.Dataset.zip(tuple(datasets)).flat_map(
        self.flatten_zip)

    return single_mtl_dataset

  def eval_metrics(self):
    return [
        metrics.Metrics.ACC, metrics.Metrics.NEG_LOG_PERPLEXITY
    ]

  def update_task_ids(self, encoder):
    """Generate task_ids for each problem.

    These ids correspond to the index of the task in the task_list.

    Args:
      encoder: this provides the size of the vocab which is used to compute
        the index offset.
    """
    primary_task = self.task_list[0]
    id_offset = encoder.vocab_size + text_encoder.NUM_RESERVED_TOKENS
    if hasattr(primary_task, "additional_reserved_tokens"):
      id_offset += len(primary_task.additional_reserved_tokens)

    for idx, _ in enumerate(self.task_list):
      # Subtract one to get actual indices in the context of 0-indexing
      self.task_list[idx].set_task_id(idx + id_offset - 1)
      print(self.task_list[idx].task_id)

  def get_max_num_classes(self):
    """Compute the maximum number of classes any subtask has.

    This is useful for modifying the size of the softmax to include the output
    labels for the classification tasks. Currently, labels from different tasks
    are overloaded.

    Returns:
      num: Highest number of output classes in any text classification sub-task
        within this MultiProblem.
    """
    num = 0
    for task in self.task_list:
      if hasattr(task, "num_classes"):
        if num < task.num_classes:
          num = task.num_classes

    return num
