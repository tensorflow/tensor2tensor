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
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import discretization
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
import tensorflow as tf


class MixingSchedule(object):
  """Available schedules for mixing datasets."""
  EXPONENTIAL = "exponential"
  CONSTANT = "constant"
  PRETRAIN = "pretrain"


class MultiProblem(problem.Problem):
  """MultiProblem base class."""

  _ADDED_EVAL_COUNT = 20000

  def __init__(self, was_reversed=False, was_copy=False):
    super(MultiProblem, self).__init__(was_reversed, was_copy)
    self.task_list = []

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    assert len(self.task_list) > 1

    for task in self.task_list:
      task.generate_data(data_dir, tmp_dir, task_id)

  def add_task_id(self, task, example, encoder, hparams, is_infer):
    """Convert example to code switching mode by adding a task id."""
    if task.has_inputs:
      example["inputs"] = example["inputs"][:-1]  # remove EOS token

    if hasattr(task, "class_labels"):
      if self.vocab_type == text_problems.VocabType.CHARACTER:
        # TODO(urvashik): handle the case where num_labels > 9
        example["targets"] = tf.cast(discretization.int_to_bit(
            example["targets"], 1, base=10) + 50, tf.int64)
        example["targets"] = tf.squeeze(example["targets"], axis=[-1])
      elif self.vocab_type == text_problems.VocabType.SUBWORD:
        offset = encoder.vocab_size + len(self.task_list)
        example["targets"] = offset + example["targets"]
    else:
      # sequence with inputs and targets eg: summarization
      if task.has_inputs:
        if hparams.multiproblem_max_input_length > 0:
          example["inputs"] = example[
              "inputs"][:hparams.multiproblem_max_input_length]
        # Do not truncate targets during inference with beam decoding.
        if hparams.multiproblem_max_target_length > 0 and not is_infer:
          example["targets"] = example[
              "targets"][:hparams.multiproblem_max_target_length]

    if task.has_inputs:
      if is_infer:
        concat_list = [example["inputs"], [task.task_id]]
        example["inputs"] = tf.concat(concat_list, 0)
      else:
        inputs = example.pop("inputs")
        concat_list = [inputs, [task.task_id], example["targets"]]
        example["targets"] = tf.concat(concat_list, 0)
    else:
      concat_list = [[task.task_id], example["targets"]]
      example["targets"] = tf.concat(concat_list, 0)

    min_task_id = min([t.task_id for t in self.task_list])
    example["task_id"] = tf.constant([task.task_id - min_task_id],
                                     dtype=tf.int64)
    return example

  def filepattern(self, data_dir, mode, shard=None):
    print("Generating multi problem filepattern")
    return [task.filepattern(data_dir, mode, shard) for task in self.task_list]

  def get_hparams(self, model_hparams=None):
    if self._hparams is not None:
      return self._hparams

    self._hparams = self.task_list[0].get_hparams(model_hparams)
    # Increase the vocab size to account for task ids and modify the modality.
    vocab_size_inc = len(self.task_list)
    vocab_size_inc += self.get_max_num_classes()
    vocab_size = self._hparams.vocabulary["targets"].vocab_size
    new_vocab_size = vocab_size + vocab_size_inc
    if model_hparams.multiproblem_vocab_size > new_vocab_size:
      new_vocab_size = model_hparams.multiproblem_vocab_size
    tf.logging.info("Old vocabulary size: %d" % vocab_size)
    tf.logging.info("New vocabulary size: %d" % new_vocab_size)
    self._hparams.vocab_size["targets"] = new_vocab_size
    self._hparams.modality["targets"] = modalities.SymbolModality(
        model_hparams, self._hparams.vocab_size["targets"])

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
              shuffle_buffer_size=1024,
              max_records=-1,
              only_last=False):

    # A list of datasets corresponding to the tasks in the task_list object
    # that need to be mixed.
    datasets = []
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    is_infer = mode == tf.estimator.ModeKeys.PREDICT

    primary_task = self.task_list[0]
    if primary_task.has_inputs:
      raise ValueError("Only support language models as primary problem which "
                       "supplies the vocabulary and the hparams.")
    enc = primary_task.feature_encoders(data_dir=data_dir)["targets"]

    for idx, task in enumerate(self.task_list):
      task_dataset = task.dataset(mode=mode,
                                  data_dir=data_dir,
                                  num_threads=num_threads,
                                  output_buffer_size=output_buffer_size,
                                  shuffle_files=shuffle_files,
                                  hparams=hparams,
                                  preprocess=preprocess,
                                  dataset_split=dataset_split,
                                  shard=shard,
                                  partition_id=partition_id,
                                  num_partitions=num_partitions,
                                  shuffle_buffer_size=shuffle_buffer_size,
                                  max_records=max_records,
                                  only_last=only_last)

      if idx == 0:
        self.update_task_ids(enc)

      if is_training:
        task_dataset = task_dataset.repeat()

      # pylint: disable=cell-var-from-loop
      task_dataset = task_dataset.map(
          lambda x: self.add_task_id(task, x, enc, hparams, is_infer))

      if not is_training and not is_infer:
        zeros = tf.zeros([self._ADDED_EVAL_COUNT, 1], dtype=tf.int64)
        pad_data = tf.data.Dataset.from_tensor_slices({
            "targets": zeros,
            "batch_prediction_key": zeros,
            "task_id": zeros,
        })
        task_dataset = task_dataset.concatenate(pad_data)

      datasets.append(task_dataset)

    # Setup the problem hparams by setting them to the LM task hparams.
    self.get_hparams(model_hparams=hparams)

    if is_training:
      # Using tf.Variable instead of get_variable to work around issues with
      # queues on multiple hosts. Note that this will separately count steps
      # on each host that's feeding the data, so in a large-scale setting you
      # may need to adjust hparams for that. For example, a 4x4 slice of a TPU
      # pod may use 2 data hosts, so we'll be only adding 1 here once for 2
      # examples -- divide the corresponding hparams by 2 to compensate.
      problem_step = tf.Variable(tf.constant(0, dtype=tf.int64),
                                 trainable=False, use_resource=True,
                                 dtype=tf.int64, name="problem_step")
      dataset_iterators = [d.make_one_shot_iterator() for d in datasets]

      def get_next_from_dataset(dataset_iter):
        return dataset_iter.get_next()

      def get_exp_sched_prob():
        """Inverse decay exponential to mix datasets."""
        with tf.control_dependencies([problem_step.assign_add(1)]):
          inv_exp_decay = common_layers.inverse_exp_decay(
              max_step=hparams.multiproblem_schedule_max_examples,
              min_value=1e-4,
              step=tf.to_float(problem_step)
          )
          # inv_exp_decay is bounded above by 1.0
          return inv_exp_decay * hparams.multiproblem_schedule_threshold

      def get_const_sched_prob():
        return hparams.multiproblem_schedule_threshold

      def get_pretrain_sched_prob():
        """Pretrain the primary tasks for max examples."""
        with tf.control_dependencies([problem_step.assign_add(1)]):
          return tf.cond(
              tf.greater(problem_step,
                         tf.cast(hparams.multiproblem_schedule_max_examples,
                                 dtype=tf.int64)),
              lambda: 1.0, lambda: 0.0)

      def mix_data(example):
        """Function to mix the different datasets according to a schedule."""
        del example
        # This block computes the probability of mixing the primary task with
        # the secondary tasks. 0 = only the primary task, 1 = only the secondary
        # tasks.
        if hparams.multiproblem_mixing_schedule == MixingSchedule.EXPONENTIAL:
          prob = get_exp_sched_prob()
          prob = tf.cond(
              tf.equal(tf.floormod(
                  problem_step, tf.cast(5e6, dtype=tf.int64)), 0),
              lambda: tf.Print(prob, [prob], message="Probability"),
              lambda: prob)
        elif hparams.multiproblem_mixing_schedule == MixingSchedule.CONSTANT:
          prob = get_const_sched_prob()
        elif hparams.multiproblem_mixing_schedule == MixingSchedule.PRETRAIN:
          prob = get_pretrain_sched_prob()
        else:
          raise ValueError("Unknown schedule %s" % str(
              hparams.multiproblem_mixing_schedule))
        tf.logging.info("Using the %s schedule to "
                        "train the MultiProblem." % str(
                            hparams.multiproblem_mixing_schedule))
        tf.logging.info("Schedule mixing threshold "
                        "%.2f" % hparams.multiproblem_schedule_threshold)

        def sample_task(curr_task, num_tasks_left, randnum):
          """A recursive function to sample a task.

          This function treats the probability as the threshold for the primary
          task and divides the remaining probability mass across the other
          tasks.

          Args:
            curr_task: The index of the task being considered for sampling.
            num_tasks_left: Number of tasks remaining to possibly sample from.
            randnum: The random number used to select the dataset.

          Returns:
            A Tensor representing an example from the task that was sampled
            from.
          """

          if num_tasks_left == 0:
            return get_next_from_dataset(dataset_iterators[curr_task])

          # When curr_task is 0, the primary task, the new prob is the same as
          # the original probability. `tf.greater` indicates that the primary
          # task receives (1-prob) of the probability mass.
          # Otherwise, `prob` is divided equally amongst all the secondary
          # tasks.
          new_prob = prob - (curr_task * prob / (len(self.task_list)-1))
          return tf.cond(
              tf.greater(randnum, new_prob),
              lambda: get_next_from_dataset(dataset_iterators[curr_task]),
              lambda: sample_task(curr_task+1, num_tasks_left-1, randnum)
          )

        return tf.data.Dataset.from_tensors(
            sample_task(0, len(self.task_list)-1, tf.random_uniform([])))

      single_mtl_dataset = tf.data.Dataset.from_tensors(tf.zeros([1])).repeat()
      single_mtl_dataset = single_mtl_dataset.flat_map(mix_data)

    else:
      if hparams.multiproblem_target_eval_only:
        single_mtl_dataset = datasets[1]
      else:
        single_mtl_dataset = tf.data.Dataset.zip(tuple(datasets)).flat_map(
            self.flatten_zip)

    return single_mtl_dataset

  def eval_metrics(self):
    for task in self.task_list:
      if "summarize" in task.name:
        return [
            metrics.Metrics.ACC, metrics.Metrics.NEG_LOG_PERPLEXITY,
            metrics.Metrics.ROUGE_2_F, metrics.Metrics.ROUGE_L_F
        ]
    return [
        metrics.Metrics.ACC, metrics.Metrics.NEG_LOG_PERPLEXITY,
    ]

  def update_task_ids(self, encoder):
    """Generate task_ids for each problem.

    These ids correspond to the index of the task in the task_list.

    Args:
      encoder: this provides the size of the vocab which is used to compute
        the index offset.
    """
    offset = encoder.vocab_size

    for idx, _ in enumerate(self.task_list):
      self.task_list[idx].set_task_id(idx + offset)
      print(self.task_list[idx].name)
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


def aggregate_task_losses(hparams,
                          problem_hparams,
                          logits,
                          target_modality,
                          feature):
  """Multiproblem loss function."""
  summaries = []
  main_task_id = hparams.problem.task_list[0].task_id
  # Primary task loss
  loss_num, loss_den = target_modality.loss(
      logits, feature,
      weights_fn=
      lambda x: common_layers.weights_multi_problem_all(x, main_task_id))

  loss_val = loss_num / tf.maximum(1.0, loss_den)
  summaries.append([hparams.problem.task_list[0].name+"_loss", loss_val])

  # Since the losses may undergo rescaling, they cannot exist as separate
  # numerators and denominators. Set the denominators to 1 in order to faciliate
  # loss averaging.
  loss_num = loss_val
  loss_den = tf.minimum(tf.convert_to_tensor(1, dtype=tf.float32), loss_den)

  for task in hparams.problem.task_list[1:]:
    # Loss only from the input sequence -- the auxiliary LM loss.
    seq_loss_num, seq_loss_den = target_modality.loss(
        logits, feature,
        weights_fn=
        lambda x: common_layers.weights_multi_problem_input(x, task.task_id))  # pylint: disable=cell-var-from-loop
    seq_loss_num *= problem_hparams.loss_multiplier

    # Unscaled sequence loss.
    seq_loss = seq_loss_num / tf.maximum(1.0, seq_loss_den)
    summaries.append([task.name+"_seq_loss", seq_loss])

    if hasattr(task, "num_classes"):
      # Loss only from the classification label.
      label_loss_num, label_loss_den = target_modality.loss(
          logits, feature,
          weights_fn=
          lambda x: common_layers.weights_multi_problem(x, task.task_id))  # pylint: disable=cell-var-from-loop
      label_loss_num *= problem_hparams.loss_multiplier

      # Unscaled classification label loss.
      label_loss = label_loss_num / tf.maximum(1.0, label_loss_den)
      summaries.append([task.name+"_label_loss", label_loss])

      # Scaling.
      if hparams.multiproblem_reweight_label_loss:
        label_loss *= hparams.multiproblem_label_weight
        seq_loss *= (1 - hparams.multiproblem_label_weight)

      if hparams.multiproblem_class_loss_multiplier:
        label_loss *= hparams.multiproblem_class_loss_multiplier
        summaries.append([task.name+"_scaled_label_loss", label_loss])

      # This is the training loss for the optimizer after all the scaling.
      task_loss_val = seq_loss + label_loss

      loss_den_ = label_loss_den

    else:
      # Loss only from the target sequence.
      target_loss_num, target_loss_den = target_modality.loss(
          logits, feature,
          weights_fn=
          lambda x: common_layers.weights_multi_problem(x, task.task_id))  # pylint: disable=cell-var-from-loop
      target_loss_num *= problem_hparams.loss_multiplier

      # Unscaled target sequence loss.
      target_loss = target_loss_num / tf.maximum(1.0, target_loss_den)
      summaries.append([task.name+"_target_loss", target_loss])

      # Scaling.
      if hparams.multiproblem_reweight_label_loss:
        target_loss *= hparams.multiproblem_label_weight
        seq_loss *= (1 - hparams.multiproblem_label_weight)

      # This is the training loss for the optimizer after all the scaling.
      task_loss_val = seq_loss + target_loss

      loss_den_ = target_loss_den

    summaries.append([task.name+"_loss", task_loss_val])
    # Adding 1 to the loss den for each task leads to averaging task losses.
    # TODO(urvashik): Fix combination with other task losses - weighted
    # average based on the number of examples from that task.
    loss_num += task_loss_val
    loss_den += tf.minimum(tf.convert_to_tensor(1, dtype=tf.float32),
                           loss_den_)

  return loss_num, loss_den, summaries
