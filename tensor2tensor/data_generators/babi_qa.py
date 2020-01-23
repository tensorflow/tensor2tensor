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

r"""Data generators for bAbi question answering dataset.


The dataset consists of 20 tasks for testing text understanding and reasoning
in the bAbI project (https://research.fb.com/downloads/babi/). The aim is that
each task tests a unique aspect of text and reasoning, and hence test different
capabilities of learning models. For more information check the following paper:
Jason Weston, Antoine Bordes, Sumit Chopra and Tomas Mikolov. Towards AI
Complete Question Answering: A Set of Prerequisite Toy Tasks, arXiv:1502.05698.
Available at: http://arxiv.org/abs/1502.05698

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil
import tarfile
import requests
import six

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


_DIR_NAME = "tasks_1-20_v1-2"
_TAR = _DIR_NAME + ".tar.gz"
_URL = "http://www.thespermwhale.com/jaseweston/babi/" + _TAR

_TASKS = {
    "qa0": "qa0_all-tasks",
    "qa1": "qa1_single-supporting-fact",
    "qa2": "qa2_two-supporting-facts",
    "qa3": "qa3_three-supporting-facts",
    "qa4": "qa4_two-arg-relations",
    "qa5": "qa5_three-arg-relations",
    "qa6": "qa6_yes-no-questions",
    "qa7": "qa7_counting",
    "qa8": "qa8_lists-sets",
    "qa9": "qa9_simple-negation",
    "qa10": "qa10_indefinite-knowledge",
    "qa11": "qa11_basic-coreference",
    "qa12": "qa12_conjunction",
    "qa13": "qa13_compound-coreference",
    "qa14": "qa14_time-reasoning",
    "qa15": "qa15_basic-deduction",
    "qa16": "qa16_basic-induction",
    "qa17": "qa17_positional-reasoning",
    "qa18": "qa18_size-reasoning",
    "qa19": "qa19_path-finding",
    "qa20": "qa20_agents-motivations"
}

# A list of problem names that are registered by this module. This will get
# populated at module load time in the code at the bottom of this file.
REGISTERED_PROBLEMS = []


def _normalize_string(raw_str):
  """Normalizes the string using tokenizer.encode.

  Args:
    raw_str: the input string

  Returns:
   A string which is ready to be tokenized using split()
  """
  return " ".join(
      token.strip()
      for token in tokenizer.encode(text_encoder.native_to_unicode(raw_str)))


def _prepare_babi_data(tmp_dir, data_dir):
  """Downloads and extracts the dataset.

  Args:
    tmp_dir: temp directory to download and extract the dataset
    data_dir: The base directory where data and vocab files are stored.

  Returns:
    tmp_dir: temp directory containing the raw data.
  """
  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  file_path = os.path.join(tmp_dir, _TAR)
  headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_1) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/63.0.3239.132 Safari/537.36"}
  resp = requests.get(_URL, headers=headers)
  with open(file_path, "wb") as f:
    f.write(resp.content)

  tar = tarfile.open(file_path)
  tar.extractall(tmp_dir)
  tar.close()

  return tmp_dir


def _build_vocab(generator, vocab_dir, vocab_name):
  """Build a vocabulary from examples.

  Args:
    generator: text generator for creating vocab.
    vocab_dir: directory where to save the vocabulary.
    vocab_name: vocab file name.

  Returns:
    text encoder.
  """
  vocab_path = os.path.join(vocab_dir, vocab_name)
  if not tf.gfile.Exists(vocab_path):
    data = []
    for line in generator:
      data.extend(line.split())
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    encoder = text_encoder.TokenTextEncoder(None, vocab_list=words)
    encoder.store_to_file(vocab_path)
  else:
    encoder = text_encoder.TokenTextEncoder(vocab_path)
  return encoder


def _babi_parser(tmp_dir,
                 babi_task_id,
                 subset,
                 dataset_split,
                 joint_training=True):
  """Parsing the bAbi dataset (train and test).

  Args:
    tmp_dir: temp directory to download and extract the dataset
    babi_task_id: babi task id
    subset: babi subset
    dataset_split: dataset split (train or eval)
    joint_training: if training the model on all tasks.

  Returns:
     babi_instances: set of training examples, each a dict containing a story,
     a question and an answer.
     babi_lines: all the texts in the data separated based on their
     appearance in the stories, questions, or answers.
  """

  def _data_file(mode, task_id):
    """Generates the path to the data file for the given mode(train/test).

    Args:
      mode: either train or test for bAbi dataset
      task_id: babi task id

    Returns:
      data file path
    """
    file_name = (_TASKS[task_id] + "_{}.txt")
    return os.path.join(_DIR_NAME, subset, file_name.format(mode))

  def _all_task_raw_data_generator(tmp_dir, data_file, dataset_split):
    """Prepares raw data for all tasks to gether..

    Args:
      tmp_dir: temp directory
      data_file: data file
      dataset_split: dataset split
    """

    tf.logging.info("Preparing dataset of all task together")
    globe_name = ("*_{}.txt")
    mode_name = "test"
    if dataset_split == problem.DatasetSplit.TRAIN:
      mode_name = "train"
    files_name = os.path.join(
        tmp_dir, _DIR_NAME, subset,
        globe_name.format(mode_name))
    with tf.gfile.GFile(data_file, "wb") as outfile:
      for filename in tf.gfile.Glob(files_name):
        if filename == data_file:
          # don"t want to copy the output into the output
          continue
        with tf.gfile.GFile(filename, "rb") as readfile:
          shutil.copyfileobj(readfile, outfile)

  def _parse_answer(answer):
    if (joint_training or babi_task_id in ["qa8", "qa19", "qa0"
                                          ]):  # "lists-sets" or "path finding"
      return "".join([d for d in answer.split(",")])  # as a single token!
    else:
      return answer

  if dataset_split == problem.DatasetSplit.TRAIN:
    babi_train_task_id = "qa0" if joint_training else babi_task_id
    data_file = os.path.join(tmp_dir, _data_file("train", babi_train_task_id))
  else:
    data_file = os.path.join(tmp_dir, _data_file("test", babi_task_id))

  if ((babi_task_id == "qa0" or joint_training) and
      not tf.gfile.Exists(os.path.join(tmp_dir, data_file))):
    _all_task_raw_data_generator(tmp_dir, data_file, dataset_split)

  tf.logging.info("Parsing %s into training/testing instances...", data_file)

  babi_instances = []
  with tf.gfile.GFile(data_file, mode="r") as f:
    story = []
    for line in f:
      line_num, line = line.strip().split(" ", 1)
      if int(line_num) == 1:
        story = []
      if "\t" in line:
        question, answer, _ = line.split("\t")
        question = _normalize_string(question)
        substories = [s for s in story if s]
        answer = _parse_answer(answer)
        instance = {
            FeatureNames.STORY: substories,
            FeatureNames.QUESTION: question,
            FeatureNames.ANSWER: answer
        }
        babi_instances.append(instance)

        story.append("")
      else:
        story.append(_normalize_string(line))

  return babi_instances


class FeatureNames(object):
  """Feature names, i.e keys for storing babi_qa data in TFExamples."""
  STORY = "story"
  QUESTION = "question"
  ANSWER = "answer"

  @classmethod
  def features(cls):
    for attr, value in cls.__dict__.items():
      if not attr.startswith("__") and not callable(getattr(cls, attr)):
        yield value


class BabiQa(text_problems.QuestionAndContext2TextProblem):
  """Base class for bAbi question answering problems."""

  def __init__(self, *args, **kwargs):

    super(BabiQa, self).__init__(*args, **kwargs)
    assert not self._was_reversed, "This problem is not reversible!"
    assert not self._was_copy, "This problem is not copyable!"

  @property
  def babi_subset(self):
    """The subset of dataset.

    This should be one of the following:
    {"en", "en-10k", "shuffled", "shuffled-10k"}
    """
    raise NotImplementedError

  @property
  def babi_task_id(self):
    """The id of the babi task.

    This should be one of the following:
    {"qa0", "qa1", "qa1",..."q20"}, where qa0 means all tasks together.
    """
    raise NotImplementedError

  def dataset_filename(self):
    return "babi_qa_" + self.babi_subset + "_" + _TASKS[self.babi_task_id]

  @property
  def vocab_file(self):
    return self.babi_subset + "_" + _TASKS[self.babi_task_id] + ".vocab"

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def is_generate_per_split(self):
    return True

  @property
  def joint_training(self):
    # training on data from all tasks.
    return True

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  def get_labels_encoder(self, data_dir):
    """Builds encoder for the given class labels.

    Args:
      data_dir: data directory

    Returns:
      An encoder for class labels.
    """
    label_filepath = os.path.join(data_dir, self.vocab_filename)
    return text_encoder.TokenTextEncoder(label_filepath)

  def generate_samples(self, data_dir, tmp_dir, dataset_split):

    tmp_dir = _prepare_babi_data(tmp_dir, data_dir)
    _build_vocab(
        self.generate_text_for_vocab(data_dir, tmp_dir), data_dir,
        self.vocab_filename)
    examples = _babi_parser(tmp_dir, self.babi_task_id, self.babi_subset,
                            dataset_split, self.joint_training)

    def _generate_samples():
      """sample generator.

      Yields:
        A dict.

      """
      for example in examples:
        context = " ".join(example[FeatureNames.STORY])
        yield {
            "context": " ".join(context.split()),
            "inputs": " ".join(example[FeatureNames.QUESTION].split()),
            "targets": example[FeatureNames.ANSWER]
        }

    return _generate_samples()

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    """A generator that generates samples that are encoded.

    Args:
      data_dir: data directory
      tmp_dir: temp directory
      dataset_split: dataset split

    Yields:
      A dict.

    """
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    label_encoder = self.get_labels_encoder(data_dir)
    for sample in generator:
      inputs = encoder.encode(sample["inputs"])
      inputs.append(text_encoder.EOS_ID)
      context = encoder.encode(sample["context"])
      context.append(text_encoder.EOS_ID)
      targets = label_encoder.encode(sample["targets"])
      sample["targets"] = targets
      yield {"inputs": inputs, "context": context, "targets": targets}

  def feature_encoders(self, data_dir):
    """Return a dict for encoding and decoding inference input/output.

    Args:
      data_dir: data directory

    Returns:
      A dict of <feature name, TextEncoder>.

    """
    encoders = (super(BabiQa, self).feature_encoders(data_dir))
    label_encoder = self.get_labels_encoder(data_dir)
    encoders["targets"] = label_encoder  # bAbi as a classification task
    return encoders

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    # NOTE: for babi, we create the vocab from both train and test data.
    for dataset_split in [
        problem.DatasetSplit.TRAIN, problem.DatasetSplit.EVAL
    ]:

      for example in _babi_parser(tmp_dir, self.babi_task_id, self.babi_subset,
                                  dataset_split, self.joint_training):

        context = " ".join(example[FeatureNames.STORY])
        yield " ".join(context.split())
        yield " ".join(example[FeatureNames.QUESTION].split())
        yield example[FeatureNames.ANSWER]

  def hparams(self, defaults, unused_model_hparams):
    """Returns problem_hparams.

    Args:
      defaults: default hyperparameters
      unused_model_hparams: model hyperparameters

    """
    (super(BabiQa, self).hparams(defaults, unused_model_hparams))
    p = defaults
    num_classes = self._encoders["targets"].vocab_size
    p.modality = {"targets": modalities.ModalityType.CLASS_LABEL}
    p.vocab_size = {"targets": num_classes}

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = (
        super(BabiQa, self).example_reading_spec())
    data_fields["targets"] = tf.FixedLenFeature([1], tf.int64)
    return (data_fields, data_items_to_decoders)

  def eval_metrics(self):
    """Specify the set of evaluation metrics for this problem.

    Returns:
      List of evaluation metrics of interest.
    """
    return [metrics.Metrics.ACC]


class BabiQaConcat(BabiQa):
  """Babi with question and story concatenated together as inputs."""

  def preprocess_example(self, example, unused_mode, unused_model_hparams):
    sep = tf.convert_to_tensor([self.QUESTION_SEPARATOR_ID],
                               dtype=example["inputs"].dtype)
    example["inputs"] = tf.concat([example["inputs"], sep, example["context"]],
                                  0)
    return example

  def hparams(self, defaults, unused_model_hparams):
    super(BabiQaConcat, self).hparams(defaults, unused_model_hparams)
    p = defaults

    if "context" in p.modality:
      del p.modality["context"]

    if "context" in p.vocab_size:
      del p.vocab_size["context"]


def _problems_to_register():
  """Problems for which we want to create datasets.

  To avoid a long file with class definition boilerplate for each problem, we
  are dynamically creating and registering problems. The set of problems to
  register is defined by this function. See below for the code that creates the
  classes and registers the problems.

  Returns:
    A dictionary mapping problem name to babi_task_id.
  """
  all_problems = {}

  # First define some problems using only concrete characters (i.e., no meta
  # characters).
  problems_on_different_tasks = {
      "AllTasks": "qa0",
      "Task1": "qa1",
      "Task2": "qa2",
      "Task3": "qa3",
      "Task4": "qa4",
      "Task5": "qa5",
      "Task6": "qa6",
      "Task7": "qa7",
      "Task8": "qa8",
      "Task9": "qa9",
      "Task10": "qa10",
      "Task11": "qa11",
      "Task12": "qa12",
      "Task13": "qa13",
      "Task14": "qa14",
      "Task15": "qa15",
      "Task16": "qa16",
      "Task17": "qa17",
      "Task18": "qa18",
      "Task19": "qa19",
      "Task20": "qa20",
  }
  all_problems.update(problems_on_different_tasks)

  return all_problems


def _register_babi_problems():
  """It dynamically instantiates a class for each babi subsets-tasks.

   @registry.register_problem
   class BabiQaConcatAllTasks_10k(EditSequenceRegexProblem):
     @property
     def babi_task_id(self):
       return "qa0"
     @property
     def babi_subset(self):
      return "en-10k"

  It does not put the classes into the global namespace, so to access the class
  we rely on the registry or this module"s REGISTERED_PROBLEMS list.
  It will be available as

     registry.problem("babi_qa_concat_all_tasks_10k")

  i.e., change camel case to snake case. Numbers are considered lower case
  characters for these purposes.
  """
  for (subset, subset_suffix) in [("en", "_1k"), ("en-10k", "_10k")]:
    for problem_name, babi_task_id in six.iteritems(_problems_to_register()):
      problem_class = type("BabiQaConcat" + problem_name + subset_suffix,
                           (BabiQaConcat,), {
                               "babi_task_id": babi_task_id,
                               "babi_subset": subset
                           })
      registry.register_problem(problem_class)
      REGISTERED_PROBLEMS.append(problem_class.name)


_register_babi_problems()
