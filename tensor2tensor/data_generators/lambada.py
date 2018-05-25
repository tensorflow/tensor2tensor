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
"""Data generators for LAMBADA data-sets.


Lmbada as a language modeling task:
  https://arxiv.org/abs/1606.06031

Lmbada as a reading comprehension task:
  https://arxiv.org/abs/1610.08431
  For lambada as reading comprehension task, one can use the dataset that is
  provided here:
  http://ttic.uchicago.edu/~kgimpel/data/lambada-train-valid.tar.gz
  In this dataset samples for which the target word is not in the context are
  removed from the trained data.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf


_UNK = "<UNK>"


_TAR = "lambada-dataset.tar.gz"
_URL = "http://clic.cimec.unitn.it/lambada/" + _TAR
_VOCAB = "lambada-vocab-2.txt"


def _prepare_lambada_data(tmp_dir, data_dir, vocab_size, vocab_filename):
  """Downloading and preparing the dataset.

  Args:
    tmp_dir: tem directory
    data_dir: data directory
    vocab_size: size of vocabulary
    vocab_filename: name of vocab file

  """

  if not tf.gfile.Exists(data_dir):
    tf.gfile.MakeDirs(data_dir)

  file_path = generator_utils.maybe_download(tmp_dir, _TAR, _URL)
  tar_all = tarfile.open(file_path)
  tar_all.extractall(tmp_dir)
  tar_all.close()
  tar_train = tarfile.open(os.path.join(tmp_dir, "train-novels.tar"))
  tar_train.extractall(tmp_dir)
  tar_train.close()

  vocab_path = os.path.join(data_dir, vocab_filename)
  if not tf.gfile.Exists(vocab_path):
    with tf.gfile.GFile(os.path.join(tmp_dir, _VOCAB), "r") as infile:
      reader = csv.reader(infile, delimiter="\t")
      words = [row[0] for row in reader]
      words = [_UNK] + words[:vocab_size]
    with tf.gfile.GFile(vocab_path, "w") as outfile:
      outfile.write("\n".join(words))


def get_dataset_split(tmp_dir, split, use_control_set):
  """Gives the file paths with regards to the given split.

  Args:
    tmp_dir: temp directory
    split: dataset split
    use_control_set: uses control dataset if true.

  Returns:
    list of file paths.

  """
  if not use_control_set:
    dataset_split = {
        problem.DatasetSplit.TRAIN: [
            f for f in tf.gfile.Glob(
                os.path.join(tmp_dir, "train-novels/*/*.txt"))
        ],
        problem.DatasetSplit.EVAL: [
            os.path.join(tmp_dir, "lambada_development_plain_text.txt")
        ],
        problem.DatasetSplit.TEST: [
            os.path.join(tmp_dir, "lambada_test_plain_text.txt")
        ]
    }

  else:
    dataset_split = {
        problem.DatasetSplit.TRAIN: [
            f for f in tf.gfile.Glob(
                os.path.join(tmp_dir, "train-novels/*/*.txt"))
        ],
        problem.DatasetSplit.EVAL: [
            os.path.join(tmp_dir, "lambada_control_test_data_plain_text.txt")
        ],
    }

  return dataset_split[split]


@registry.register_problem
class LambadaLm(text_problems.Text2SelfProblem):
  """Lambada as language modeling task."""

  @property
  def is_generate_per_split(self):
    """If true, a single call to generate_samples generates for a single split.

    Returns:
      Boolean.
    """
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.

    Returns:
      A dict containing splits information.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def vocab_size(self):
    # Similar to the setup of the main paper
    return 60000

  @property
  def oov_token(self):
    return _UNK

  @property
  def use_control_set(self):
    """If evaluate on control set."""
    return False

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generates samples.

    Args:
      data_dir: data directory
      tmp_dir: temp directory
      dataset_split: dataset split

    Returns:
      sample generator

    """
    _prepare_lambada_data(tmp_dir, data_dir, self.vocab_size,
                          self.vocab_filename)
    files = get_dataset_split(tmp_dir, dataset_split, self.use_control_set)

    def _generate_samples():
      """sample generator.

      Yields:
        A dict.

      """
      for filepath in files:
        with tf.gfile.GFile(filepath, "r") as f:
          for line in f:
            line = " ".join(line.split())
            yield {"targets": line}

    return _generate_samples()


@registry.register_problem
class LambadaLmControl(LambadaLm):
  """Lambada as language modeling task on control dataset."""

  @property
  def control_set(self):
    """If test on control set."""
    return False


@registry.register_problem
class LambadaRc(text_problems.Text2ClassProblem):
  """Lambada as reading comprehension task."""

  @property
  def is_generate_per_split(self):
    """If true, a single call to generate_samples generates for a single split.

    Returns:
      Boolean.
    """
    return True

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each.

    Returns:
      A dict containing splits information.
    """
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  @property
  def vocab_size(self):
    # Similar to the setup of the main paper
    return 60000

  @property
  def oov_token(self):
    return _UNK

  @property
  def use_control_set(self):
    """If test on control set."""
    return False

  def get_labels_encoder(self, data_dir):
    """Builds encoder for the given class labels.

    Args:
      data_dir: data directory

    Returns:
      An encoder for class labels.
    """
    label_filepath = os.path.join(data_dir, self.vocab_filename)
    return text_encoder.TokenTextEncoder(
        label_filepath, replace_oov=self.oov_token)

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generates samples.

    Args:
      data_dir: data directory
      tmp_dir: temp directory
      dataset_split: dataset split

    Returns:
      sample generator

    """
    _prepare_lambada_data(tmp_dir, data_dir, self.vocab_size,
                          self.vocab_filename)
    files = get_dataset_split(tmp_dir, dataset_split, self.use_control_set)

    def _generate_samples():
      """sample generator.

      Yields:
        A dict.

      """
      for filepath in files:
        with tf.gfile.GFile(filepath, "r") as f:
          for line in f:
            input_target = line.split()
            yield {
                "inputs": " ".join(input_target[:-1]),
                "label": input_target[-1]
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
    txt_encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    label_encoder = self.get_labels_encoder(data_dir)
    for sample in generator:
      inputs = txt_encoder.encode(sample["inputs"])
      inputs.append(text_encoder.EOS_ID)
      targets = label_encoder.encode(sample["label"])
      yield {"inputs": inputs, "targets": targets}

  def feature_encoders(self, data_dir):
    """Return a dict for encoding and decoding inference input/output.

    Args:
      data_dir: data directory

    Returns:
      A dict of <feature name, TextEncoder>.

    """
    txt_encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
    label_encoder = self.get_labels_encoder(data_dir)
    return {"inputs": txt_encoder, "targets": label_encoder}

  def hparams(self, defaults, unused_model_hparams):
    """Returns problem_hparams.

    Args:
      defaults: default hyperparameters
      unused_model_hparams: model hyperparameters

    """

    p = defaults
    source_vocab_size = self._encoders["inputs"].vocab_size
    num_classes = self._encoders["targets"].vocab_size
    p.input_modality = {
        "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
    }
    p.target_modality = (registry.Modalities.CLASS_LABEL, num_classes)


@registry.register_problem
class LambadaRcControl(LambadaRc):
  """Lambada as reading comprehension task on control dataset."""

  @property
  def control_set(self):
    """If test on control set."""
    return True
