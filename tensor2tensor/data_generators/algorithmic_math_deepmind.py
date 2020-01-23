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

r"""Data generators for the DeepMind Mathematics Dataset.

See https://github.com/deepmind/mathematics_dataset for the original repository.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


_URL = "https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"


@registry.register_problem
class AlgorithmicMathDeepmindAll(text_problems.Text2TextProblem):
  """DeepMind Mathematics Problem, v1.0, all data."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 128,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Downloads and extracts the dataset and generates examples.

    Args:
      data_dir: The base directory where data and vocab files are stored.
      tmp_dir: temp directory to download and extract the dataset.
      dataset_split: split of the data-set.

    Yields:
      The data examples.
    """
    # Create directories if needed.
    if not tf.gfile.Exists(tmp_dir):
      tf.gfile.MakeDirs(tmp_dir)
    if not tf.gfile.Exists(data_dir):
      tf.gfile.MakeDirs(data_dir)

    # Download and extract the data.
    filename = os.path.basename(_URL)
    path = generator_utils.maybe_download(tmp_dir, filename, _URL)
    tarfile.open(path, "r:gz").extractall(tmp_dir)

    # Create the list of directories with data files.
    train_dirs = ["v1.0/train-easy", "v1.0/train-medium", "v1.0/train-hard"]
    eval_dirs = ["v1.0/interpolate", "v1.0/extrapolate"]
    dirs = eval_dirs
    if dataset_split == problem.DatasetSplit.TRAIN:
      dirs = train_dirs
    dirs = [os.path.join(tmp_dir, d) for d in dirs]

    # Iterate over directories and files generating examples.
    for d in dirs:
      files = tf.gfile.Glob(d + "/*.txt")
      for fname in files:
        # In each text file, the first line is the input, the next the answer,
        # and so on until the end of the file.
        cur_input = None
        with tf.gfile.Open(fname, "rb") as f:
          for line in f:
            if cur_input is None:
              cur_input = line.strip()
            else:
              yield {"inputs": cur_input, "targets": line.strip()}
              cur_input = None
