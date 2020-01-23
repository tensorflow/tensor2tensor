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

"""Yelp dataset."""

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


@registry.register_problem
class SentimentYelpPolarity(text_problems.Text2ClassProblem):
  """Yelp dataset."""
  URL = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz"

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**13  # 8k vocab suffices for this small dataset.

  @property
  def num_classes(self):
    return 2

  def class_labels(self, data_dir):
    del data_dir
    return ["1", "2"]

  def doc_generator(self, yelp_dir, dataset, include_label=False):

    file_path = os.path.join(yelp_dir, dataset + ".csv")
    with tf.gfile.Open(file_path) as yelp_f:
      lines = yelp_f.readlines()
      for line in lines:
        label = line[1]
        doc = line[5:-2].strip()
        if include_label:
          yield doc, label
        else:
          yield doc

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate examples."""
    # Download and extract
    compressed_filename = os.path.basename(self.URL)
    download_path = generator_utils.maybe_download(tmp_dir, compressed_filename,
                                                   self.URL)
    yelp_dir = os.path.join(tmp_dir, "yelp_review_polarity_csv")
    if not tf.gfile.Exists(yelp_dir):
      with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # Generate examples
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset = "train" if train else "test"
    for doc, label in self.doc_generator(yelp_dir, dataset, include_label=True):
      yield {
          "inputs": doc,
          "label": int(label),
      }


@registry.register_problem
class SentimentYelpPolarityCharacters(SentimentYelpPolarity):
  """Yelp dataset, character level."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_CHR_SENT
