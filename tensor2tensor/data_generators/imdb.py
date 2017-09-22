# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""IMDB Sentiment Classification Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf

# End-of-sentence marker.
EOS = text_encoder.EOS_ID


@registry.register_problem
class SentimentIMDB(problem.Problem):
  """IMDB sentiment classification."""
  URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

  @property
  def num_shards(self):
    return 10

  @property
  def vocab_file(self):
    return "sentiment_imdb.vocab"

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8k vocab suffices for this small dataset.

  def doc_generator(self, imdb_dir, dataset, include_label=False):
    dirs = [(os.path.join(imdb_dir, dataset, "pos"), True), (os.path.join(
        imdb_dir, dataset, "neg"), False)]

    for d, label in dirs:
      for filename in os.listdir(d):
        with tf.gfile.Open(os.path.join(d, filename)) as imdb_f:
          doc = imdb_f.read().strip()
          if include_label:
            yield doc, label
          else:
            yield doc

  def generator(self, data_dir, tmp_dir, train):
    """Generate examples."""
    # Download and extract
    compressed_filename = os.path.basename(self.URL)
    download_path = generator_utils.maybe_download(tmp_dir, compressed_filename,
                                                   self.URL)
    imdb_dir = os.path.join(tmp_dir, "aclImdb")
    if not tf.gfile.Exists(imdb_dir):
      with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # Generate vocab
    encoder = generator_utils.get_or_generate_vocab_inner(
        data_dir, self.vocab_file, self.targeted_vocab_size,
        self.doc_generator(imdb_dir, "train"))

    # Generate examples
    dataset = "train" if train else "test"
    for doc, label in self.doc_generator(imdb_dir, dataset, include_label=True):
      yield {
          "inputs": encoder.encode(doc) + [EOS],
          "targets": [int(label)],
      }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(data_dir, 1, shuffled=False)
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True), train_paths,
        self.generator(data_dir, tmp_dir, False), dev_paths)

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    source_vocab_size = self._encoders["inputs"].vocab_size
    p.input_modality = {
        "inputs": (registry.Modalities.SYMBOL, source_vocab_size)
    }
    p.target_modality = (registry.Modalities.CLASS_LABEL, 2)
    p.input_space_id = problem.SpaceID.EN_TOK
    p.target_space_id = problem.SpaceID.GENERIC

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    return {
        "inputs": encoder,
        "targets": text_encoder.ClassLabelEncoder(["neg", "pos"]),
    }

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.FixedLenFeature([1], tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)
