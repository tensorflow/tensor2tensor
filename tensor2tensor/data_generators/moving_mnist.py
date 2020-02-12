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

"""Moving MNIST dataset.

Unsupervised Learning of Video Representations using LSTMs
Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov
https://arxiv.org/abs/1502.04681

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import video_utils
from tensor2tensor.layers import modalities
from tensor2tensor.utils import contrib
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.video import moving_sequence


DATA_URL = (
    "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy")
SPLIT_TO_SIZE = {
    problem.DatasetSplit.TRAIN: 100000,
    problem.DatasetSplit.EVAL: 10000,
    problem.DatasetSplit.TEST: 10000}


@registry.register_problem
class VideoMovingMnist(video_utils.VideoProblem):
  """MovingMnist Dataset."""

  @property
  def num_channels(self):
    return 1

  @property
  def frame_height(self):
    return 64

  @property
  def frame_width(self):
    return 64

  @property
  def is_generate_per_split(self):
    return True

  # num_videos * num_frames
  @property
  def total_number_of_frames(self):
    return 100000 * 20

  def max_frames_per_video(self, hparams):
    return 20

  @property
  def random_skip(self):
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [
        {"split": problem.DatasetSplit.TRAIN, "shards": 10},
        {"split": problem.DatasetSplit.EVAL, "shards": 1},
        {"split": problem.DatasetSplit.TEST, "shards": 1}]

  @property
  def extra_reading_spec(self):
    """Additional data fields to store on disk and their decoders."""
    data_fields = {
        "frame_number": tf.FixedLenFeature([1], tf.int64),
    }
    decoders = {
        "frame_number":
            contrib.slim().tfexample_decoder.Tensor(tensor_key="frame_number"),
    }
    return data_fields, decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.VIDEO,
                  "targets": modalities.ModalityType.VIDEO}
    p.vocab_size = {"inputs": 256,
                    "targets": 256}

  def get_test_iterator(self, tmp_dir):
    path = generator_utils.maybe_download(
        tmp_dir, os.path.basename(DATA_URL), DATA_URL)
    with tf.io.gfile.GFile(path, "rb") as fp:
      mnist_test = np.load(fp)
    mnist_test = np.transpose(mnist_test, (1, 0, 2, 3))
    mnist_test = np.expand_dims(mnist_test, axis=-1)
    mnist_test = tf.data.Dataset.from_tensor_slices(mnist_test)
    return mnist_test.make_initializable_iterator()

  def map_fn(self, image, label):
    sequence = moving_sequence.image_as_moving_sequence(
        image, sequence_length=20)
    return sequence.image_sequence

  def get_train_iterator(self):
    mnist_ds = tfds.load("mnist:3.*.*", split=tfds.Split.TRAIN,
                         as_supervised=True)
    mnist_ds = mnist_ds.repeat()
    moving_mnist_ds = mnist_ds.map(self.map_fn).batch(2)
    moving_mnist_ds = moving_mnist_ds.map(lambda x: tf.reduce_max(x, axis=0))
    return moving_mnist_ds.make_initializable_iterator()

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    with tf.Graph().as_default():
      # train and eval set are generated on-the-fly.
      # test set is the official test-set.
      if dataset_split == problem.DatasetSplit.TEST:
        moving_ds = self.get_test_iterator(tmp_dir)
      else:
        moving_ds = self.get_train_iterator()

      next_video = moving_ds.get_next()
      with tf.Session() as sess:
        sess.run(moving_ds.initializer)

        n_samples = SPLIT_TO_SIZE[dataset_split]
        for _ in range(n_samples):
          next_video_np = sess.run(next_video)
          for frame_number, frame in enumerate(next_video_np):
            yield {
                "frame_number": [frame_number],
                "frame": frame,
            }
