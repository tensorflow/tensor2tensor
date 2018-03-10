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

"""MNIST."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import random

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry

import tensorflow as tf

# URLs and filenames for MNIST data.
_MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
_MNIST_TRAIN_DATA_FILENAME = "train-images-idx3-ubyte.gz"
_MNIST_TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte.gz"
_MNIST_TEST_DATA_FILENAME = "t10k-images-idx3-ubyte.gz"
_MNIST_TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz"
_MNIST_IMAGE_SIZE = 28


def _get_mnist(directory):
  """Download all MNIST files to directory unless they are there."""
  for filename in [
      _MNIST_TRAIN_DATA_FILENAME, _MNIST_TRAIN_LABELS_FILENAME,
      _MNIST_TEST_DATA_FILENAME, _MNIST_TEST_LABELS_FILENAME
  ]:
    generator_utils.maybe_download(directory, filename, _MNIST_URL + filename)


def _extract_mnist_images(filename, num_images):
  """Extract images from an MNIST file into a numpy array.

  Args:
    filename: The path to an MNIST images file.
    num_images: The number of images in the file.

  Returns:
    A numpy array of shape [number_of_images, height, width, channels].
  """
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1)
  return data


def _extract_mnist_labels(filename, num_labels):
  """Extract labels from an MNIST file into integers.

  Args:
    filename: The path to an MNIST labels file.
    num_labels: The number of labels in the file.

  Returns:
    A int64 numpy array of shape [num_labels]
  """
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels


def mnist_common_generator(tmp_dir,
                           training,
                           how_many,
                           data_filename,
                           label_filename,
                           start_from=0):
  """Image generator for MNIST.

  Args:
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many images and labels to generate.
    data_filename: file that contains features data.
    label_filename: file that contains labels.
    start_from: from which image to start.

  Returns:
    An instance of image_generator that produces MNIST images.
  """
  data_path = os.path.join(tmp_dir, data_filename)
  labels_path = os.path.join(tmp_dir, label_filename)
  images = _extract_mnist_images(data_path, 60000 if training else 10000)
  labels = _extract_mnist_labels(labels_path, 60000 if training else 10000)
  # Shuffle the data to make sure classes are well distributed.
  data = list(zip(images, labels))
  random.shuffle(data)
  images, labels = list(zip(*data))
  return image_utils.image_generator(images[start_from:start_from + how_many],
                                     labels[start_from:start_from + how_many])


def mnist_generator(tmp_dir, training, how_many, start_from=0):
  """Image generator for MNIST.

  Args:
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many images and labels to generate.
    start_from: from which image to start.

  Returns:
    An instance of image_generator that produces MNIST images.
  """
  _get_mnist(tmp_dir)
  d = _MNIST_TRAIN_DATA_FILENAME if training else _MNIST_TEST_DATA_FILENAME
  l = _MNIST_TRAIN_LABELS_FILENAME if training else _MNIST_TEST_LABELS_FILENAME
  return mnist_common_generator(tmp_dir, training, how_many, d, l, start_from)


@registry.register_problem
class ImageMnistTune(image_utils.Image2ClassProblem):
  """MNIST, tuning data."""

  @property
  def num_channels(self):
    return 1

  @property
  def is_small(self):
    return True

  @property
  def num_classes(self):
    return 10

  @property
  def class_labels(self):
    return [str(c) for c in range(self.num_classes)]

  @property
  def train_shards(self):
    return 10

  def preprocess_example(self, example, mode, unused_hparams):
    image = example["inputs"]
    image.set_shape([_MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1])
    if not self._was_reversed:
      image = tf.image.per_image_standardization(image)
    example["inputs"] = image
    return example

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return mnist_generator(tmp_dir, True, 55000)
    else:
      return mnist_generator(tmp_dir, True, 5000, 55000)


@registry.register_problem
class ImageMnist(ImageMnistTune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return mnist_generator(tmp_dir, True, 60000)
    else:
      return mnist_generator(tmp_dir, False, 10000)


# URLs and filenames for MNIST data.
_FASHION_MNIST_URL = ("http://fashion-mnist.s3-website.eu-central-1"
                      ".amazonaws.com/")
_FASHION_MNIST_LOCAL_FILE_PREFIX = "fashion-"
_FASHION_MNIST_IMAGE_SIZE = 28


def _get_fashion_mnist(directory):
  """Download all FashionMNIST files to directory unless they are there."""
  # Fashion mnist files have the same names as MNIST.
  # We must choose a separate name (by adding 'fashion-' prefix) in the tmp_dir.
  for filename in [
      _MNIST_TRAIN_DATA_FILENAME, _MNIST_TRAIN_LABELS_FILENAME,
      _MNIST_TEST_DATA_FILENAME, _MNIST_TEST_LABELS_FILENAME
  ]:
    generator_utils.maybe_download(directory,
                                   _FASHION_MNIST_LOCAL_FILE_PREFIX + filename,
                                   _FASHION_MNIST_URL + filename)


def fashion_mnist_generator(tmp_dir, training, how_many, start_from=0):
  """Image generator for FashionMNIST.

  Args:
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many images and labels to generate.
    start_from: from which image to start.

  Returns:
    An instance of image_generator that produces MNIST images.
  """
  _get_fashion_mnist(tmp_dir)
  d = _FASHION_MNIST_LOCAL_FILE_PREFIX + (
      _MNIST_TRAIN_DATA_FILENAME if training else _MNIST_TEST_DATA_FILENAME)
  l = _FASHION_MNIST_LOCAL_FILE_PREFIX + (
      _MNIST_TRAIN_LABELS_FILENAME if training else _MNIST_TEST_LABELS_FILENAME)
  return mnist_common_generator(tmp_dir, training, how_many, d, l, start_from)


@registry.register_problem
class ImageFashionMnist(image_utils.Image2ClassProblem):
  """Fashion MNIST."""

  @property
  def is_small(self):
    return True

  @property
  def num_classes(self):
    return 10

  @property
  def class_labels(self):
    return [str(c) for c in range(self.num_classes)]

  @property
  def train_shards(self):
    return 10

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return fashion_mnist_generator(tmp_dir, True, 60000)
    else:
      return fashion_mnist_generator(tmp_dir, False, 10000)
