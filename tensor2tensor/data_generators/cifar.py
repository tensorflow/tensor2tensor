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

"""CIFAR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

import numpy as np

from six.moves import cPickle

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import mnist
from tensor2tensor.utils import registry

import tensorflow as tf

# URLs and filenames for CIFAR data.
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_PREFIX = "cifar-10-batches-py/"
_CIFAR10_TRAIN_FILES = [
    "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4",
    "data_batch_5"
]
_CIFAR10_TEST_FILES = ["test_batch"]
_CIFAR10_IMAGE_SIZE = 32


def _get_cifar10(directory):
  """Download and extract CIFAR to directory unless it is there."""
  filename = os.path.basename(_CIFAR10_URL)
  path = generator_utils.maybe_download(directory, filename, _CIFAR10_URL)
  tarfile.open(path, "r:gz").extractall(directory)


def cifar10_generator(tmp_dir, training, how_many, start_from=0):
  """Image generator for CIFAR-10.

  Args:
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many images and labels to generate.
    start_from: from which image to start.

  Returns:
    An instance of image_generator that produces CIFAR-10 images and labels.
  """
  _get_cifar10(tmp_dir)
  data_files = _CIFAR10_TRAIN_FILES if training else _CIFAR10_TEST_FILES
  all_images, all_labels = [], []
  for filename in data_files:
    path = os.path.join(tmp_dir, _CIFAR10_PREFIX, filename)
    with tf.gfile.Open(path, "r") as f:
      data = cPickle.load(f)
    images = data["data"]
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, _CIFAR10_IMAGE_SIZE,
                             _CIFAR10_IMAGE_SIZE))
    all_images.extend([
        np.squeeze(images[j]).transpose((1, 2, 0)) for j in xrange(num_images)
    ])
    labels = data["labels"]
    all_labels.extend([labels[j] for j in xrange(num_images)])
  return image_utils.image_generator(
      all_images[start_from:start_from + how_many],
      all_labels[start_from:start_from + how_many])


@registry.register_problem
class ImageCifar10Tune(mnist.ImageMnistTune):
  """Cifar-10 Tune."""

  @property
  def num_channels(self):
    return 3

  @property
  def class_labels(self):
    return [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
    ]

  def preprocess_example(self, example, mode, unused_hparams):
    image = example["inputs"]
    image.set_shape([_CIFAR10_IMAGE_SIZE, _CIFAR10_IMAGE_SIZE, 3])
    if mode == tf.estimator.ModeKeys.TRAIN:
      image = image_utils.cifar_image_augmentation(image)
    image = tf.image.per_image_standardization(image)
    example["inputs"] = image
    return example

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return cifar10_generator(tmp_dir, True, 48000)
    else:
      return cifar10_generator(tmp_dir, True, 2000, 48000)


@registry.register_problem
class ImageCifar10(ImageCifar10Tune):

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return cifar10_generator(tmp_dir, True, 50000)
    else:
      return cifar10_generator(tmp_dir, False, 10000)


@registry.register_problem
class ImageCifar10Plain(ImageCifar10):

  def preprocess_example(self, example, mode, unused_hparams):
    image = example["inputs"]
    image.set_shape([_CIFAR10_IMAGE_SIZE, _CIFAR10_IMAGE_SIZE, 3])
    image = tf.image.per_image_standardization(image)
    example["inputs"] = image
    return example


@registry.register_problem
class ImageCifar10PlainGen(ImageCifar10Plain):
  """CIFAR-10 32x32 for image generation without standardization preprep."""

  def dataset_filename(self):
    return "image_cifar10_plain"  # Reuse CIFAR-10 plain data.

  def preprocess_example(self, example, mode, unused_hparams):
    example["inputs"].set_shape([_CIFAR10_IMAGE_SIZE, _CIFAR10_IMAGE_SIZE, 3])
    example["inputs"] = tf.to_int64(example["inputs"])
    return example


@registry.register_problem
class ImageCifar10Plain8(ImageCifar10):
  """CIFAR-10 rescaled to 8x8 for output: Conditional image generation."""

  def dataset_filename(self):
    return "image_cifar10_plain"  # Reuse CIFAR-10 plain data.

  def preprocess_example(self, example, mode, unused_hparams):
    image = example["inputs"]
    image = image_utils.resize_by_area(image, 8)
    image = tf.image.per_image_standardization(image)
    example["inputs"] = image
    return example


@registry.register_problem
class Img2imgCifar10(ImageCifar10):
  """CIFAR-10 rescaled to 8x8 for input and 32x32 for output."""

  def dataset_filename(self):
    return "image_cifar10_plain"  # Reuse CIFAR-10 plain data.

  def preprocess_example(self, example, unused_mode, unused_hparams):

    inputs = example["inputs"]
    # For Img2Img resize input and output images as desired.
    example["inputs"] = image_utils.resize_by_area(inputs, 8)
    example["targets"] = image_utils.resize_by_area(inputs, 32)
    return example

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity", 256)}
    p.target_modality = ("image:identity", 256)
    p.batch_size_multiplier = 256
    p.max_expected_batch_size_per_shard = 4
    p.input_space_id = 1
    p.target_space_id = 1
