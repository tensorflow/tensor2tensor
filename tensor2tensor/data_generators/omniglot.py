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

"""Omniglot."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import zipfile

# Dependency imports

import numpy as np
import matplotlib.image as im
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

from tensorflow.python.eager import context

# URLs and filenames for Omniglot data.
_OMNIGLOT_URL = "https://github.com/brendenlake/omniglot/archive/"
_OMNIGLOT_FILENAME = "master.zip"
_OMNIGLOT_TRAIN_FILENAME = "omniglot-master/python/images_background.zip"
_OMNIGLOT_TEST_FILENAME = "omniglot-master/python/images_evaluation.zip"
_OMNIGLOT_TRAIN_DIRECTORY = "images_background"
_OMNIGLOT_TEST_DIRECTORY = "images_evaluation"


def _get_omniglot(directory):
  """Download all Omniglot files to directory unless they are there."""
  uri = os.path.join(_OMNIGLOT_URL, _OMNIGLOT_FILENAME)
  generator_utils.maybe_download(directory, _OMNIGLOT_FILENAME, uri)

  def extract_all(directory, filename):
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(directory)
    zip_ref.close()

  extract_all(directory, os.path.join(directory, _OMNIGLOT_FILENAME))
  extract_all(directory, os.path.join(directory, _OMNIGLOT_TRAIN_FILENAME))
  extract_all(directory, os.path.join(directory, _OMNIGLOT_TEST_FILENAME))


def _extract_omniglot_images(directory):
  """Extract images from an Omniglot directory into a numpy array.

  Args:
    directory: The path to a directory with Omniglot images files.

  Returns:
    A numpy array of shape [number_of_images, height, width].
  """
  data = []
  filenames = tf.gfile.Glob(os.path.join(directory,'*'))
  for filename in filenames:
      image = im.imread(filename)
      if len(image.shape) == 2:
        image = np.expand_dims(image, -1)
      data.append(image)
  return np.stack(data)


def _resize_images(images, size):
  """Resize images to size.

  Args:
    images: 4-D Tensor of shape [batch, height, width, channels].
    size: A 1-D int32 Tensor of 2 elements: new_height, new_width.

  Returns:
    A 4-D float Tensor of shape [batch, new_height, new_width, channels].
  """
  if context.in_eager_mode():
    return tf.image.resize_images(images, size).numpy()
  else:
    with tf.Graph().as_default():
      resized_images = tf.image.resize_images(images, size)
      with tf.Session() as sess:
        return sess.run(resized_images)


class OmniglotProblem(problem.Problem):
  """Omniglot, N-way k-shot classification.

  The problem of N-way classification is set up as follows:
  select N unseen classes, provide the model with K different
  instances of each of the N classes, and evaluate the model’s
  ability to classify new instances within the N classes.

  Description from: https://arxiv.org/pdf/1703.03400.pdf
  """

  @property
  def num_ways(self):
    """Number of classes per example."""
    raise NotImplementedError()

  @property
  def num_shots(self):
    """Number of instances per class."""
    raise NotImplementedError()

  @property
  def is_small(self):
    """Determines how many train/dev examples."""
    return True

  @property
  def num_train(self):
    """The number of train examples."""
    return 10000 if self.is_small else 1000000

  @property
  def num_dev(self):
    """The number of dev examples."""
    return 1000 if self.is_small else 100000

  @property
  def image_size(self):
    """The height and width of the images."""
    return 28

  @property
  def num_shards(self):
    return 100

  def generator(self, tmp_dir, is_training):
    """Generates Omniglot learning tasks.

    Following the experimental protocol proposed in
    https://arxiv.org/abs/1606.04080

    Args:
      tmp_dir: path to temporary storage directory.
      is_training: a Boolean; if true, we use the train set, otherwise the test
          set.

    Yields:
      A dictionary representing the images with the following fields:
      * inputs: the uint8 string encoding of (num_ways * num_shots + 1) images,
      * targets: the uint8 string encoding of (num_ways * num_shots + 1) labels,
      Every field is actually a singleton list of the corresponding type.
      The last image/label is the evaluation example.
    """
    d = _OMNIGLOT_TRAIN_DIRECTORY if is_training else _OMNIGLOT_TEST_DIRECTORY
    n = self.num_train if is_training else self.num_dev

    _get_omniglot(tmp_dir)

    pattern = os.path.join(tmp_dir, d, '**', '*')
    directories = tf.gfile.Glob(pattern)

    def _extract_and_resize(directory):
      images = _extract_omniglot_images(directory)
      return _resize_images(images, (self.image_size, self.image_size))

    # load the entire dataset into memory (it's not that big)
    print("Loading Omniglot %s data set into memory"
          % ('train' if is_training else 'dev'))
    data = [_extract_and_resize(d) for d in directories]

    for _ in xrange(n):
      ways = random.sample(data, self.num_ways)
      random_label = np.random.randint(self.num_ways)

      inputs = []
      targets = []

      for label, way in enumerate(ways):
        num_to_sample = (self.num_shots + 1 if label == random_label
                         else self.num_shots)
        # sample from data
        take_indices = np.random.choice(
            way.shape[0], num_to_sample, replace=False)
        inputs.append(np.take(way, take_indices, axis=0))
        targets.append(np.ones(num_to_sample) * label)

      # make the evaluation example the last in the sequence
      # this convention is parsed in `preprocess_example`
      inputs = np.concatenate(sorted(inputs, key=len))
      targets = np.concatenate(sorted(targets, key=len))

      yield {
        "inputs": [inputs.astype(np.uint8).tobytes()],
        "targets": [targets.astype(np.uint8).tobytes()],
      }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(tmp_dir, is_training=True),
        self.training_filepaths(data_dir, self.num_shards, shuffled=False),
        self.generator(tmp_dir, is_training=False),
        self.dev_filepaths(data_dir, 1, shuffled=False))

  def hparams(self, defaults, model_hparams):
    defaults.input_modality = {}
    defaults.target_modality = (registry.Modalities.SYMBOL, self.num_ways)
    defaults.input_space_id = problem.SpaceID.GENERIC
    defaults.target_space_id = problem.SpaceID.GENERIC

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.FixedLenFeature([], tf.string),
        "targets": tf.FixedLenFeature([], tf.string),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

  def preprocess_example(self, example, mode, unused_hparams):
    del mode

    num_images = self.num_ways * self.num_shots + 1
    inputs = tf.decode_raw(example["inputs"], tf.uint8)
    targets = tf.decode_raw(example["targets"], tf.uint8)

    inputs = tf.reshape(
      inputs, [num_images, self.image_size, self.image_size, 1])
    targets = tf.reshape(targets, [num_images])
    inputs = tf.to_float(inputs)
    targets = tf.to_int32(targets)

    example["targets"] = targets
    example["inputs"] = inputs

    # split by background and evaluation
    example["inputs/background"] = inputs[:-1]
    example["inputs/evaluation"] = inputs[-1:]
    example["targets/background/labels"] = targets[:-1]
    example["targets/evaluation/labels"] = targets[-1:]

    # one-hot encoded targets
    one_hot_targets = tf.one_hot(targets, self.num_ways, dtype=tf.int32)
    example["targets/background/one_hot"] = one_hot_targets[:-1]
    example["targets/evaluation/one_hot"] = one_hot_targets[-1:]
    return example


@registry.register_problem("omniglot_5w1s")
class Omniglot5Way1ShotProblem(OmniglotProblem):
  """Omniglot, 5-way 1-shot classification."""

  @property
  def num_ways(self):
    return 5

  @property
  def num_shots(self):
    return 1


@registry.register_problem("omniglot_5w5s")
class Omniglot5Way5ShotProblem(OmniglotProblem):
  """Omniglot, 5-way 5-shot classification."""

  @property
  def num_ways(self):
    return 5

  @property
  def num_shots(self):
    return 5


@registry.register_problem("omniglot_20w1s")
class Omniglot20Way1ShotProblem(OmniglotProblem):
  """Omniglot, 20-way 1-shot classification."""

  @property
  def num_ways(self):
    return 20

  @property
  def num_shots(self):
    return 1


@registry.register_problem("omniglot_20w5s")
class Omniglot20Way5ShotProblem(OmniglotProblem):
  """Omniglot, 20-way 5-shot classification."""

  @property
  def num_ways(self):
    return 20

  @property
  def num_shots(self):
    return 5
