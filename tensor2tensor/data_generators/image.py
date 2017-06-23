# Copyright 2017 Google Inc.
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

"""Data generators for image data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import gzip
import io
import json
import os
import random
import tarfile
import zipfile

# Dependency imports

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import generator_utils

import tensorflow as tf


def image_generator(images, labels):
  """Generator for images that takes image and labels lists and creates pngs.

  Args:
    images: list of images given as [width x height x channels] numpy arrays.
    labels: list of ints, same length as images.

  Yields:
    A dictionary representing the images with the following fields:
    * image/encoded: the string encoding the image as PNG,
    * image/format: the string "png" representing image format,
    * image/class/label: an integer representing the label,
    * image/height: an integer representing the height,
    * image/width: an integer representing the width.
    Every field is actually a singleton list of the corresponding type.

  Raises:
    ValueError: if images is an empty list.
  """
  if not images:
    raise ValueError("Must provide some images for the generator.")
  (width, height, channels) = images[0].shape
  with tf.Graph().as_default():
    image_t = tf.placeholder(dtype=tf.uint8, shape=(width, height, channels))
    encoded_image_t = tf.image.encode_png(image_t)
    with tf.Session() as sess:
      for (image, label) in zip(images, labels):
        enc_string = sess.run(encoded_image_t, feed_dict={image_t: image})
        yield {
            "image/encoded": [enc_string],
            "image/format": ["png"],
            "image/class/label": [label],
            "image/height": [height],
            "image/width": [width]
        }


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
  data_path = os.path.join(tmp_dir, d)
  labels_path = os.path.join(tmp_dir, l)
  images = _extract_mnist_images(data_path, 60000 if training else 10000)
  labels = _extract_mnist_labels(labels_path, 60000 if training else 10000)
  # Shuffle the data to make sure classes are well distributed.
  data = list(zip(images, labels))
  random.shuffle(data)
  images, labels = list(zip(*data))
  return image_generator(images[start_from:start_from + how_many],
                         labels[start_from:start_from + how_many])


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
  # Shuffle the data to make sure classes are well distributed.
  data = zip(all_images, all_labels)
  random.shuffle(data)
  all_images, all_labels = zip(*data)
  return image_generator(all_images[start_from:start_from + how_many],
                         all_labels[start_from:start_from + how_many])


# URLs and filenames for MSCOCO data.
_MSCOCO_ROOT_URL = "http://msvocds.blob.core.windows.net/"
_MSCOCO_URLS = [
    "coco2014/train2014.zip", "coco2014/val2014.zip", "coco2014/test2014.zip",
    "annotations-1-0-3/captions_train-val2014.zip"
]
_MSCOCO_TRAIN_PREFIX = "train2014"
_MSCOCO_EVAL_PREFIX = "val2014"
_MSCOCO_TRAIN_CAPTION_FILE = "annotations/captions_train2014.json"
_MSCOCO_EVAL_CAPTION_FILE = "annotations/captions_val2014.json"


def _get_mscoco(directory):
  """Download and extract MSCOCO datasets to directory unless it is there."""
  for url in _MSCOCO_URLS:
    filename = os.path.basename(url)
    download_url = os.path.join(_MSCOCO_ROOT_URL, url)
    path = generator_utils.maybe_download(directory, filename, download_url)
    unzip_dir = os.path.join(directory, filename.strip(".zip"))
    if not tf.gfile.Exists(unzip_dir):
      zipfile.ZipFile(path, "r").extractall(directory)


def mscoco_generator(tmp_dir,
                     training,
                     how_many,
                     start_from=0,
                     eos_list=None,
                     vocab_filename=None,
                     vocab_size=0):
  """Image generator for MSCOCO captioning problem with token-wise captions.

  Args:
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    how_many: how many images and labels to generate.
    start_from: from which image to start.
    eos_list: optional list of end of sentence tokens, otherwise use default
      value `1`.
    vocab_filename: file within `tmp_dir` to read vocabulary from.
    vocab_size: integer target to generate vocabulary size to.

  Yields:
    A dictionary representing the images with the following fields:
    * image/encoded: the string encoding the image as JPEG,
    * image/format: the string "jpeg" representing image format,
    * image/class/label: a list of integers representing the caption,
    * image/height: an integer representing the height,
    * image/width: an integer representing the width.
    Every field is actually a list of the corresponding type.
  """
  eos_list = [1] if eos_list is None else eos_list
  if vocab_filename is not None:
    vocab_symbolizer = generator_utils.get_or_generate_vocab(
        tmp_dir, vocab_filename, vocab_size)
  _get_mscoco(tmp_dir)
  caption_filepath = (_MSCOCO_TRAIN_CAPTION_FILE
                      if training else _MSCOCO_EVAL_CAPTION_FILE)
  caption_filepath = os.path.join(tmp_dir, caption_filepath)
  prefix = _MSCOCO_TRAIN_PREFIX if training else _MSCOCO_EVAL_PREFIX
  caption_file = io.open(caption_filepath)
  caption_json = json.load(caption_file)
  # Dictionary from image_id to ((filename, height, width), captions).
  image_dict = dict()
  for image in caption_json["images"]:
    image_dict[image["id"]] = [(image["file_name"], image["height"],
                                image["width"]), []]
  annotations = caption_json["annotations"]
  annotation_count = len(annotations)
  image_count = len(image_dict)
  tf.logging.info("Processing %d images and %d labels\n" % (image_count,
                                                            annotation_count))
  for annotation in annotations:
    image_id = annotation["image_id"]
    image_dict[image_id][1].append(annotation["caption"])

  data = list(image_dict.values())[start_from:start_from + how_many]
  random.shuffle(data)
  for image_info, labels in data:
    image_filename = image_info[0]
    image_filepath = os.path.join(tmp_dir, prefix, image_filename)
    with tf.gfile.Open(image_filepath, "r") as f:
      encoded_image_data = f.read()
      height, width = image_info[1], image_info[2]
      for label in labels:
        if vocab_filename is None:
          label = [ord(c) for c in label] + eos_list
        else:
          label = vocab_symbolizer.encode(label) + eos_list
        yield {
            "image/encoded": [encoded_image_data],
            "image/format": ["jpeg"],
            "image/class/label": label,
            "image/height": [height],
            "image/width": [width]
        }
