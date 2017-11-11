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

"""Data generators for image data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import io
import json
import os
import random
import struct
import tarfile
import zipfile

# Dependency imports

import numpy as np
from six.moves import cPickle
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry

import tensorflow as tf


def resize_by_area(img, size):
  """image resize function used by quite a few image problems."""
  return tf.to_int64(
      tf.image.resize_images(img, [size, size], tf.image.ResizeMethod.AREA))


class ImageProblem(problem.Problem):

  def example_reading_spec(self, label_repr=None):
    if label_repr is None:
      label_repr = ("image/class/label", tf.FixedLenFeature((1,), tf.int64))

    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
    }
    label_key, label_type = label_repr  # pylint: disable=unpacking-non-sequence
    data_fields[label_key] = label_type

    data_items_to_decoders = {
        "inputs":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=3),
        "targets":
            tf.contrib.slim.tfexample_decoder.Tensor(label_key),
    }

    return data_fields, data_items_to_decoders


@registry.register_problem("image_celeba_tune")
class ImageCeleba(ImageProblem):
  """CelebA dataset, aligned and cropped images."""
  IMG_DATA = ("img_align_celeba.zip",
              "https://drive.google.com/uc?export=download&"
              "id=0B7EVK8r0v71pZjFTYXZWM3FlRnM")
  LANDMARKS_DATA = ("celeba_landmarks_align",
                    "https://drive.google.com/uc?export=download&"
                    "id=0B7EVK8r0v71pd0FJY3Blby1HUTQ")
  ATTR_DATA = ("celeba_attr", "https://drive.google.com/uc?export=download&"
               "id=0B7EVK8r0v71pblRyaVFSWGxPY0U")

  LANDMARK_HEADINGS = ("lefteye_x lefteye_y righteye_x righteye_y "
                       "nose_x nose_y leftmouth_x leftmouth_y rightmouth_x "
                       "rightmouth_y").split()
  ATTR_HEADINGS = (
      "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs "
      "Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair "
      "Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair "
      "Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache "
      "Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
      "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings "
      "Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
  ).split()

  def preprocess_example(self, example, unused_mode, unused_hparams):

    inputs = example["inputs"]
    # Remove boundaries in CelebA images. Remove 40 pixels each side
    # vertically and 20 pixels each side horizontally.
    inputs = tf.image.crop_to_bounding_box(inputs, 40, 20, 218 - 80, 178 - 40)
    example["inputs"] = resize_by_area(inputs, 8)
    example["targets"] = resize_by_area(inputs, 32)
    return example

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity_no_pad", None)}
    p.target_modality = ("image:identity_no_pad", None)
    p.batch_size_multiplier = 256
    p.max_expected_batch_size_per_shard = 4
    p.input_space_id = 1
    p.target_space_id = 1

  def generator(self, tmp_dir, how_many, start_from=0):
    """Image generator for CELEBA dataset.

    Args:
      tmp_dir: path to temporary storage directory.
      how_many: how many images and labels to generate.
      start_from: from which image to start.

    Yields:
      A dictionary representing the images with the following fields:
      * image/encoded: the string encoding the image as JPEG,
      * image/format: the string "jpeg" representing image format,
    """
    out_paths = []
    for fname, url in [self.IMG_DATA, self.LANDMARKS_DATA, self.ATTR_DATA]:
      path = generator_utils.maybe_download_from_drive(tmp_dir, fname, url)
      out_paths.append(path)

    img_path, landmarks_path, attr_path = out_paths  # pylint: disable=unbalanced-tuple-unpacking
    unzipped_folder = img_path[:-4]
    if not tf.gfile.Exists(unzipped_folder):
      zipfile.ZipFile(img_path, "r").extractall(tmp_dir)

    with tf.gfile.Open(landmarks_path) as f:
      landmarks_raw = f.read()

    with tf.gfile.Open(attr_path) as f:
      attr_raw = f.read()

    def process_landmarks(raw_data):
      landmarks = {}
      lines = raw_data.split("\n")
      headings = lines[1].strip().split()
      for line in lines[2:-1]:
        values = line.strip().split()
        img_name = values[0]
        landmark_values = [int(v) for v in values[1:]]
        landmarks[img_name] = landmark_values
      return landmarks, headings

    def process_attrs(raw_data):
      attrs = {}
      lines = raw_data.split("\n")
      headings = lines[1].strip().split()
      for line in lines[2:-1]:
        values = line.strip().split()
        img_name = values[0]
        attr_values = [int(v) for v in values[1:]]
        attrs[img_name] = attr_values
      return attrs, headings

    img_landmarks, _ = process_landmarks(landmarks_raw)
    img_attrs, _ = process_attrs(attr_raw)

    image_files = tf.gfile.Glob(unzipped_folder + "/*.jpg")
    for filename in image_files[start_from:start_from + how_many]:
      img_name = os.path.basename(filename)
      landmarks = img_landmarks[img_name]
      attrs = img_attrs[img_name]

      with tf.gfile.Open(filename, "r") as f:
        encoded_image_data = f.read()
        yield {
            "image/encoded": [encoded_image_data],
            "image/format": ["jpeg"],
            "attributes": attrs,
            "landmarks": landmarks,
        }

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(tmp_dir, 162770),  # train
        self.training_filepaths(data_dir, self.train_shards, shuffled=False),
        self.generator(tmp_dir, 19867, 162770),  # dev
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))


@registry.register_problem
class ImageFSNS(ImageProblem):
  """Problem spec for French Street Name recognition."""

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    list_url = ("https://raw.githubusercontent.com/tensorflow/models/master/"
                "street/python/fsns_urls.txt")
    fsns_urls = generator_utils.maybe_download(tmp_dir, "fsns_urls.txt",
                                               list_url)
    fsns_files = [
        f.strip() for f in open(fsns_urls, "r") if f.startswith("http://")
    ]
    for url in fsns_files:
      if "/train/train" in url:
        generator_utils.maybe_download(
            data_dir, "image_fsns-train" + url[-len("-00100-of-00512"):], url)
      elif "/validation/validation" in url:
        generator_utils.maybe_download(
            data_dir, "image_fsns-dev" + url[-len("-00100-of-00512"):], url)
      elif "charset" in url:
        generator_utils.maybe_download(data_dir, "charset_size134.txt", url)

  def feature_encoders(self, data_dir):
    # This vocab file must be present within the data directory.
    vocab_filename = os.path.join(data_dir, "charset_size134.txt")
    return {
        "inputs": text_encoder.ImageEncoder(),
        "targets": text_encoder.SubwordTextEncoder(vocab_filename)
    }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": (registry.Modalities.IMAGE, None)}
    vocab_size = self._encoders["targets"].vocab_size
    p.target_modality = (registry.Modalities.SYMBOL, vocab_size)
    p.batch_size_multiplier = 256
    p.max_expected_batch_size_per_shard = 2
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.EN_TOK

  def example_reading_spec(self):
    label_key = "image/unpadded_label"
    label_type = tf.VarLenFeature(tf.int64)
    return super(ImageFSNS, self).example_reading_spec(
        self, label_repr=(label_key, label_type))


class Image2ClassProblem(ImageProblem):
  """Base class for image classification problems."""

  @property
  def is_small(self):
    raise NotImplementedError()

  @property
  def num_classes(self):
    raise NotImplementedError()

  @property
  def train_shards(self):
    raise NotImplementedError()

  @property
  def dev_shards(self):
    return 1

  @property
  def class_labels(self):
    return ["ID_%d" % i for i in range(self.num_classes)]

  def feature_encoders(self, data_dir):
    del data_dir
    return {
        "inputs": text_encoder.ImageEncoder(),
        "targets": text_encoder.ClassLabelEncoder(self.class_labels)
    }

  def generator(self, data_dir, tmp_dir, is_training):
    raise NotImplementedError()

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": (registry.Modalities.IMAGE, None)}
    p.target_modality = (registry.Modalities.CLASS_LABEL,
                         self.num_classes)
    p.batch_size_multiplier = 4 if self.is_small else 256
    p.max_expected_batch_size_per_shard = 8 if self.is_small else 2
    p.loss_multiplier = 3.0 if self.is_small else 1.0
    if self._was_reversed:
      p.loss_multiplier = 1.0
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE_LABEL

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=False),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))


def imagenet_preprocess_example(example, mode):
  """Preprocessing used for Imagenet and similar problems."""

  def preprocess(img):
    img = tf.image.resize_images(img, [360, 360])
    img = common_layers.image_augmentation(tf.to_float(img) / 255.)
    return tf.to_int64(img * 255.)

  def resize(img):
    return tf.to_int64(tf.image.resize_images(img, [299, 299]))

  inputs = tf.cast(example["inputs"], tf.int64)
  if mode == tf.estimator.ModeKeys.TRAIN:
    example["inputs"] = tf.cond(  # Preprocess 90% of the time.
        tf.less(tf.random_uniform([]), 0.9),
        lambda img=inputs: preprocess(img),
        lambda img=inputs: resize(img))
  else:
    example["inputs"] = resize(inputs)
  return example


@registry.register_problem
class ImageImagenet(Image2ClassProblem):
  """Imagenet."""

  @property
  def is_small(self):
    return False

  @property
  def num_classes(self):
    return 1000

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    # TODO(lukaszkaiser): find a better way than printing this.
    print("To generate the ImageNet dataset in the proper format, follow "
          "instructions at https://github.com/tensorflow/models/blob/master"
          "/inception/README.md#getting-started")

  def preprocess_example(self, example, mode, _):
    return imagenet_preprocess_example(example, mode)


@registry.register_problem
class ImageImagenet32(Image2ClassProblem):
  """Imagenet rescaled to 32x32."""

  def dataset_filename(self):
    return "image_imagenet"  # Reuse Imagenet data.

  @property
  def is_small(self):
    return True  # Modalities like for CIFAR.

  @property
  def num_classes(self):
    return 1000

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    # TODO(lukaszkaiser): find a better way than printing this.
    print("To generate the ImageNet dataset in the proper format, follow "
          "instructions at https://github.com/tensorflow/models/blob/master"
          "/inception/README.md#getting-started")

  def preprocess_example(self, example, mode, unused_hparams):
    # Just resize with area.
    if self._was_reversed:
      example["inputs"] = tf.to_int64(
          tf.image.resize_images(example["inputs"], [32, 32],
                                 tf.image.ResizeMethod.AREA))
    else:
      example = imagenet_preprocess_example(example, mode)
      example["inputs"] = tf.to_int64(
          tf.image.resize_images(example["inputs"], [32, 32]))
    return example


@registry.register_problem
class ImageImagenet64(Image2ClassProblem):
  """Imagenet rescaled to 64x64."""

  def dataset_filename(self):
    return "image_imagenet"  # Reuse Imagenet data.

  @property
  def is_small(self):
    return True  # Modalities like for CIFAR.

  @property
  def num_classes(self):
    return 1000

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    # TODO(lukaszkaiser): find a better way than printing this.
    print("To generate the ImageNet dataset in the proper format, follow "
          "instructions at https://github.com/tensorflow/models/blob/master"
          "/inception/README.md#getting-started")

  def preprocess_example(self, example, mode, unused_hparams):
    inputs = example["inputs"]
    # Just resize with area.
    if self._was_reversed:
      example["inputs"] = resize_by_area(inputs, 64)
    else:
      example = imagenet_preprocess_example(example, mode)
      example["inputs"] = example["inputs"] = resize_by_area(inputs, 64)
    return example


@registry.register_problem
class Img2imgImagenet(ImageProblem):
  """Imagenet rescaled to 8x8 for input and 32x32 for output."""

  def dataset_filename(self):
    return "image_imagenet"  # Reuse Imagenet data.

  def preprocess_example(self, example, unused_mode, unused_hparams):

    inputs = example["inputs"]
    # For Img2Img resize input and output images as desired.
    example["inputs"] = resize_by_area(inputs, 8)
    example["targets"] = resize_by_area(inputs, 32)
    return example

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity_no_pad", None)}
    p.target_modality = ("image:identity_no_pad", None)
    p.batch_size_multiplier = 256
    p.max_expected_batch_size_per_shard = 4
    p.input_space_id = 1
    p.target_space_id = 1


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
            "image/class/label": [int(label)],
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


@registry.register_problem
class ImageMnistTune(Image2ClassProblem):
  """MNIST, tuning data."""

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
  return image_generator(all_images[start_from:start_from + how_many],
                         all_labels[start_from:start_from + how_many])


@registry.register_problem
class ImageCifar10Tune(ImageMnistTune):
  """Cifar-10 Tune."""

  @property
  def class_labels(self):
    return [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
    ]

  def preprocess_example(self, example, mode, unused_hparams):
    example["inputs"].set_shape([_CIFAR10_IMAGE_SIZE, _CIFAR10_IMAGE_SIZE, 3])
    if mode == tf.estimator.ModeKeys.TRAIN:
      example["inputs"] = common_layers.cifar_image_augmentation(
          example["inputs"])
    example["inputs"] = tf.to_int64(example["inputs"])
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
    example["inputs"].set_shape([_CIFAR10_IMAGE_SIZE, _CIFAR10_IMAGE_SIZE, 3])
    example["inputs"] = tf.to_int64(example["inputs"])
    return example


@registry.register_problem
class ImageCifar10Plain8(ImageCifar10):
  """CIFAR-10 rescaled to 8x8 for output: Conditional image generation."""

  def dataset_filename(self):
    return "image_cifar10_plain"  # Reuse CIFAR-10 plain data.

  def preprocess_example(self, example, mode, unused_hparams):
    example["inputs"] = resize_by_area(example["inputs"], 8)
    return example


@registry.register_problem
class Img2imgCifar10(ImageCifar10):
  """CIFAR-10 rescaled to 8x8 for input and 32x32 for output."""

  def dataset_filename(self):
    return "image_cifar10_plain"  # Reuse CIFAR-10 plain data.

  def preprocess_example(self, example, unused_mode, unused_hparams):

    inputs = example["inputs"]
    # For Img2Img resize input and output images as desired.
    example["inputs"] = resize_by_area(inputs, 8)
    example["targets"] = resize_by_area(inputs, 32)
    return example

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity_no_pad", None)}
    p.target_modality = ("image:identity_no_pad", None)
    p.batch_size_multiplier = 256
    p.max_expected_batch_size_per_shard = 4
    p.input_space_id = 1
    p.target_space_id = 1


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


def mscoco_generator(data_dir,
                     tmp_dir,
                     training,
                     how_many,
                     start_from=0,
                     eos_list=None,
                     vocab_filename=None,
                     vocab_size=0):
  """Image generator for MSCOCO captioning problem with token-wise captions.

  Args:
    data_dir: path to the data directory.
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
        data_dir, tmp_dir, vocab_filename, vocab_size)
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


class Image2TextProblem(ImageProblem):
  """Base class for image-to-text problems."""

  @property
  def is_character_level(self):
    raise NotImplementedError()

  @property
  def targeted_vocab_size(self):
    raise NotImplementedError()  # Not needed if self.is_character_level.

  @property
  def target_space_id(self):
    raise NotImplementedError()

  @property
  def train_shards(self):
    raise NotImplementedError()

  @property
  def dev_shards(self):
    raise NotImplementedError()

  def generator(self, data_dir, tmp_dir, is_training):
    raise NotImplementedError()

  def feature_encoders(self, data_dir):
    if self.is_character_level:
      encoder = text_encoder.ByteTextEncoder()
    else:
      vocab_filename = os.path.join(
          data_dir, "vocab.endefr.%d" % self.targeted_vocab_size)
      encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    return {"targets": encoder}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": (registry.Modalities.IMAGE, None)}
    encoder = self._encoders["targets"]
    p.target_modality = (registry.Modalities.SYMBOL, encoder.vocab_size)
    p.batch_size_multiplier = 256
    p.max_expected_batch_size_per_shard = 2
    p.loss_multiplier = 1.0
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = self.target_space_id

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=False),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))


@registry.register_problem
class ImageMsCocoCharacters(Image2TextProblem):
  """MSCOCO, character level."""

  @property
  def is_character_level(self):
    return True

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  def preprocess_example(self, example, mode, _):
    return imagenet_preprocess_example(example, mode)

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return mscoco_generator(data_dir, tmp_dir, True, 80000)
    else:
      return mscoco_generator(data_dir, tmp_dir, False, 40000)
    raise NotImplementedError()


@registry.register_problem
class ImageMsCocoTokens8k(ImageMsCocoCharacters):
  """MSCOCO, 8k tokens vocab."""

  @property
  def is_character_level(self):
    return False

  @property
  def targeted_vocab_size(self):
    return 2**13  # 8192

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def train_shards(self):
    return 100

  @property
  def dev_shards(self):
    return 10

  def generator(self, data_dir, tmp_dir, is_training):
    vocab_filename = "vocab.endefr.%d" % self.targeted_vocab_size
    if is_training:
      return mscoco_generator(
          data_dir,
          tmp_dir,
          True,
          80000,
          vocab_filename=vocab_filename,
          vocab_size=self.targeted_vocab_size)
    else:
      return mscoco_generator(
          data_dir,
          tmp_dir,
          False,
          40000,
          vocab_filename=vocab_filename,
          vocab_size=self.targeted_vocab_size)


@registry.register_problem
class ImageMsCocoTokens32k(ImageMsCocoTokens8k):
  """MSCOCO, 32k tokens vocab."""

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768


@registry.register_problem
class OcrTest(Image2TextProblem):
  """OCR test problem."""

  @property
  def is_small(self):
    return True

  @property
  def is_character_level(self):
    return True

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def train_shards(self):
    return 1

  @property
  def dev_shards(self):
    return 1

  def preprocess_example(self, example, mode, _):
    # Resize from usual size ~1350x60 to 90x4 in this test.
    img = example["inputs"]
    example["inputs"] = tf.to_int64(
        tf.image.resize_images(img, [90, 4], tf.image.ResizeMethod.AREA))
    return example

  def generator(self, data_dir, tmp_dir, is_training):
    # In this test problem, we assume that the data is in tmp_dir/ocr/ in
    # files names 0.png, 0.txt, 1.png, 1.txt and so on until num_examples.
    num_examples = 2
    ocr_dir = os.path.join(tmp_dir, "ocr/")
    tf.logging.info("Looking for OCR data in %s." % ocr_dir)
    for i in xrange(num_examples):
      image_filepath = os.path.join(ocr_dir, "%d.png" % i)
      text_filepath = os.path.join(ocr_dir, "%d.txt" % i)
      with tf.gfile.Open(text_filepath, "rb") as f:
        label = f.read()
      with tf.gfile.Open(image_filepath, "rb") as f:
        encoded_image_data = f.read()
      # In PNG files width and height are stored in these bytes.
      width, height = struct.unpack(">ii", encoded_image_data[16:24])
      yield {
          "image/encoded": [encoded_image_data],
          "image/format": ["png"],
          "image/class/label": label.strip(),
          "image/height": [height],
          "image/width": [width]
      }
