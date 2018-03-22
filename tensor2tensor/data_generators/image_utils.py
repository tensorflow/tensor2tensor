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

"""Base classes and utilities for image datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry

import tensorflow as tf

from tensorflow.python.eager import context


def resize_by_area(img, size):
  """image resize function used by quite a few image problems."""
  return tf.to_int64(
      tf.image.resize_images(img, [size, size], tf.image.ResizeMethod.AREA))


class ImageProblem(problem.Problem):
  """Base class for problems with images."""

  @property
  def num_channels(self):
    """Number of color channels."""
    return 3

  def example_reading_spec(self, label_repr=None):
    data_fields = {
        "image/encoded": tf.FixedLenFeature((), tf.string),
        "image/format": tf.FixedLenFeature((), tf.string),
    }

    data_items_to_decoders = {
        "inputs":
            tf.contrib.slim.tfexample_decoder.Image(
                image_key="image/encoded",
                format_key="image/format",
                channels=self.num_channels),
    }

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    if not self._was_reversed:
      example["inputs"] = tf.image.per_image_standardization(example["inputs"])
    return example

  def eval_metrics(self):
    eval_metrics = [
        metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5,
        metrics.Metrics.ACC_PER_SEQ, metrics.Metrics.NEG_LOG_PERPLEXITY
    ]
    if self._was_reversed:
      eval_metrics += [metrics.Metrics.IMAGE_SUMMARY]
    return eval_metrics


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

  def example_reading_spec(self):
    label_key = "image/class/label"
    data_fields, data_items_to_decoders = (
        super(Image2ClassProblem, self).example_reading_spec())
    data_fields[label_key] = tf.FixedLenFeature((1,), tf.int64)

    data_items_to_decoders[
        "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(label_key)
    return data_fields, data_items_to_decoders

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": (registry.Modalities.IMAGE, 256)}
    p.target_modality = (registry.Modalities.CLASS_LABEL, self.num_classes)
    p.batch_size_multiplier = 4 if self.is_small else 256
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


def _encoded_images(images):
  if context.in_eager_mode():
    for image in images:
      yield tf.image.encode_png(image).numpy()
  else:
    (width, height, channels) = images[0].shape
    with tf.Graph().as_default():
      image_t = tf.placeholder(dtype=tf.uint8, shape=(width, height, channels))
      encoded_image_t = tf.image.encode_png(image_t)
      with tf.Session() as sess:
        for image in images:
          enc_string = sess.run(encoded_image_t, feed_dict={image_t: image})
          yield enc_string


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
  width, height, _ = images[0].shape
  for (enc_image, label) in zip(_encoded_images(images), labels):
    yield {
        "image/encoded": [enc_image],
        "image/format": ["png"],
        "image/class/label": [int(label)],
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

  def example_reading_spec(self):
    label_key = "image/class/label"
    data_fields, data_items_to_decoders = (
        super(Image2TextProblem, self).example_reading_spec())
    data_fields[label_key] = tf.VarLenFeature(tf.int64)
    data_items_to_decoders[
        "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(label_key)
    return data_fields, data_items_to_decoders

  def feature_encoders(self, data_dir):
    if self.is_character_level:
      encoder = text_encoder.ByteTextEncoder()
    else:
      vocab_filename = os.path.join(
          data_dir, "vocab.ende.%d" % self.targeted_vocab_size)
      encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    input_encoder = text_encoder.ImageEncoder()
    return {"inputs": input_encoder, "targets": encoder}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": (registry.Modalities.IMAGE, 256)}
    encoder = self._encoders["targets"]
    p.target_modality = (registry.Modalities.SYMBOL, encoder.vocab_size)
    p.batch_size_multiplier = 256
    p.loss_multiplier = 1.0
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = self.target_space_id

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=False),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))


def image_augmentation(images, do_colors=False, crop_size=None):
  """Image augmentation: cropping, flipping, and color transforms."""
  if crop_size is None:
    crop_size = [299, 299]
  images = tf.random_crop(images, crop_size + [3])
  images = tf.image.random_flip_left_right(images)
  if do_colors:  # More augmentation, but might be slow.
    images = tf.image.random_brightness(images, max_delta=32. / 255.)
    images = tf.image.random_saturation(images, lower=0.5, upper=1.5)
    images = tf.image.random_hue(images, max_delta=0.2)
    images = tf.image.random_contrast(images, lower=0.5, upper=1.5)
  return images


def cifar_image_augmentation(images):
  """Image augmentation suitable for CIFAR-10/100.

  As described in https://arxiv.org/pdf/1608.06993v3.pdf (page 5).

  Args:
    images: a Tensor.
  Returns:
    Tensor of the same shape as images.
  """
  images = tf.image.resize_image_with_crop_or_pad(images, 40, 40)
  images = tf.random_crop(images, [32, 32, 3])
  images = tf.image.random_flip_left_right(images)
  return images
