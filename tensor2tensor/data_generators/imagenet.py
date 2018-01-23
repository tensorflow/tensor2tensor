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

"""ImageNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import registry

import tensorflow as tf


def imagenet_preprocess_example(example, mode, resize_size=None):
  """Preprocessing used for Imagenet and similar problems."""
  if resize_size is None:
    resize_size = [299, 299]

  def preprocess(img):
    img = tf.image.resize_images(img, [360, 360])
    img = image_utils.image_augmentation(
        tf.to_float(img) / 255., crop_size=resize_size)
    return tf.to_int64(img * 255.)

  def resize(img):
    return tf.to_int64(tf.image.resize_images(img, resize_size))

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
class ImageImagenet(image_utils.Image2ClassProblem):
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
    # TODO(rsepassi): Match cloud_tpu preprocessing
    # From cloud_tpu/models/resnet/imagenet_input.py
    return imagenet_preprocess_example(example, mode)


class ImageImagenetRescaled(ImageImagenet):
  """Imagenet rescaled to rescale_size."""

  @property
  def rescale_size(self):
    # return [224, 224]
    raise NotImplementedError()

  def dataset_filename(self):
    return "image_imagenet"  # Reuse Imagenet data.

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    tf.logging.warning(
        "Generate data for rescaled ImageNet problems with image_imagenet")

  def preprocess_example(self, example, mode, _):
    return imagenet_preprocess_example(
        example, mode, resize_size=self.rescale_size)


@registry.register_problem
class ImageImagenet224(ImageImagenetRescaled):
  """Imagenet rescaled to 224x224."""

  @property
  def rescale_size(self):
    return [224, 224]


@registry.register_problem
class ImageImagenet32(ImageImagenetRescaled):
  """Imagenet rescaled to 32x32."""

  @property
  def rescale_size(self):
    return [32, 32]

  @property
  def is_small(self):
    return True  # Modalities like for CIFAR.

  def preprocess_example(self, example, mode, _):
    # Just resize with area.
    if self._was_reversed:
      example["inputs"] = tf.to_int64(
          tf.image.resize_images(example["inputs"], self.rescale_size,
                                 tf.image.ResizeMethod.AREA))
    else:
      example = imagenet_preprocess_example(example, mode)
      example["inputs"] = tf.to_int64(
          tf.image.resize_images(example["inputs"], self.rescale_size))
    return example


@registry.register_problem
class ImageImagenet64(ImageImagenet32):
  """Imagenet rescaled to 64x64."""

  @property
  def rescale_size(self):
    return [64, 64]


@registry.register_problem
class Img2imgImagenet(image_utils.ImageProblem):
  """Imagenet rescaled to 8x8 for input and 32x32 for output."""

  def dataset_filename(self):
    return "image_imagenet"  # Reuse Imagenet data.

  def preprocess_example(self, example, unused_mode, unused_hparams):

    inputs = example["inputs"]
    # For Img2Img resize input and output images as desired.
    example["inputs"] = image_utils.resize_by_area(inputs, 8)
    example["targets"] = image_utils.resize_by_area(inputs, 32)
    return example

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    tf.logging.warning("Generate data for img2img_imagenet with image_imagenet")

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity", 256)}
    p.target_modality = ("image:identity", 256)
    p.batch_size_multiplier = 256
    p.max_expected_batch_size_per_shard = 4
    p.input_space_id = 1
    p.target_space_id = 1
