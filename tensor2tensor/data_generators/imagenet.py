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

"""ImageNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

# URLs and filenames for IMAGENET 32x32 data from
# https://arxiv.org/abs/1601.06759.
_IMAGENET_SMALL_ROOT_URL = "http://image-net.org/small/"
_IMAGENET_SMALL_URLS = [
    "train_32x32.tar", "valid_32x32.tar"]
_IMAGENET_SMALL_TRAIN_PREFIX = "train_32x32"
_IMAGENET_SMALL_EVAL_PREFIX = "valid_32x32"
_IMAGENET_SMALL_IMAGE_SIZE = 32


# URLs and filenames for IMAGENET 64x64 data.
_IMAGENET_MEDIUM_ROOT_URL = "http://image-net.org/small/"
_IMAGENET_MEDIUM_URLS = [
    "train_64x64.tar", "valid_64x64.tar"]
_IMAGENET_MEDIUM_TRAIN_PREFIX = "train_64x64"
_IMAGENET_MEDIUM_EVAL_PREFIX = "valid_64x64"
_IMAGENET_MEDIUM_IMAGE_SIZE = 64


# Derived from ImageNet data
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def imagenet_pixelrnn_generator(tmp_dir,
                                training,
                                size=_IMAGENET_SMALL_IMAGE_SIZE):
  """Image generator for Imagenet 64x64 downsampled images.

  It assumes that the data has been downloaded from
  http://image-net.org/small/*_32x32.tar or
  http://image-net.org/small/*_64x64.tar into tmp_dir.
  Args:
    tmp_dir: path to temporary storage directory.
    training: a Boolean; if true, we use the train set, otherwise the test set.
    size: image size (assumes height and width are same)

  Yields:
    A dictionary representing the images with the following fields:
    * image/encoded: the string encoding the image as JPEG,
    * image/format: the string "jpeg" representing image format,
    * image/height: an integer representing the height,
    * image/width: an integer representing the width.
    Every field is actually a list of the corresponding type.
  """
  if size == _IMAGENET_SMALL_IMAGE_SIZE:
    train_prefix = _IMAGENET_SMALL_TRAIN_PREFIX
    eval_prefix = _IMAGENET_SMALL_EVAL_PREFIX
  else:
    train_prefix = _IMAGENET_MEDIUM_TRAIN_PREFIX
    eval_prefix = _IMAGENET_MEDIUM_EVAL_PREFIX
  prefix = train_prefix if training else eval_prefix
  images_filepath = os.path.join(tmp_dir, prefix)
  image_files = tf.gfile.Glob(images_filepath + "/*")
  height = size
  width = size
  const_label = 0
  for filename in image_files:
    with tf.gfile.Open(filename, "r") as f:
      encoded_image = f.read()
      yield {
          "image/encoded": [encoded_image],
          "image/format": ["png"],
          "image/class/label": [const_label],
          "image/height": [height],
          "image/width": [width]
      }


def imagenet_preprocess_example(example, mode, resize_size=None,
                                normalize=True):
  """Preprocessing used for Imagenet and similar problems."""
  resize_size = resize_size or [299, 299]
  assert resize_size[0] == resize_size[1]

  image = example["inputs"]
  if mode == tf.estimator.ModeKeys.TRAIN:
    image = preprocess_for_train(image, image_size=resize_size[0],
                                 normalize=normalize)
  else:
    image = preprocess_for_eval(image, image_size=resize_size[0],
                                normalize=normalize)

  example["inputs"] = image
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
          "instructions at https://github.com/tensorflow/models/tree/master"
          "/research/inception/README.md#getting-started")

  def preprocess_example(self, example, mode, _):
    return imagenet_preprocess_example(example, mode)


class ImageImagenetRescaled(ImageImagenet):
  """Imagenet rescaled to rescale_size."""

  @property
  def rescale_size(self):
    # return [224, 224]
    raise NotImplementedError()

  @property
  def normalize_image(self):
    """Whether the image should be normalized in preprocessing."""
    return True

  def dataset_filename(self):
    return "image_imagenet"  # Reuse Imagenet data.

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    tf.logging.warning(
        "Generate data for rescaled ImageNet problems with image_imagenet")

  def preprocess_example(self, example, mode, _):
    return imagenet_preprocess_example(
        example, mode, resize_size=self.rescale_size,
        normalize=self.normalize_image)


@registry.register_problem
class ImageImagenet224(ImageImagenetRescaled):
  """Imagenet rescaled to 224x224."""

  @property
  def rescale_size(self):
    return [224, 224]


@registry.register_problem
class ImageImagenet224NoNormalization(ImageImagenet224):
  """Imagenet rescaled to 224x224 without normalization."""

  @property
  def normalize_image(self):
    """Whether the image should be normalized in preprocessing."""
    return False


@registry.register_problem
class ImageImagenet256(ImageImagenetRescaled):
  """Imagenet rescaled to 256x256."""

  @property
  def rescale_size(self):
    return [256, 256]


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
class ImageImagenet32Gen(ImageImagenet):
  """Imagenet 32 from the pixen cnn paper."""

  @property
  def train_shards(self):
    return 1024

  @property
  def dev_shards(self):
    return 10

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=True),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=True))

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return imagenet_pixelrnn_generator(
          tmp_dir, int(True), size=_IMAGENET_SMALL_IMAGE_SIZE)
    else:
      return imagenet_pixelrnn_generator(
          tmp_dir, int(is_training), size=_IMAGENET_SMALL_IMAGE_SIZE)

  def preprocess_example(self, example, mode, unused_hparams):
    example["inputs"].set_shape([_IMAGENET_SMALL_IMAGE_SIZE,
                                 _IMAGENET_SMALL_IMAGE_SIZE, 3])
    example["inputs"] = tf.to_int64(example["inputs"])
    return example


@registry.register_problem
class ImageImagenet64Gen(ImageImagenet):
  """Imagenet 64 from the pixen cnn paper."""

  @property
  def train_shards(self):
    return 1024

  @property
  def dev_shards(self):
    return 10

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, True),
        self.training_filepaths(data_dir, self.train_shards, shuffled=True),
        self.generator(data_dir, tmp_dir, False),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=True))

  def generator(self, data_dir, tmp_dir, is_training):
    if is_training:
      return imagenet_pixelrnn_generator(
          tmp_dir, int(True), size=_IMAGENET_MEDIUM_IMAGE_SIZE)
    else:
      return imagenet_pixelrnn_generator(
          tmp_dir, int(False), size=_IMAGENET_MEDIUM_IMAGE_SIZE)

  def preprocess_example(self, example, mode, unused_hparams):
    example["inputs"].set_shape([_IMAGENET_MEDIUM_IMAGE_SIZE,
                                 _IMAGENET_MEDIUM_IMAGE_SIZE, 3])
    example["inputs"] = tf.to_int64(example["inputs"])
    return example


@registry.register_problem
class ImageImagenetMultiResolutionGen(ImageImagenet64Gen):
  """ImageNet at multiple resolutions.

  The resolutions are specified as a hyperparameter during preprocessing.
  """

  def dataset_filename(self):
    return "image_imagenet64_gen"

  @property
  def train_shards(self):
    return 1024

  @property
  def dev_shards(self):
    return 10

  def preprocess_example(self, example, mode, hparams):
    image = example["inputs"]
    # Get resize method. Include a default if not specified, or if it's not in
    # TensorFlow's collection of pre-implemented resize methods.
    resize_method = getattr(hparams, "resize_method", "BICUBIC")
    resize_method = getattr(tf.image.ResizeMethod, resize_method, resize_method)

    if resize_method == "DILATED":
      scaled_images = image_utils.make_multiscale_dilated(
          image, hparams.resolutions, num_channels=self.num_channels)
    else:
      scaled_images = image_utils.make_multiscale(
          image, hparams.resolutions,
          resize_method=resize_method, num_channels=self.num_channels)

    # Pack tuple of scaled images into one tensor. We do this by enforcing the
    # columns to match for every resolution.
    # TODO(avaswani, trandustin): We should create tuples because this will not
    # work if height*width of low res < width of high res
    highest_res = hparams.resolutions[-1]
    example["inputs"] = tf.concat([
        tf.reshape(scaled_image,
                   [res**2 // highest_res, highest_res, self.num_channels])
        for scaled_image, res in zip(scaled_images, hparams.resolutions)],
                                  axis=0)
    return example


@registry.register_problem
class ImageImagenet64GenFlat(ImageImagenet64Gen):
  """Imagenet 64 from the pixen cnn paper, as a flat array."""

  def dataset_filename(self):
    return "image_imagenet64_gen"  # Reuse data.

  def preprocess_example(self, example, mode, unused_hparams):
    example["inputs"].set_shape(
        [_IMAGENET_MEDIUM_IMAGE_SIZE, _IMAGENET_MEDIUM_IMAGE_SIZE, 3])
    example["inputs"] = tf.to_int64(example["inputs"])
    example["inputs"] = tf.reshape(example["inputs"], (-1,))

    del example["targets"]  # Ensure unconditional generation

    return example

  def hparams(self, defaults, model_hparams):
    super(ImageImagenet64GenFlat, self).hparams(defaults, model_hparams)
    # Switch to symbol modality
    p = defaults
    p.modality["inputs"] = modalities.ModalityType.SYMBOL_WEIGHTS_ALL
    p.input_space_id = problem.SpaceID.GENERIC


@registry.register_problem
class ImageImagenet32Small(ImageImagenet):
  """Imagenet small from the pixel cnn paper."""

  @property
  def is_small(self):
    return False  # Modalities like for CIFAR.

  @property
  def num_classes(self):
    return 1000

  @property
  def train_shards(self):
    return 1024

  @property
  def dev_shards(self):
    return 10

  def preprocess_example(self, example, mode, unused_hparams):
    example["inputs"].set_shape([_IMAGENET_SMALL_IMAGE_SIZE,
                                 _IMAGENET_SMALL_IMAGE_SIZE, 3])
    example["inputs"] = tf.to_int64(example["inputs"])
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
    p.modality = {"inputs": modalities.ModalityType.IDENTITY,
                  "targets": modalities.ModalityType.IDENTITY}
    p.vocab_size = {"inputs": 256,
                    "targets": 256}
    p.batch_size_multiplier = 256
    p.input_space_id = 1
    p.target_space_id = 1


# The following preprocessing functions were taken from
# cloud_tpu/models/resnet/resnet_preprocessing.py
# ==============================================================================
def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: `Tensor` image of shape [height, width, channels].
    offset_height: `Tensor` indicating the height offset.
    offset_width: `Tensor` indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3), ["Rank of image must be equal to 3."])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ["Crop size greater than the image size."])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: `Tensor` of image (it will be converted to floats in [0, 1]).
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(scope, default_name="distorted_bounding_box_crop",
                     values=[image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox


def _random_crop(image, size):
  """Make a random crop of (`size` x `size`)."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  random_image, bbox = distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=1,
      scope=None)
  bad = _at_least_x_are_true(tf.shape(image), tf.shape(random_image), 3)

  image = tf.cond(
      bad, lambda: _center_crop(_do_scale(image, size), size),
      lambda: tf.image.resize_bicubic([random_image], [size, size])[0])
  return image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def _at_least_x_are_true(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are true."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _do_scale(image, size):
  """Rescale the image by scaling the smaller spatial dimension to `size`."""
  shape = tf.cast(tf.shape(image), tf.float32)
  w_greater = tf.greater(shape[0], shape[1])
  shape = tf.cond(w_greater,
                  lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
                  lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

  return tf.image.resize_bicubic([image], shape)[0]


def _center_crop(image, size):
  """Crops to center of image with specified `size`."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  offset_height = ((image_height - size) + 1) / 2
  offset_width = ((image_width - size) + 1) / 2
  image = _crop(image, offset_height, offset_width, size, size)
  return image


def _normalize(image):
  """Normalize the image to zero mean and unit variance."""
  offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
  image -= offset

  scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
  image /= scale
  return image


def preprocess_for_train(image, image_size=224, normalize=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    image_size: int, how large the output image should be.
    normalize: bool, if True the image is normalized.

  Returns:
    A preprocessed image `Tensor`.
  """
  if normalize: image = tf.to_float(image) / 255.0
  image = _random_crop(image, image_size)
  if normalize: image = _normalize(image)
  image = _flip(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image


def preprocess_for_eval(image, image_size=224, normalize=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    image_size: int, how large the output image should be.
    normalize: bool, if True the image is normalized.

  Returns:
    A preprocessed image `Tensor`.
  """
  if normalize: image = tf.to_float(image) / 255.0
  image = _do_scale(image, image_size + 32)
  if normalize: image = _normalize(image)
  image = _center_crop(image, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image
