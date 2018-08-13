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
"""Utilities for VQA data sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

# some functions are copied and modified from
# vgg_preprocessing and inception_preprocessing in
# models/research/slim/preprocessing/

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
    the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(
      tf.greater(height, width), lambda: smallest_side / width,
      lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
    the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_images(
      image, size=[new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)

  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def _flip(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def _distort_color(image, color_ordering=0, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, "distort_color", [image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      raise ValueError("color_ordering must be in [0, 3]")

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def _apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)
  ])[0]


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError("Input must be of size [height, width, C>0]")
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError("len(means) must match the number of channels")

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def vqa_v2_preprocess_image(
    image,
    height,
    width,
    mode,
    resize_side=512,
    distort=True,
    image_model_fn="resnet_v1_152",
):
  """vqa v2 preprocess image."""

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  assert resize_side > 0
  if resize_side:
    image = _aspect_preserving_resize(image, resize_side)
  if mode == tf.estimator.ModeKeys.TRAIN:
    image = tf.random_crop(image, [height, width, 3])
  else:
    # Central crop, assuming resize_height > height, resize_width > width.
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)

  image = tf.clip_by_value(image, 0.0, 1.0)

  if mode == tf.estimator.ModeKeys.TRAIN and distort:
    image = _flip(image)
    num_distort_cases = 4
    # pylint: disable=unnecessary-lambda
    image = _apply_with_random_selector(
        image, lambda x, ordering: _distort_color(x, ordering),
        num_cases=num_distort_cases)

  if image_model_fn.startswith("resnet_v1"):
    # resnet_v1 uses vgg preprocessing
    image = image * 255.
    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
  elif image_model_fn.startswith("resnet_v2"):
    # resnet v2 uses inception preprocessing
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

  return image
