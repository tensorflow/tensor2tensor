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

"""image_utils test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import decoding

import tensorflow.compat.v1 as tf


class ImageTest(tf.test.TestCase):

  def testImageAugmentation(self):
    x = np.random.rand(500, 500, 3)
    with self.test_session() as session:
      y = image_utils.image_augmentation(tf.constant(x))
      res = session.run(y)
    self.assertEqual(res.shape, (299, 299, 3))

  def testImageGenerator(self):
    # 2 random images
    np.random.seed(1111)  # To avoid any flakiness.
    image1 = np.random.randint(0, 255, size=(10, 12, 3))
    image2 = np.random.randint(0, 255, size=(10, 12, 3))
    # Call image generator on the 2 images with labels [1, 2].
    encoded_imgs, labels = [], []
    for dictionary in image_utils.image_generator([image1, image2], [1, 2]):
      self.assertEqual(
          sorted(list(dictionary)), [
              "image/class/label", "image/encoded", "image/format",
              "image/height", "image/width"
          ])
      self.assertEqual(dictionary["image/format"], ["png"])
      self.assertEqual(dictionary["image/height"], [12])
      self.assertEqual(dictionary["image/width"], [10])
      encoded_imgs.append(dictionary["image/encoded"])
      labels.append(dictionary["image/class/label"])

    # Check that the result labels match the inputs.
    self.assertEqual(len(labels), 2)
    self.assertEqual(labels[0], [1])
    self.assertEqual(labels[1], [2])

    # Decode images and check that they match the inputs.
    self.assertEqual(len(encoded_imgs), 2)
    image_t = tf.placeholder(dtype=tf.string)
    decoded_png_t = tf.image.decode_png(image_t)
    with self.test_session() as sess:
      encoded_img1 = encoded_imgs[0]
      self.assertEqual(len(encoded_img1), 1)
      decoded1 = sess.run(decoded_png_t, feed_dict={image_t: encoded_img1[0]})
      self.assertAllClose(decoded1, image1)
      encoded_img2 = encoded_imgs[1]
      self.assertEqual(len(encoded_img2), 1)
      decoded2 = sess.run(decoded_png_t, feed_dict={image_t: encoded_img2[0]})
      self.assertAllClose(decoded2, image2)

  def testMakeMultiscaleDivisible(self):
    image = tf.random_normal([256, 256, 3])
    resolutions = [8, 16, 64, 256]
    scaled_images = image_utils.make_multiscale(image, resolutions)
    self.assertEqual(scaled_images[0].shape, (8, 8, 3))
    self.assertEqual(scaled_images[1].shape, (16, 16, 3))
    self.assertEqual(scaled_images[2].shape, (64, 64, 3))
    self.assertEqual(scaled_images[3].shape, (256, 256, 3))

  def testMakeMultiscaleIndivisible(self):
    image = tf.random_normal([256, 256, 3])
    resolutions = [255]
    scaled_images = image_utils.make_multiscale(image, resolutions)
    self.assertEqual(scaled_images[0].shape, (255, 255, 3))

  def testMakeMultiscaleLarger(self):
    image = tf.random_normal([256, 256, 3])
    resolutions = [257]
    scaled_images = image_utils.make_multiscale(image, resolutions)
    self.assertEqual(scaled_images[0].shape, (257, 257, 3))

  def testMakeMultiscaleDilatedDivisible(self):
    image = tf.random_normal([256, 256, 3])
    resolutions = [8, 16, 64, 256]
    scaled_images = image_utils.make_multiscale_dilated(image, resolutions)
    self.assertEqual(scaled_images[0].shape, (8, 8, 3))
    self.assertEqual(scaled_images[1].shape, (16, 16, 3))
    self.assertEqual(scaled_images[2].shape, (64, 64, 3))
    self.assertEqual(scaled_images[3].shape, (256, 256, 3))

  def testMakeMultiscaleDilatedIndivisible(self):
    image = tf.random_normal([256, 256, 3])
    resolutions = [255]
    scaled_images = image_utils.make_multiscale_dilated(image, resolutions)
    self.assertEqual(scaled_images[0].shape, (256, 256, 3))

  def testMakeMultiscaleDilatedLarger(self):
    image = tf.random_normal([256, 256, 3])
    resolutions = [257]
    with self.assertRaisesRegexp(ValueError, "strides.* must be non-zero"):
      _ = image_utils.make_multiscale_dilated(image, resolutions)

  def testRandomShift(self):
    image = tf.random_normal([256, 256, 3])
    image_shift = image_utils.random_shift(image, wsr=0.1, hsr=0.1)
    self.assertEqual(image_shift.shape, [256, 256, 3])

  def testImageToSummaryValue(self):
    rng = np.random.RandomState(0)
    x = rng.randint(0, 255, (32, 32, 3))
    x_summary = image_utils.image_to_tf_summary_value(x, "X_image")
    self.assertEqual(x_summary.tag, "X_image")

  def testConvertPredictionsToImageSummaries(self):
    # Initialize predictions.
    rng = np.random.RandomState(0)
    x = rng.randint(0, 255, (32, 32, 3))
    predictions = [[{"outputs": x, "inputs": x}] * 50]

    decode_hparams = decoding.decode_hparams()
    # should return 20 summaries of images, 10 outputs and 10 inputs if
    # display_decoded_images is set to True.
    for display, summaries_length in zip([True, False], [20, 0]):
      decode_hparams.display_decoded_images = display
      decode_hooks = decoding.DecodeHookArgs(
          estimator=None, problem=None, output_dirs=None,
          hparams=decode_hparams, decode_hparams=decode_hparams,
          predictions=predictions)
      summaries = image_utils.convert_predictions_to_image_summaries(
          decode_hooks)
      self.assertEqual(len(summaries), summaries_length)
      if summaries:
        self.assertIsInstance(summaries[0], tf.Summary.Value)


if __name__ == "__main__":
  tf.test.main()
