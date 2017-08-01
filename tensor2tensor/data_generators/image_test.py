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

"""Image generators test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from tensor2tensor.data_generators import image

import tensorflow as tf


class ImageTest(tf.test.TestCase):

  def testImageGenerator(self):
    # 2 random images
    np.random.seed(1111)  # To avoid any flakiness.
    image1 = np.random.randint(0, 255, size=(10, 12, 3))
    image2 = np.random.randint(0, 255, size=(10, 12, 3))
    # Call image generator on the 2 images with labels [1, 2].
    encoded_imgs, labels = [], []
    for dictionary in image.image_generator([image1, image2], [1, 2]):
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


if __name__ == "__main__":
  tf.test.main()
