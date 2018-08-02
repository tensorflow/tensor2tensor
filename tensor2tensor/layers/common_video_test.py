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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.layers import common_video

import tensorflow as tf


class CommonVideoTest(tf.test.TestCase):

  @staticmethod
  def RunScheduledSampleFunc(func, var, batch_size):
    ground_truth_x = list(range(1, batch_size+1))
    generated_x = [-x for x in ground_truth_x]
    ground_truth_x = tf.convert_to_tensor(ground_truth_x)
    generated_x = tf.convert_to_tensor(generated_x)
    ss_out = func(ground_truth_x, generated_x, batch_size, var)
    with tf.Session() as session:
      output = session.run([ground_truth_x, generated_x, ss_out])
    return output

  def testScheduledSampleProbStart(self):
    ground_truth_x, _, ss_out = CommonVideoTest.RunScheduledSampleFunc(
        common_video.scheduled_sample_prob, 1.0, 10)
    self.assertAllEqual(ground_truth_x, ss_out)

  def testScheduledSampleProbMid(self):
    _, _, ss_out = CommonVideoTest.RunScheduledSampleFunc(
        common_video.scheduled_sample_prob, 0.5, 1000)
    positive_count = np.sum(ss_out > 0)
    self.assertAlmostEqual(positive_count / 1000.0, 0.5, places=2)

  def testScheduledSampleProbEnd(self):
    _, generated_x, ss_out = CommonVideoTest.RunScheduledSampleFunc(
        common_video.scheduled_sample_prob, 0.0, 10)
    self.assertAllEqual(generated_x, ss_out)

  def testScheduledSampleCountStart(self):
    ground_truth_x, _, ss_out = CommonVideoTest.RunScheduledSampleFunc(
        common_video.scheduled_sample_count, 10, 10)
    self.assertAllEqual(ground_truth_x, ss_out)

  def testScheduledSampleCountMid(self):
    _, _, ss_out = CommonVideoTest.RunScheduledSampleFunc(
        common_video.scheduled_sample_count, 5, 10)
    positive_count = np.sum(ss_out > 0)
    self.assertEqual(positive_count, 5)

  def testScheduledSampleCountEnd(self):
    _, generated_x, ss_out = CommonVideoTest.RunScheduledSampleFunc(
        common_video.scheduled_sample_count, 0, 10)
    self.assertAllEqual(generated_x, ss_out)

  def testDynamicTileAndConcat(self):
    with tf.Graph().as_default():
      # image = (1 X 4 X 4 X 1)
      image = [[1, 2, 3, 4],
               [2, 4, 5, 6],
               [7, 8, 9, 10],
               [7, 9, 10, 1]]
      image_t = tf.expand_dims(tf.expand_dims(image, axis=0), axis=-1)
      image_t = tf.cast(image_t, dtype=tf.float32)

      # latent = (1 X 2)
      latent = np.array([[90, 100]])
      latent_t = tf.cast(tf.convert_to_tensor(latent), dtype=tf.float32)

      with tf.Session() as session:
        tiled = common_video.tile_and_concat(
            image_t, latent_t)
        tiled_np, image_np = session.run([tiled, image_t])
        tiled_latent = tiled_np[0, :, :, -1]
        self.assertAllEqual(tiled_np.shape, (1, 4, 4, 2))

        self.assertAllEqual(tiled_np[:, :, :, :1], image_np)
        self.assertAllEqual(
            tiled_latent,
            [[90, 90, 90, 90],
             [100, 100, 100, 100],
             [90, 90, 90, 90],
             [100, 100, 100, 100]])


if __name__ == "__main__":
  tf.test.main()
