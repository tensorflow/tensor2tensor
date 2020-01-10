# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Tests for video utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensor2tensor.layers import common_video
from tensor2tensor.utils import test_utils

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


class CommonVideoTest(parameterized.TestCase, tf.test.TestCase):

  def _run_scheduled_sample_func(self, func, var, batch_size):
    ground_truth_x = list(range(1, batch_size+1))
    generated_x = [-x for x in ground_truth_x]
    ground_truth_x = tf.convert_to_tensor(ground_truth_x)
    generated_x = tf.convert_to_tensor(generated_x)
    ss_out = func(ground_truth_x, generated_x, batch_size, var)
    output = self.evaluate([ground_truth_x, generated_x, ss_out])
    return output

  @test_utils.run_in_graph_and_eager_modes()
  def testScheduledSampleProbStart(self):
    ground_truth_x, _, ss_out = self._run_scheduled_sample_func(
        common_video.scheduled_sample_prob, 1.0, 10)
    self.assertAllEqual(ground_truth_x, ss_out)

  @test_utils.run_in_graph_and_eager_modes()
  def testScheduledSampleProbMid(self):
    _, _, ss_out = self._run_scheduled_sample_func(
        common_video.scheduled_sample_prob, 0.5, 1000)
    positive_count = np.sum(ss_out > 0)
    self.assertAlmostEqual(positive_count / 1000.0, 0.5, places=1)

  @test_utils.run_in_graph_and_eager_modes()
  def testScheduledSampleProbEnd(self):
    _, generated_x, ss_out = self._run_scheduled_sample_func(
        common_video.scheduled_sample_prob, 0.0, 10)
    self.assertAllEqual(generated_x, ss_out)

  @test_utils.run_in_graph_and_eager_modes()
  def testScheduledSampleCountStart(self):
    ground_truth_x, _, ss_out = self._run_scheduled_sample_func(
        common_video.scheduled_sample_count, 10, 10)
    self.assertAllEqual(ground_truth_x, ss_out)

  @test_utils.run_in_graph_and_eager_modes()
  def testScheduledSampleCountMid(self):
    _, _, ss_out = self._run_scheduled_sample_func(
        common_video.scheduled_sample_count, 5, 10)
    positive_count = np.sum(ss_out > 0)
    self.assertEqual(positive_count, 5)

  @test_utils.run_in_graph_and_eager_modes()
  def testScheduledSampleCountEnd(self):
    _, generated_x, ss_out = self._run_scheduled_sample_func(
        common_video.scheduled_sample_count, 0, 10)
    self.assertAllEqual(generated_x, ss_out)

  @test_utils.run_in_graph_and_eager_modes()
  def testDynamicTileAndConcat(self):
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

    tiled = common_video.tile_and_concat(
        image_t, latent_t)
    tiled_np, image_np = self.evaluate([tiled, image_t])
    tiled_latent = tiled_np[0, :, :, -1]
    self.assertAllEqual(tiled_np.shape, (1, 4, 4, 2))

    self.assertAllEqual(tiled_np[:, :, :, :1], image_np)
    self.assertAllEqual(
        tiled_latent,
        [[90, 90, 90, 90],
         [100, 100, 100, 100],
         [90, 90, 90, 90],
         [100, 100, 100, 100]])

  @test_utils.run_in_graph_mode_only()
  def testGifSummary(self):
    for c in (1, 3):
      images_shape = (1, 12, 48, 64, c)  # batch, time, height, width, channels
      images = np.random.randint(256, size=images_shape).astype(np.uint8)

      with self.test_session():
        summary = common_video.gif_summary(
            "gif", tf.convert_to_tensor(images), fps=10)
        summary_string = summary.eval()

      summary = tf.Summary()
      summary.ParseFromString(summary_string)

      self.assertEqual(1, len(summary.value))
      self.assertTrue(summary.value[0].HasField("image"))
      encoded = summary.value[0].image.encoded_image_string

      self.assertEqual(encoded, common_video._encode_gif(images[0], fps=10))  # pylint: disable=protected-access

  def check_if_patch_exists(self, videos, video_patches, num_frames):
    """Check that given patch is present in video."""
    for video, video_patch in zip(videos, video_patches):
      total_frames = len(video)
      is_present = []
      for start_ind in range(total_frames - num_frames + 1):
        curr_patch = video[start_ind: start_ind + num_frames]
        is_present.append(np.allclose(curr_patch, video_patch))
      self.assertTrue(np.any(is_present))

  def testBasicLstm(self):
    """Tests that the parameters of the LSTM are shared across time."""
    with tf.Graph().as_default():
      state = None
      for _ in range(10):
        inputs = tf.random_uniform(shape=(32, 16))
        _, state = common_video.basic_lstm(
            inputs, state, num_units=100, name="basic")
      num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
      # 4 * ((100 + 16)*100 + 100) => 4 * (W_{xh} + W_{hh} + b)
      self.assertEqual(num_params, 46800)

  @parameterized.named_parameters(
      ("two_frames", 2), ("ten_frames", 10), ("default", -1))
  def testExtractRandomVideoPatch(self, num_frames=2):
    with tf.Graph().as_default():
      rng = np.random.RandomState(0)
      video_np = rng.randint(0, 255, size=(12, 20, 256, 256, 3))
      video = tf.convert_to_tensor(video_np)
      video_patch = common_video.extract_random_video_patch(
          video, num_frames=num_frames)
      with tf.Session() as sess:
        video_patch_np = sess.run(video_patch)
        if num_frames != -1:
          self.assertEqual(video_patch_np.shape, (12, num_frames, 256, 256, 3))
          self.check_if_patch_exists(video_np, video_patch_np, num_frames)
        else:
          self.assertTrue(np.allclose(video_np, video_patch_np))


if __name__ == "__main__":
  tf.test.main()
