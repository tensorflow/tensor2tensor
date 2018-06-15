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
"""Basic tests for video prediction models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.data_generators import video_generated  # pylint: disable=unused-import
from tensor2tensor.models.research import next_frame
from tensor2tensor.utils import registry

import tensorflow as tf


class NextFrameTest(tf.test.TestCase):

  def TestVideoModel(self,
                     in_frames,
                     out_frames,
                     hparams,
                     model,
                     expected_last_dim):

    x = np.random.random_integers(0, high=255, size=(8, in_frames, 64, 64, 3))
    y = np.random.random_integers(0, high=255, size=(8, out_frames, 64, 64, 3))

    hparams.video_num_input_frames = in_frames
    hparams.video_num_target_frames = out_frames

    problem = registry.problem("video_stochastic_shapes10k")
    p_hparams = problem.get_hparams(hparams)
    hparams.problem = problem
    hparams.problem_hparams = p_hparams

    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = model(
          hparams, tf.estimator.ModeKeys.TRAIN)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    expected_shape = y.shape + (expected_last_dim,)
    self.assertEqual(res.shape, expected_shape)

  def TestBasicModel(self, in_frames, out_frames):
    self.TestVideoModel(
        in_frames,
        out_frames,
        next_frame.next_frame(),
        next_frame.NextFrameBasic,
        256)

  def testBasicModelSingleInputFrameSingleOutputFrames(self):
    self.TestBasicModel(1, 1)

  def testBasicModelSingleInputFrameMultiOutputFrames(self):
    self.TestBasicModel(1, 6)

  def testBasicModelMultiInputFrameSingleOutputFrames(self):
    self.TestBasicModel(4, 1)

  def testBasicModelMultiInputFrameMultiOutputFrames(self):
    self.TestBasicModel(7, 5)

  def TestStochasticModel(self, in_frames, out_frames):
    self.TestVideoModel(
        in_frames,
        out_frames,
        next_frame.next_frame_stochastic(),
        next_frame.NextFrameStochastic,
        1)

  def testStochasticModelSingleInputFrameSingleOutputFrames(self):
    self.TestStochasticModel(1, 1)

  def testStochasticModelSingleInputFrameMultiOutputFrames(self):
    self.TestStochasticModel(1, 6)

  def testStochasticModelMultiInputFrameSingleOutputFrames(self):
    self.TestStochasticModel(4, 1)

  def testStochasticModelMultiInputFrameMultiOutputFrames(self):
    self.TestStochasticModel(7, 5)


if __name__ == "__main__":
  tf.test.main()
