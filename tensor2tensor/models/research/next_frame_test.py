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
from tensor2tensor.models.research import next_frame_emily
from tensor2tensor.models.research import next_frame_params
from tensor2tensor.models.research import next_frame_savp
from tensor2tensor.models.research import next_frame_sv2p
from tensor2tensor.utils import registry

import tensorflow as tf


def fill_hparams(hparams, in_frames, out_frames):
  hparams.video_num_input_frames = in_frames
  hparams.video_num_target_frames = out_frames
  problem = registry.problem("video_stochastic_shapes10k")
  p_hparams = problem.get_hparams(hparams)
  hparams.problem = problem
  hparams.problem_hparams = p_hparams
  hparams.tiny_mode = True
  return hparams


def action_modalities(hparams):
  hparams.problem_hparams.input_modality = {
      "inputs": ("video:l2raw", 256),
      "input_action": ("symbol:one_hot", 5)
  }
  hparams.problem_hparams.target_modality = {
      "targets": ("video:l2raw", 256),
      "target_action": ("symbol:one_hot", 5),
  }
  return hparams


def full_modalities(hparams):
  hparams.problem_hparams.input_modality = {
      "inputs": ("video:l2raw", 256),
      "input_reward": ("symbol:one_hot", 3),
      "input_action": ("symbol:one_hot", 5)
  }
  hparams.problem_hparams.target_modality = {
      "targets": ("video:l2raw", 256),
      "target_reward": ("symbol:one_hot", 3),
      "target_action": ("symbol:one_hot", 5),
  }
  return hparams


def create_basic_features(in_frames, out_frames):
  x = np.random.randint(0, 256, size=(8, in_frames, 64, 64, 3))
  y = np.random.randint(0, 256, size=(8, out_frames, 64, 64, 3))
  features = {
      "inputs": tf.constant(x, dtype=tf.int32),
      "targets": tf.constant(y, dtype=tf.int32),
  }
  return features


def create_action_features(in_frames, out_frames):
  features = create_basic_features(in_frames, out_frames)
  x = np.random.randint(0, 5, size=(8, in_frames, 1))
  y = np.random.randint(0, 5, size=(8, out_frames, 1))
  features["input_action"] = tf.constant(x, dtype=tf.int32)
  features["target_action"] = tf.constant(y, dtype=tf.int32)
  return features


def create_full_features(in_frames, out_frames):
  features = create_basic_features(in_frames, out_frames)
  x = np.random.randint(0, 5, size=(8, in_frames, 1))
  y = np.random.randint(0, 5, size=(8, out_frames, 1))
  features["input_reward"] = tf.constant(x, dtype=tf.int32)
  features["target_reward"] = tf.constant(y, dtype=tf.int32)
  return features


def get_tensor_shape(tensor):
  return tuple([d.value for d in tensor.shape])


class NextFrameTest(tf.test.TestCase):

  def RunModel(self, model, hparams, features):
    with self.test_session() as session:
      model = model(
          hparams, tf.estimator.ModeKeys.TRAIN)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    return res

  def TestVideoModel(self,
                     in_frames,
                     out_frames,
                     hparams,
                     model,
                     expected_last_dim,
                     upsample_method="conv2d_transpose"):
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams.upsample_method = upsample_method

    features = create_basic_features(in_frames, out_frames)
    output = self.RunModel(model, hparams, features)

    targets = features["targets"]
    expected_shape = get_tensor_shape(targets) + (expected_last_dim,)
    self.assertEqual(output.shape, expected_shape)

  def TestVideoModelWithActions(self,
                                in_frames,
                                out_frames,
                                hparams,
                                model,
                                expected_last_dim):
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams = action_modalities(hparams)
    hparams.reward_prediction = False

    features = create_action_features(in_frames, out_frames)
    output = self.RunModel(model, hparams, features)

    targets = features["targets"]
    expected_shape = get_tensor_shape(targets) + (expected_last_dim,)
    self.assertEqual(output.shape, expected_shape)

  def TestVideoModelWithActionAndRewards(self,
                                         in_frames,
                                         out_frames,
                                         hparams,
                                         model,
                                         expected_last_dim):
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams = full_modalities(hparams)

    features = create_full_features(in_frames, out_frames)

    res = self.RunModel(model, hparams, features)

    output, targets = res["targets"], features["targets"]
    expected_shape = get_tensor_shape(targets) + (expected_last_dim,)
    self.assertEqual(output.shape, expected_shape)

    output, targets = res["target_reward"], features["target_reward"]
    expected_shape = get_tensor_shape(targets)[:2] + (3,)
    self.assertEqual(output.shape, expected_shape)

  def TestOnVariousInputOutputSizes(self, hparams, model, expected_last_dim):
    self.TestVideoModel(1, 1, hparams, model, expected_last_dim)
    self.TestVideoModel(1, 6, hparams, model, expected_last_dim)
    self.TestVideoModel(4, 1, hparams, model, expected_last_dim)
    self.TestVideoModel(7, 5, hparams, model, expected_last_dim)

  def TestWithActions(self, hparams, model, expected_last_dim):
    test_func = self.TestVideoModelWithActionAndRewards
    test_func(1, 1, hparams, model, expected_last_dim)
    test_func(1, 6, hparams, model, expected_last_dim)
    test_func(4, 1, hparams, model, expected_last_dim)
    test_func(7, 5, hparams, model, expected_last_dim)

  def TestWithActionAndRewards(self, hparams, model, expected_last_dim):
    test_func = self.TestVideoModelWithActionAndRewards
    test_func(1, 1, hparams, model, expected_last_dim)
    test_func(1, 6, hparams, model, expected_last_dim)
    test_func(4, 1, hparams, model, expected_last_dim)
    test_func(7, 5, hparams, model, expected_last_dim)

  def TestOnVariousUpSampleLayers(self, hparams, model, expected_last_dim):
    self.TestVideoModel(4, 1, hparams, model, expected_last_dim,
                        upsample_method="bilinear_upsample_conv")
    self.TestVideoModel(4, 1, hparams, model, expected_last_dim,
                        upsample_method="nn_upsample_conv")

  def testBasic(self):
    self.TestOnVariousInputOutputSizes(
        next_frame_params.next_frame(),
        next_frame.NextFrameBasic,
        256)

  def testStochastic(self):
    self.TestOnVariousInputOutputSizes(
        next_frame_params.next_frame_stochastic(),
        next_frame_sv2p.NextFrameStochastic,
        1)

  def testStochasticWithActionsAndRewards(self):
    self.TestWithActionAndRewards(
        next_frame_params.next_frame_stochastic(),
        next_frame_sv2p.NextFrameStochastic,
        1)

  def testStochasticTwoFrames(self):
    self.TestOnVariousInputOutputSizes(
        next_frame_params.next_frame_stochastic(),
        next_frame_sv2p.NextFrameStochasticTwoFrames,
        1)

  def testStochasticEmily(self):
    self.TestOnVariousInputOutputSizes(
        next_frame_params.next_frame_stochastic_emily(),
        next_frame_emily.NextFrameStochasticEmily,
        1)

  def testStochasticSavp(self):
    self.TestOnVariousInputOutputSizes(
        next_frame_params.next_frame_savp(),
        next_frame_savp.NextFrameSAVP,
        1)
    self.TestOnVariousUpSampleLayers(
        next_frame_params.next_frame_savp(),
        next_frame_savp.NextFrameSAVP,
        1)


if __name__ == "__main__":
  tf.test.main()
