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

"""Utilties for testing video models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.data_generators import video_generated  # pylint: disable=unused-import

from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


def fill_hparams(hparams, in_frames, out_frames):
  hparams.video_num_input_frames = in_frames
  hparams.video_num_target_frames = out_frames
  problem = registry.problem("video_stochastic_shapes10k")
  p_hparams = problem.get_hparams(hparams)
  hparams.problem = problem
  hparams.problem_hparams = p_hparams
  hparams.tiny_mode = True
  hparams.reward_prediction = False
  return hparams


def action_modalities(hparams):
  """Modalities with actions."""
  hparams.problem_hparams.modality = {
      "inputs": modalities.ModalityType.VIDEO_L2_RAW,
      "input_action": modalities.ModalityType.SYMBOL,
      "targets": modalities.ModalityType.VIDEO_L2_RAW,
      "target_action": modalities.ModalityType.SYMBOL,
  }
  hparams.problem_hparams.vocab_size = {
      "inputs": 256,
      "input_action": 5,
      "targets": 256,
      "target_action": 5,
  }
  return hparams


def full_modalities(hparams):
  """Full modalities with actions and rewards."""
  hparams.problem_hparams.modality = {
      "inputs": modalities.ModalityType.VIDEO_L2_RAW,
      "input_action": modalities.ModalityType.SYMBOL,
      "input_reward": modalities.ModalityType.SYMBOL,
      "targets": modalities.ModalityType.VIDEO_L2_RAW,
      "target_action": modalities.ModalityType.SYMBOL,
      "target_reward": modalities.ModalityType.SYMBOL,
  }
  hparams.problem_hparams.vocab_size = {
      "inputs": 256,
      "input_action": 5,
      "input_reward": 3,
      "targets": 256,
      "target_action": 5,
      "target_reward": 3,
  }
  hparams.force_full_predict = True
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


class BaseNextFrameTest(tf.test.TestCase):
  """Base helper class for next frame tests."""

  def RunModel(self, model, hparams, features):
    with tf.Session() as session:
      model = model(hparams, tf.estimator.ModeKeys.TRAIN)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    return res

  def InferModel(self, model, hparams, features):
    with tf.Session() as session:
      model = model(hparams, tf.estimator.ModeKeys.PREDICT)
      output = model.infer(features)
      session.run(tf.global_variables_initializer())
      res = session.run(output)
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

  def TestVideoModelInfer(self,
                          in_frames,
                          out_frames,
                          hparams,
                          model,
                          expected_last_dim,
                          upsample_method="conv2d_transpose"):
    del expected_last_dim
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams.upsample_method = upsample_method

    features = create_basic_features(in_frames, out_frames)
    output = self.InferModel(model, hparams, features)

    self.assertTrue(isinstance(output, dict))
    self.assertTrue("outputs" in output.keys())
    self.assertTrue("scores" in output.keys())
    self.assertTrue("targets" in output.keys())
    expected_shape = get_tensor_shape(features["targets"])
    self.assertEqual(output["targets"].shape, expected_shape)

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

  def TestVideoModelWithActionsInfer(self,
                                     in_frames,
                                     out_frames,
                                     hparams,
                                     model,
                                     expected_last_dim):
    del expected_last_dim
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams = action_modalities(hparams)
    hparams.reward_prediction = False

    features = create_action_features(in_frames, out_frames)
    output = self.InferModel(model, hparams, features)

    self.assertTrue(isinstance(output, dict))
    self.assertTrue("outputs" in output.keys())
    self.assertTrue("scores" in output.keys())
    self.assertTrue("targets" in output.keys())
    expected_shape = get_tensor_shape(features["targets"])
    self.assertEqual(output["targets"].shape, expected_shape)

  def TestVideoModelWithActionAndRewards(self,
                                         in_frames,
                                         out_frames,
                                         hparams,
                                         model,
                                         expected_last_dim):
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams = full_modalities(hparams)
    hparams.reward_prediction = True

    features = create_full_features(in_frames, out_frames)

    res = self.RunModel(model, hparams, features)

    output, targets = res["targets"], features["targets"]
    expected_shape = get_tensor_shape(targets) + (expected_last_dim,)
    self.assertEqual(output.shape, expected_shape)

    output, targets = res["target_reward"], features["target_reward"]
    # Assuming Symbol Modality
    expected_shape = get_tensor_shape(targets)[:2] + (1, 1, 1, 1, 3,)
    self.assertEqual(output.shape, expected_shape)

  def TestVideoModelWithActionAndRewardsInfer(self,
                                              in_frames,
                                              out_frames,
                                              hparams,
                                              model,
                                              expected_last_dim):
    del expected_last_dim
    hparams = fill_hparams(hparams, in_frames, out_frames)
    hparams = full_modalities(hparams)
    hparams.reward_prediction = True

    features = create_full_features(in_frames, out_frames)

    output = self.InferModel(model, hparams, features)

    self.assertTrue(isinstance(output, dict))
    self.assertTrue("outputs" in output.keys())
    self.assertTrue("scores" in output.keys())
    self.assertTrue("targets" in output.keys())
    self.assertTrue("target_reward" in output.keys())
    expected_shape = get_tensor_shape(features["targets"])
    self.assertEqual(output["targets"].shape, expected_shape)
    expected_shape = get_tensor_shape(features["target_reward"])[:2]
    self.assertEqual(output["target_reward"].shape, expected_shape)

  def TestOnVariousInputOutputSizes(
      self, hparams, model, expected_last_dim, test_infer=True):
    test_funcs = [self.TestVideoModel]
    if test_infer:
      test_funcs += [self.TestVideoModelInfer]
    for test_func in test_funcs:
      test_func(1, 1, hparams, model, expected_last_dim)
      test_func(1, 6, hparams, model, expected_last_dim)
      test_func(4, 1, hparams, model, expected_last_dim)
      test_func(7, 5, hparams, model, expected_last_dim)

  def TestWithActions(self, hparams, model, expected_last_dim, test_infer=True):
    test_funcs = [self.TestVideoModelWithActions]
    if test_infer:
      test_funcs += [self.TestVideoModelWithActionsInfer]
    for test_func in test_funcs:
      test_func(1, 1, hparams, model, expected_last_dim)
      test_func(1, 6, hparams, model, expected_last_dim)
      test_func(4, 1, hparams, model, expected_last_dim)
      test_func(7, 5, hparams, model, expected_last_dim)

  def TestWithActionAndRewards(
      self, hparams, model, expected_last_dim, test_infer=True):
    test_funcs = [self.TestVideoModelWithActionAndRewards]
    if test_infer:
      test_funcs += [self.TestVideoModelWithActionAndRewardsInfer]
    for test_func in test_funcs:
      test_func(1, 1, hparams, model, expected_last_dim)
      test_func(1, 6, hparams, model, expected_last_dim)
      test_func(4, 1, hparams, model, expected_last_dim)
      test_func(7, 5, hparams, model, expected_last_dim)

  def TestOnVariousUpSampleLayers(self, hparams, model, expected_last_dim):
    self.TestVideoModel(4, 1, hparams, model, expected_last_dim,
                        upsample_method="bilinear_upsample_conv")
    self.TestVideoModel(4, 1, hparams, model, expected_last_dim,
                        upsample_method="nn_upsample_conv")
