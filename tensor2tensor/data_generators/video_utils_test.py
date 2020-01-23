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

"""video_utils test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
import numpy as np
from tensor2tensor.data_generators import video_generated  # pylint: disable=unused-import
from tensor2tensor.data_generators import video_utils
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


class VideoUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def get_predictions(self, num_decodes=2):
    rng = np.random.RandomState(0)
    # num_samples=4
    inputs = rng.randint(0, 255, (4, 2, 64, 64, 3))
    outputs = rng.randint(0, 255, (4, 5, 64, 64, 3))
    targets = rng.randint(0, 255, (4, 5, 64, 64, 3))
    predictions = []
    for input_, output, target in zip(inputs, outputs, targets):
      curr_pred = {"inputs": input_, "outputs": output, "targets": target}
      predictions.append(curr_pred)

    # num_decodes=2
    predictions = [predictions] * num_decodes
    problem = registry.problem("video_stochastic_shapes10k")
    return predictions, problem

  def testVideoAugmentation(self):
    # smoke-test, test for shapes.
    with tf.Graph().as_default():
      inputs = tf.random_uniform(shape=(3, 64, 64, 3))
      targets = tf.random_uniform(shape=(10, 64, 64, 3))
      features = {"inputs": inputs, "targets": targets}
      augment = video_utils.video_augmentation(
          features, hue=True, saturate=True, contrast=True)
      with tf.Session() as sess:
        augment_dict = sess.run(augment)
        self.assertEqual(augment_dict["inputs"].shape, (3, 64, 64, 3))
        self.assertEqual(augment_dict["targets"].shape, (10, 64, 64, 3))

  def testDecodeInMemoryTrue(self):
    predictions, problem = self.get_predictions()
    decode_hparams = decoding.decode_hparams()
    decode_hparams.decode_in_memory = True
    decode_hooks = decoding.DecodeHookArgs(
        estimator=None, problem=problem, output_dirs=None,
        hparams=decode_hparams, decode_hparams=decode_hparams,
        predictions=predictions)
    metrics = video_utils.summarize_video_metrics(decode_hooks)

  @parameterized.named_parameters(
      ("d5_o6", 5, 6))
      # ("d5", 5), ("d10", 10), ("d5_o6", 5, 6))
  def testConvertPredictionsToVideoSummaries(self, num_decodes=5,
                                             max_output_steps=5):
    # Initialize predictions.
    rng = np.random.RandomState(0)
    inputs = rng.randint(0, 255, (2, 32, 32, 3))
    outputs = rng.randint(0, 255, (max_output_steps, 32, 32, 3))
    targets = rng.randint(0, 255, (5, 32, 32, 3))

    # batch it up.
    prediction = [{"outputs": outputs, "inputs": inputs, "targets": targets}]*5
    predictions = [prediction] * num_decodes
    decode_hparams = decoding.decode_hparams(
        overrides="max_display_decodes=5")

    decode_hooks = decoding.DecodeHookArgs(
        estimator=None, problem=None, output_dirs=None,
        hparams=decode_hparams, decode_hparams=decode_hparams,
        predictions=predictions)
    summaries = video_utils.display_video_hooks(decode_hooks)

    for summary in summaries:
      self.assertIsInstance(summary, tf.Summary.Value)


if __name__ == "__main__":
  tf.test.main()
