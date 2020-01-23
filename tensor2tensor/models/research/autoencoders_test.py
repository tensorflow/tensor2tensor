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

"""Autoencoders tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.data_generators import mnist  # pylint: disable=unused-import
from tensor2tensor.models.research import autoencoders  # pylint: disable=unused-import
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf


class AutoencoderTest(tf.test.TestCase):

  def get_mnist_random_output(self, model_name, hparams_set=None,
                              mode=tf.estimator.ModeKeys.TRAIN):
    hparams_set = hparams_set or model_name
    x = np.random.randint(256, size=(1, 28, 28, 1))
    y = np.random.randint(10, size=(1, 1))
    features = {
        "targets": tf.constant(x, dtype=tf.int32),
        "inputs": tf.constant(y, dtype=tf.int32),
    }
    hparams = trainer_lib.create_hparams(
        hparams_set, problem_name="image_mnist_rev", data_dir=".")
    model = registry.model(model_name)(hparams, mode)
    tf.train.create_global_step()
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    return res

  @property
  def mnist_output_shape(self):
    return (1, 28, 28, 1, 256)

  def testAutoencoderBasic(self):
    res = self.get_mnist_random_output("autoencoder_basic")
    self.assertEqual(res.shape, self.mnist_output_shape)

  def testAutoencoderAutoregressive(self):
    res = self.get_mnist_random_output("autoencoder_autoregressive")
    self.assertEqual(res.shape, self.mnist_output_shape)

  def testAutoencoderResidual(self):
    res = self.get_mnist_random_output("autoencoder_residual")
    self.assertEqual(res.shape, self.mnist_output_shape)

  def testAutoencoderBasicDiscrete(self):
    res = self.get_mnist_random_output("autoencoder_basic_discrete")
    self.assertEqual(res.shape, self.mnist_output_shape)

  def testAutoencoderResidualDiscrete(self):
    res = self.get_mnist_random_output("autoencoder_residual_discrete")
    self.assertEqual(res.shape, self.mnist_output_shape)

  def testAutoencoderOrderedDiscrete(self):
    res = self.get_mnist_random_output("autoencoder_ordered_discrete")
    self.assertEqual(res.shape, self.mnist_output_shape)

  def testAutoencoderOrderedDiscreteVQ(self):
    res = self.get_mnist_random_output(
        "autoencoder_ordered_discrete", "autoencoder_ordered_discrete_vq")
    self.assertEqual(res.shape, self.mnist_output_shape)

  # TODO(lukaszkaiser): Re-enable test by conserving lost shape information
  # in autoencoder_stacked.
  # def testAutoencoderStacked(self):
  #  res = self.get_mnist_random_output("autoencoder_stacked")
  #  self.assertEqual(res.shape, self.mnist_output_shape)

if __name__ == "__main__":
  tf.test.main()
