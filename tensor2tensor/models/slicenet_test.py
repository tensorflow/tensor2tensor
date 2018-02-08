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

"""Tests for SliceNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

from tensor2tensor.data_generators import cifar  # pylint: disable=unused-import
from tensor2tensor.layers import modalities  # pylint: disable=unused-import
from tensor2tensor.models import slicenet
from tensor2tensor.utils import registry

import tensorflow as tf


class SliceNetTest(tf.test.TestCase):

  def testSliceNet(self):
    x = np.random.random_integers(0, high=255, size=(3, 5, 5, 3))
    y = np.random.random_integers(0, high=9, size=(3, 5, 1, 1))
    hparams = slicenet.slicenet_params1_tiny()
    hparams.add_hparam("data_dir", "")
    problem = registry.problem("image_cifar10")
    p_hparams = problem.get_hparams(hparams)
    hparams.problems = [p_hparams]
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
          "target_space_id": tf.constant(1, dtype=tf.int32),
      }
      model = slicenet.SliceNet(hparams, tf.estimator.ModeKeys.TRAIN,
                                p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (3, 1, 1, 1, 10))


if __name__ == "__main__":
  tf.test.main()
