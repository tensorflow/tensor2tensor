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

"""Basic nets tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

from tensor2tensor.data_generators import mnist  # pylint: disable=unused-import
from tensor2tensor.models import basic
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf


class BasicTest(tf.test.TestCase):

  def testBasicFcRelu(self):
    x = np.random.randint(256, size=(1, 28, 28, 1))
    y = np.random.randint(10, size=(1, 1))
    hparams = trainer_lib.create_hparams(
        "basic_fc_small", problem_name="image_mnist", data_dir=".")
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = basic.BasicFcRelu(hparams, tf.estimator.ModeKeys.TRAIN)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (1, 1, 1, 1, 10))


if __name__ == "__main__":
  tf.test.main()
