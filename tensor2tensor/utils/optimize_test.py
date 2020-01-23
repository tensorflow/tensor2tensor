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

"""Tests for tensor2tensor.utils.optimize."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import optimize
import tensorflow.compat.v1 as tf


class OptimizeTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      "sgd",
      "SGD",
      "rms_prop",
      "RMSProp",
      "adagrad",
      "Adagrad",
      "adam",
      "Adam",
      "adam_w",
      "AdamW",
  )
  def test_names(self, opt_name):
    hparams = hparams_lib.create_hparams("basic_1")
    optimize.ConditionalOptimizer(opt_name, 0.1, hparams)


if __name__ == "__main__":
  tf.test.main()
