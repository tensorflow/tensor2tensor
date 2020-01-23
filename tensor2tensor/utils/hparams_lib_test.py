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

"""Tests for trainer_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.utils import hparams_lib

import tensorflow.compat.v1 as tf


class HparamsLibTest(tf.test.TestCase):

  def testCreateHparamsFromJson(self):
    # Get json_path
    pkg = os.path.abspath(__file__)
    pkg, _ = os.path.split(pkg)
    pkg, _ = os.path.split(pkg)
    json_path = os.path.join(
        pkg, "test_data", "transformer_test_ckpt", "hparams.json")

    # Create hparams
    hparams = hparams_lib.create_hparams_from_json(json_path)
    self.assertEqual(75, len(hparams.values()))


if __name__ == "__main__":
  tf.test.main()
