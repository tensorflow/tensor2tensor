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

"""Tests for tensor2tensor.data_generators.common_voice."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensor2tensor.data_generators import common_voice

import tensorflow.compat.v1 as tf

pkg_dir, _ = os.path.split(__file__)
_TESTDATA = os.path.join(pkg_dir, "test_data")


class CommonVoiceTest(tf.test.TestCase):

  def testCollectData(self):
    output = common_voice._collect_data(_TESTDATA)
    self.assertEqual(1, len(output))

    # NOTE: No header.
    self.assertTrue("my_media" == output[0][0])
    self.assertTrue("my_label" == output[0][2])


if __name__ == "__main__":
  tf.test.main()
