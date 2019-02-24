# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Basic tests for emily's model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models.video import emily
from tensor2tensor.models.video import tests_utils


import tensorflow as tf


class NextFrameTest(tests_utils.BaseNextFrameTest):

  def testEmily(self):
    self.TestOnVariousInputOutputSizes(
        emily.next_frame_emily(),
        emily.NextFrameEmily,
        1)


if __name__ == "__main__":
  tf.test.main()
