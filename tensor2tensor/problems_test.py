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

"""tensor2tensor.problems test."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor import problems

import tensorflow.compat.v1 as tf


class ProblemsTest(tf.test.TestCase):

  def testImport(self):
    self.assertIsNotNone(problems)

if __name__ == "__main__":
  tf.test.main()
