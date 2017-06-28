# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Tests for vocab_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from ..utils import misc_utils


class MiscUtilsTest(tf.test.TestCase):

  def testFormatBpeText(self):
    bpe_line = (
        b"En@@ ough to make already reluc@@ tant men hesitate to take screening"
        b" tests ."
    )
    expected_result = (
        b"Enough to make already reluctant men hesitate to take screening tests"
        b" ."
    )
    self.assertEqual(expected_result,
                     misc_utils.format_bpe_text(bpe_line.split(b" ")))


if __name__ == '__main__':
  tf.test.main()
