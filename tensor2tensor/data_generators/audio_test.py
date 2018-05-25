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
"""Tests for tensor2tensor.data_generators.audio."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
from tensor2tensor.data_generators import audio

import tensorflow as tf


class AudioTest(tf.test.TestCase):

  def testDataCollection(self):
    # Generate a trivial source and target file.
    tmp_dir = self.get_temp_dir()
    test_files = [
        "dir1/file1",
        "dir1/file2",
        "dir1/dir2/file3",
        "dir1/dir2/dir3/file4",
    ]
    for filename in test_files:
      input_filename = os.path.join(tmp_dir, filename + ".WAV")
      target_filename = os.path.join(tmp_dir, filename + ".WRD")
      directories = os.path.dirname(input_filename)
      if not os.path.exists(directories):
        os.makedirs(directories)
      io.open(input_filename, "wb")
      io.open(target_filename, "wb")

    data_dict = audio._collect_data(tmp_dir, ".WAV", ".WRD")
    expected = [os.path.join(tmp_dir, filename) for filename in test_files]
    self.assertEqual(sorted(list(data_dict)), sorted(expected))

    # Clean up.
    for filename in test_files:
      os.remove(os.path.join(tmp_dir, "%s.WAV" % filename))
      os.remove(os.path.join(tmp_dir, "%s.WRD" % filename))


if __name__ == "__main__":
  tf.test.main()
