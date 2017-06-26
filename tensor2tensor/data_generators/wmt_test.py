# Copyright 2017 Google Inc.
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

"""WMT generators test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import tempfile

# Dependency imports

import six
from tensor2tensor.data_generators import wmt

import tensorflow as tf


class WMTTest(tf.test.TestCase):

  def testCharacterGenerator(self):
    # Generate a trivial source and target file.
    tmp_dir = self.get_temp_dir()
    (_, tmp_file_path) = tempfile.mkstemp(dir=tmp_dir)
    with io.open(tmp_file_path + ".src", "wb") as src_file:
      src_file.write("source1\n")
      src_file.write("source2\n")
    with io.open(tmp_file_path + ".tgt", "wb") as tgt_file:
      tgt_file.write("target1\n")
      tgt_file.write("target2\n")

    # Call character generator on the generated files.
    results_src, results_tgt = [], []
    for dictionary in wmt.character_generator(tmp_file_path + ".src",
                                              tmp_file_path + ".tgt"):
      self.assertEqual(sorted(list(dictionary)), ["inputs", "targets"])
      results_src.append(dictionary["inputs"])
      results_tgt.append(dictionary["targets"])

    # Check that the results match the files.
    self.assertEqual(len(results_src), 2)
    self.assertEqual("".join([six.int2byte(i)
                              for i in results_src[0]]), "source1")
    self.assertEqual("".join([six.int2byte(i)
                              for i in results_src[1]]), "source2")
    self.assertEqual("".join([six.int2byte(i)
                              for i in results_tgt[0]]), "target1")
    self.assertEqual("".join([six.int2byte(i)
                              for i in results_tgt[1]]), "target2")

    # Clean up.
    os.remove(tmp_file_path + ".src")
    os.remove(tmp_file_path + ".tgt")
    os.remove(tmp_file_path)


if __name__ == "__main__":
  tf.test.main()
