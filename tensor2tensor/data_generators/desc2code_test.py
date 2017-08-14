# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Tests for desc2code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from tensor2tensor.data_generators import desc2code

import tensorflow as tf

CODE_CPP_IN = """
  #include <iostream>

void main() {  // This comment will be removed
  // This too.
  //
  /* Not    this     one     */
\t
\t
  int a \t\n  =   3;//
//
}

"""

CODE_CPP_OUT = ("#include <iostream> void main() { /* Not this one */ int a = "
                "3; }")


class Desc2codeTest(tf.test.TestCase):

  def testCppPreprocess(self):
    """Check that the file correctly preprocess the code source."""
    cpp_pb = desc2code.ProgrammingDesc2codeCpp()

    self.assertEqual(  # Add space beween two lines
        cpp_pb.preprocess_target("firstline//comm1\nsecondline//comm2\n"),
        "firstline secondline")
    # Checking for boths comments and spaces
    self.assertEqual(cpp_pb.preprocess_target(CODE_CPP_IN), CODE_CPP_OUT)
    self.assertEqual(
        cpp_pb.preprocess_target("  not removed //abcd  "),
        "not removed //abcd")


if __name__ == "__main__":
  tf.test.main()
