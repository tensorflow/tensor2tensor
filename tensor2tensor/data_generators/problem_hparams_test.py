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

"""Tests for tensor2tensor.problem_hparams."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.data_generators import problem_hparams

import tensorflow as tf


class ProblemHparamsTest(tf.test.TestCase):

  def testParseProblemName(self):
    problem_name = "base"
    self.assertEqual(problem_hparams.parse_problem_name(problem_name),
                     ("base", False, False))
    problem_name = "base_rev"
    self.assertEqual(
        problem_hparams.parse_problem_name(problem_name), ("base", True, False))
    problem_name = "base_copy"
    self.assertEqual(
        problem_hparams.parse_problem_name(problem_name), ("base", False, True))
    problem_name = "base_copy_rev"
    self.assertEqual(
        problem_hparams.parse_problem_name(problem_name), ("base", True, True))
    problem_name = "base_rev_copy"
    self.assertEqual(
        problem_hparams.parse_problem_name(problem_name), ("base", True, True))


if __name__ == "__main__":
  tf.test.main()
