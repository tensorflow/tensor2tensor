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

"""Algorithmic generators test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import algorithmic

import tensorflow as tf


class AlgorithmicTest(tf.test.TestCase):

  def testIdentityGenerator(self):
    identity_problem = algorithmic.AlgorithmicIdentityBinary40()
    counter = 0
    for d in identity_problem.generator(3, 8, 10):
      counter += 1
      self.assertEqual(d["inputs"], d["targets"])
    self.assertEqual(counter, 10)

  def testReverseGenerator(self):
    reversing_problem = algorithmic.AlgorithmicReverseBinary40()
    counter = 0
    for d in reversing_problem.generator(3, 8, 10):
      counter += 1
      self.assertEqual(list(reversed(d["inputs"])), d["targets"])
    self.assertEqual(counter, 10)

  def testZipfDistribution(self):
    # Following Zipf's Law with alpha equals 1: the first in rank is two times
    # more probable/frequent that the second in rank, three times more prob/freq
    # that the third in rank and so on.
    d = algorithmic.zipf_distribution(10, 1.0001)
    for i in xrange(len(d[1:])-1):
      self.assertEqual("%.4f" % (abs(d[i+1]-d[i+2])*(i+2)), "%.4f" % d[1])

  def testReverseGeneratorNlpLike(self):
    counter = 0
    for d in algorithmic.reverse_generator_nlplike(3, 8, 10):
      counter += 1
      self.assertEqual(list(reversed(d["inputs"])), d["targets"])
    self.assertEqual(counter, 10)

  def testLowerEndianToNumber(self):
    self.assertEqual(algorithmic.lower_endian_to_number([0], 2), 0)
    self.assertEqual(algorithmic.lower_endian_to_number([0], 7), 0)
    self.assertEqual(algorithmic.lower_endian_to_number([1], 2), 1)
    self.assertEqual(algorithmic.lower_endian_to_number([5], 8), 5)
    self.assertEqual(algorithmic.lower_endian_to_number([0, 1], 2), 2)
    self.assertEqual(algorithmic.lower_endian_to_number([0, 1, 1], 2), 6)
    self.assertEqual(algorithmic.lower_endian_to_number([7, 3, 1, 2], 10), 2137)

  def testNumberToLowerEndian(self):
    self.assertEqual(algorithmic.number_to_lower_endian(0, 2), [0])
    self.assertEqual(algorithmic.number_to_lower_endian(0, 7), [0])
    self.assertEqual(algorithmic.number_to_lower_endian(1, 2), [1])
    self.assertEqual(algorithmic.number_to_lower_endian(5, 8), [5])
    self.assertEqual(algorithmic.number_to_lower_endian(2, 2), [0, 1])
    self.assertEqual(algorithmic.number_to_lower_endian(6, 2), [0, 1, 1])
    self.assertEqual(algorithmic.number_to_lower_endian(2137, 10), [7, 3, 1, 2])

  def testAdditionGenerator(self):
    addition_problem = algorithmic.AlgorithmicAdditionBinary40()
    counter = 0
    for d in addition_problem.generator(4, 8, 10):
      counter += 1
      self.assertEqual(d["inputs"].count(4), 1)
      self.assertEqual(d["inputs"].count(5), 0)
      self.assertEqual(d["targets"].count(4), 0)
      self.assertEqual(d["targets"].count(5), 0)
    self.assertEqual(counter, 10)

  def testMultiplicationGenerator(self):
    multiplication_problem = algorithmic.AlgorithmicMultiplicationBinary40()
    counter = 0
    for d in multiplication_problem.generator(4, 8, 10):
      counter += 1
      self.assertEqual(d["inputs"].count(4), 1)
      self.assertEqual(d["inputs"].count(5), 0)
      self.assertEqual(d["targets"].count(4), 0)
      self.assertEqual(d["targets"].count(5), 0)
    self.assertEqual(counter, 10)


if __name__ == "__main__":
  tf.test.main()
