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

"""Tests for tensor2tensor.data_generators.algorithmic_math."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import six
import sympy
from tensor2tensor.data_generators import algorithmic_math

import tensorflow as tf


class AlgorithmicMathTest(tf.test.TestCase):

  def testAlgebraInverse(self):
    dataset_objects = algorithmic_math.math_dataset_init(26)
    counter = 0
    for d in algorithmic_math.algebra_inverse(26, 0, 3, 10):
      counter += 1
      decoded_input = dataset_objects.int_decoder(d["inputs"])
      solve_var, expression = decoded_input.split(":")
      lhs, rhs = expression.split("=")

      # Solve for the solve-var.
      result = sympy.solve("%s-(%s)" % (lhs, rhs), solve_var)
      target_expression = dataset_objects.int_decoder(d["targets"])

      # Check that the target and sympy's solutions are equivalent.
      self.assertEqual(
          0, sympy.simplify(str(result[0]) + "-(%s)" % target_expression))
    self.assertEqual(counter, 10)

  def testAlgebraSimplify(self):
    dataset_objects = algorithmic_math.math_dataset_init(8, digits=5)
    counter = 0
    for d in algorithmic_math.algebra_simplify(8, 0, 3, 10):
      counter += 1
      expression = dataset_objects.int_decoder(d["inputs"])
      target = dataset_objects.int_decoder(d["targets"])

      # Check that the input and output are equivalent expressions.
      self.assertEqual(0, sympy.simplify("%s-(%s)" % (expression, target)))
    self.assertEqual(counter, 10)

  def testCalculusIntegrate(self):
    dataset_objects = algorithmic_math.math_dataset_init(
        8, digits=5, functions={"log": "L"})
    counter = 0
    for d in algorithmic_math.calculus_integrate(8, 0, 3, 10):
      counter += 1
      decoded_input = dataset_objects.int_decoder(d["inputs"])
      var, expression = decoded_input.split(":")
      target = dataset_objects.int_decoder(d["targets"])

      for fn_name, fn_char in six.iteritems(dataset_objects.functions):
        target = target.replace(fn_char, fn_name)

      # Take the derivative of the target.
      derivative = str(sympy.diff(target, var))

      # Check that the derivative of the integral equals the input.
      self.assertEqual(0, sympy.simplify("%s-(%s)" % (expression, derivative)))
    self.assertEqual(counter, 10)


if __name__ == "__main__":
  tf.test.main()
