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
"""Tests for Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf

import tensorflow as tf


class MeshTensorFlowTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (mtf.Dimension(name="x", size=5),),
      (("x", 5),),
  )
  def testConvertToDimension(self, inputs):
    dimension = mtf.convert_to_dimension(inputs)
    self.assertEqual(dimension.name, "x")
    self.assertEqual(dimension.size, 5)

  def testConvertToDimensionGenericInputs(self):
    dimension = mtf.convert_to_dimension(None)
    self.assertEqual(dimension, None)
    with self.assertRaises(TypeError):
      mtf.convert_to_dimension(5)

  @parameterized.parameters(
      (mtf.Shape([mtf.Dimension(name="x", size=4),
                  mtf.Dimension(name="y", size=8)]),),
      ("x:4;y:8",),
      ("x:4.y:8",),
      ("x:4 y:8",),
      ("x:4,y:8",),
  )
  def testConvertToShape(self, inputs):
    shape = mtf.convert_to_shape(inputs)
    self.assertEqual(shape, mtf.Shape([mtf.Dimension(name="x", size=4),
                                       mtf.Dimension(name="y", size=8)]))

  def testConvertToShapeGenericInputs(self):
    shape = mtf.convert_to_shape(None)
    self.assertEqual(shape, None)
    with self.assertRaises(ValueError):
      mtf.convert_to_shape("x;4")

  @parameterized.parameters(
      (mtf.LayoutRules([("d_ff", "model"), ("heads", "model")]),),
      ("d_ff:model;heads:model",),
      ("d_ff:model.heads:model",),
      ("d_ff:model heads:model",),
      ("d_ff:model,heads:model",),
      ([("d_ff", "model"), ("heads", "model")],),
  )
  def testConvertToLayoutRules(self, inputs):
    layout_rules = mtf.convert_to_layout_rules(inputs)
    self.assertEqual(
        layout_rules._pairs,
        mtf.LayoutRules([("d_ff", "model"), ("heads", "model")])._pairs)

  def testConvertToLayoutRulesGenericInputs(self):
    with self.assertRaises(ValueError):
      mtf.convert_to_layout_rules("d_ff;heads")

if __name__ == "__main__":
  tf.test.main()
