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
from tensor2tensor.mesh_tensorflow import placement_mesh_impl

import tensorflow as tf


class MeshTensorFlowTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (mtf.Dimension("x", 5),),
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
      (mtf.Shape([mtf.Dimension("x", 4),
                  mtf.Dimension("y", 8)]),),
      ("x:4;y:8",),
      ("x:4.y:8",),
      ("x:4 y:8",),
      ("x:4,y:8",),
  )
  def testConvertToShape(self, inputs):
    shape = mtf.convert_to_shape(inputs)
    self.assertEqual(shape, mtf.Shape([mtf.Dimension("x", 4),
                                       mtf.Dimension("y", 8)]))

  def testConvertToShapeGenericInputs(self):
    shape = mtf.convert_to_shape([])
    self.assertEqual(shape.dims, [])
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

  def testTensorLayout(self):
    tensor_layout = mtf.TensorLayout([0, 2, 1])
    self.assertEqual(tensor_layout.mesh_axis_to_tensor_axis(0), ())
    self.assertEqual(tensor_layout.mesh_axis_to_tensor_axis(1), (0,))
    self.assertEqual(tensor_layout.mesh_axis_to_tensor_axis(2), (0, 2))
    tensor_layout = mtf.TensorLayout([None, 0])
    self.assertFalse(tensor_layout.is_fully_replicated)
    tensor_layout = mtf.TensorLayout([None, None, None])
    self.assertTrue(tensor_layout.is_fully_replicated)

  def testGraph(self):
    graph = mtf.Graph()
    self.assertLen(graph.operations, 0)
    self.assertLen(graph.tensors, 0)
    self.assertLen(graph.trainable_variables, 0)
    self.assertLen(graph.all_variables, 0)
    mesh = mtf.Mesh(graph, "mesh_test")
    _ = mtf.import_tf_tensor(mesh,
                             tf_tensor=tf.constant(0.),
                             shape=mtf.Shape([]))
    self.assertLen(graph.operations, 1)
    self.assertLen(graph.tensors, 1)
    self.assertLen(graph.trainable_variables, 0)
    self.assertLen(graph.all_variables, 0)
    _ = mtf.get_variable(mesh, "variable_0", mtf.Shape([]), trainable=True)
    self.assertLen(graph.operations, 2)
    self.assertLen(graph.tensors, 2)
    self.assertLen(graph.trainable_variables, 1)
    self.assertLen(graph.all_variables, 1)
    _ = mtf.get_variable(mesh, "variable_1", mtf.Shape([]), trainable=False)
    self.assertLen(graph.operations, 3)
    self.assertLen(graph.tensors, 3)
    self.assertLen(graph.trainable_variables, 1)
    self.assertLen(graph.all_variables, 2)

  def testLowering(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    inputs = tf.constant(0.)
    mtf_inputs = mtf.import_tf_tensor(mesh,
                                      tf_tensor=inputs,
                                      shape=mtf.Shape([]))
    mesh_impl = placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    outputs = lowering.export_to_tf_tensor(mtf_inputs)
    with self.test_session() as sess:
      inputs_value, outputs_value = sess.run([inputs, outputs])
    self.assertEqual(inputs_value, outputs_value)

    # Check that methods run without error.
    _ = lowering.copy_masters_to_slices()
    _ = lowering.copy_slices_to_masters()

  def testMesh(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    self.assertEqual(mesh.graph, graph)

  def testMeshImpl(self):
    shape = mtf.Shape([mtf.Dimension("batch", 4),
                       mtf.Dimension("model", 8)])
    layout_rules = mtf.LayoutRules([("batch", "batch"),
                                    ("d_ff", "model"),
                                    ("heads", "model")])
    mesh_impl = mtf.MeshImpl(shape=shape, layout_rules=layout_rules)
    self.assertEqual(mesh_impl.shape, shape)
    self.assertEqual(mesh_impl.ndims, len(shape))
    self.assertEqual(mesh_impl.layout_rules, layout_rules)
    self.assertEqual(mesh_impl.size, shape.size)
    self.assertTrue(mesh_impl.supports_control_dependencies)

    batch = mtf.Dimension("batch", 128)
    length = mtf.Dimension("length", 500)
    d_ff = mtf.Dimension("d_ff", 2048)
    heads = mtf.Dimension("heads", 8)
    self.assertEqual(mesh_impl.tensor_dimension_to_mesh_axis(batch), 0)
    self.assertEqual(mesh_impl.tensor_dimension_to_mesh_axis(d_ff), 1)
    self.assertEqual(mesh_impl.tensor_dimension_to_mesh_axis(heads), 1)
    self.assertEqual(mesh_impl.tensor_layout(mtf.Shape([batch, length, d_ff])),
                     mtf.TensorLayout([0, None, 1]))

if __name__ == "__main__":
  tf.test.main()
