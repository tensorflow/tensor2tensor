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
"""Tests for Mesh TensorFlow layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensor2tensor.layers import common_layers
from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
from tensor2tensor.mesh_tensorflow import mtf_layers
from tensor2tensor.mesh_tensorflow import placement_mesh_impl

import tensorflow as tf


class MtfLayersTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (4, True),
      (8, False),
  )
  def testDense(self, units, use_bias):
    batch = 2
    channels = 3
    inputs = tf.random_normal([batch, channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    channels_dim = mtf.Dimension("channels", channels)
    depth_dim = mtf.Dimension("depth", units)

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))
    mtf_outputs = mtf_layers.dense(mtf_inputs,
                                   output_dim=depth_dim,
                                   reduced_dims=[channels_dim],
                                   activation=mtf.relu,
                                   use_bias=use_bias)
    mesh_impl = placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    expected_outputs = tf.keras.layers.Dense(units=units,
                                             activation=tf.nn.relu,
                                             use_bias=use_bias)(inputs)
    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(tf_group)
      actual, expected = sess.run([actual_outputs, expected_outputs])

    self.assertEqual(actual.shape, expected.shape)

  def testLayerNorm(self):
    batch = 2
    channels = 3
    inputs = tf.random_normal([batch, channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    channels_dim = mtf.Dimension("channels", channels)

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))
    mtf_outputs = mtf_layers.layer_norm(mtf_inputs,
                                        dim=channels_dim)
    mesh_impl = placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    expected_outputs = common_layers.layer_norm(inputs)
    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(tf_group)
      actual, expected = sess.run([actual_outputs, expected_outputs])

    self.assertEqual(actual.shape, expected.shape)

  def testWeightsNonzero(self):
    inputs = tf.constant([[3, 1, 0], [1, 0, 0]])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", inputs.shape.as_list()[0])
    channels_dim = mtf.Dimension("channels", inputs.shape.as_list()[1])

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))
    mtf_outputs = mtf_layers.weights_nonzero(mtf_inputs)
    mesh_impl = placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    expected_outputs = common_layers.weights_nonzero(inputs)
    tf_group = lowering.copy_masters_to_slices()
    with self.test_session() as sess:
      sess.run(tf_group)
      actual, expected = sess.run([actual_outputs, expected_outputs])

    self.assertAllEqual(actual, expected)

  def testDenseReluDense(self):
    batch = 2
    channels = 3
    hidden = 5
    inputs = tf.random_normal([batch, channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    channels_dim = mtf.Dimension("channels", channels)
    hidden_dim = mtf.Dimension("hidden", hidden)

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))
    mtf_outputs = mtf_layers.dense_relu_dense(mtf_inputs,
                                              hidden_channels=hidden_dim)
    mesh_impl = placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(tf_group)
      actual = sess.run(actual_outputs)

    self.assertEqual(actual.shape, inputs.shape)

  @parameterized.parameters(
      (2, 16, 3, 4, 2, 2),
      (1, 8, 5, 3, 1, 4),
  )
  def testMaskedLocalAttention1D(self, batch, length, io_channels, kv_channels,
                                 heads, block_length):
    length_q = length
    length_m = length
    query = tf.random_normal([batch, length_q, io_channels])
    memory = tf.random_normal([batch, length_m, io_channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    length_q_dim = mtf.Dimension("length_q", length_q)
    length_m_dim = mtf.Dimension("length_m", length_m)
    io_channels_dim = mtf.Dimension("io_channels", io_channels)
    kv_channels_dim = mtf.Dimension("kv_channels", kv_channels)
    heads_dim = mtf.Dimension("heads", heads)

    mtf_query = mtf.import_tf_tensor(
        mesh, query,
        shape=mtf.Shape([batch_dim, length_q_dim, io_channels_dim]))
    mtf_memory = mtf.import_tf_tensor(
        mesh, memory,
        shape=mtf.Shape([batch_dim, length_m_dim, io_channels_dim]))
    mtf_outputs = mtf_layers.masked_local_attention_1d(
        mtf_query,
        mtf_memory,
        kv_channels=kv_channels_dim,
        heads=heads_dim,
        block_length=block_length)
    mesh_impl = placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(tf_group)
      actual = sess.run(actual_outputs)

    self.assertEqual(actual.shape, (batch, length_q, io_channels))

  @parameterized.parameters(
      (2, 4, 5, 7, 3, 1),
  )
  def testDotProductAttention(
      self, batch, heads, length_q, length_kv, depth_k, depth_v):
    query = tf.random_normal([batch, heads, length_q, depth_k])
    key = tf.random_normal([batch, heads, length_kv, depth_k])
    value = tf.random_normal([batch, heads, length_kv, depth_v])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    heads_dim = mtf.Dimension("heads", heads)
    length_q_dim = mtf.Dimension("length_q", length_q)
    length_kv_dim = mtf.Dimension("length_kv", length_kv)
    depth_k_dim = mtf.Dimension("depth_k", depth_k)
    depth_v_dim = mtf.Dimension("depth_v", depth_v)

    mtf_query = mtf.import_tf_tensor(
        mesh, query,
        shape=mtf.Shape(
            [batch_dim, heads_dim, length_q_dim, depth_k_dim]))
    mtf_key = mtf.import_tf_tensor(
        mesh, key,
        shape=mtf.Shape(
            [batch_dim, heads_dim, length_kv_dim, depth_k_dim]))
    mtf_value = mtf.import_tf_tensor(
        mesh, value,
        shape=mtf.Shape(
            [batch_dim, heads_dim, length_kv_dim, depth_v_dim]))
    mtf_outputs = mtf_layers.dot_product_attention(
        mtf_query,
        mtf_key,
        mtf_value,
        mask=None)
    mesh_impl = placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(tf_group)
      actual = sess.run(actual_outputs)

    self.assertEqual(actual.shape, (batch, heads, length_q, depth_v))

  @parameterized.parameters(
      (16, 4),
      (32, 8),
  )
  def testMultiheadAttention(self, kv_channels, heads):
    batch = 2
    length = 8
    channels = 3
    query = tf.random_normal([batch, length, channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    length_dim = mtf.Dimension("length", length)
    channels_dim = mtf.Dimension("channels", channels)
    kv_channels_dim = mtf.Dimension("kv_channels", kv_channels)
    heads_dim = mtf.Dimension("heads", heads)

    mtf_query = mtf.import_tf_tensor(
        mesh, query,
        shape=mtf.Shape([batch_dim, length_dim, channels_dim]))
    mtf_outputs = mtf_layers.multihead_attention(
        mtf_query,
        memory_antecedent=None,
        mask=None,
        kv_channels=kv_channels_dim,
        heads=heads_dim)
    mesh_impl = placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(tf_group)
      actual = sess.run(actual_outputs)

    self.assertEqual(actual.shape, query.shape)

if __name__ == "__main__":
  tf.test.main()
