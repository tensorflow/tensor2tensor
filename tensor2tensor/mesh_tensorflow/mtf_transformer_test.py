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
"""Tests for Transformer on Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.mesh_tensorflow import mesh_tensorflow as mtf
from tensor2tensor.mesh_tensorflow import mtf_transformer
from tensor2tensor.mesh_tensorflow import placement_mesh_impl

import tensorflow as tf

# Constants shared between all functions.
BATCH_SIZE = 2
INPUT_LENGTH = 6
TARGET_LENGTH = 6
VOCAB_SIZE = 128


def get_model(hparams=None, mode=tf.estimator.ModeKeys.TRAIN,
              has_input=True, model_cls=mtf_transformer.MtfTransformer):
  if hparams is None:
    hparams = mtf_transformer.mtf_transformer_single()
  hparams.max_length = INPUT_LENGTH
  hparams.batch_size = BATCH_SIZE

  p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE, VOCAB_SIZE)
  if not has_input:
    p_hparams.input_modality = {}
  hparams.problem_hparams = p_hparams

  inputs = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
  targets = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
  if has_input:
    features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")

  return model_cls(hparams, mode, p_hparams), features, hparams


def get_placement_mesh(hparams):
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  mesh_shape = mtf.convert_to_shape(hparams.mesh_shape)

  mesh_devices = [""] * mesh_shape.size
  mesh_impl = placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, hparams.layout, mesh_devices)
  return mesh, mesh_impl


class MtfTransformerTest(tf.test.TestCase):

  def testMtfTransformer(self):
    hparams = mtf_transformer.mtf_transformer_single()

    model, features, hparams = get_model(hparams)
    hparams.mesh_shape = ""
    hparams.layout = ""
    mesh, mesh_impl = get_placement_mesh(hparams)

    logits, _ = model.mtf_model_fn(features, mesh)
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    tf_group = lowering.copy_masters_to_slices()
    tf_logits = lowering.export_to_tf_tensor(logits)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(tf_group)
      res = session.run(tf_logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, VOCAB_SIZE))

  def testMtfTransformerDataParallel(self):
    hparams = mtf_transformer.mtf_transformer_single()

    model, features, hparams = get_model(hparams)
    hparams.mesh_shape = "all:2"
    hparams.layout = "batch:all"
    mesh, mesh_impl = get_placement_mesh(hparams)

    logits, _ = model.mtf_model_fn(features, mesh)
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    tf_group = lowering.copy_masters_to_slices()
    tf_logits = lowering.export_to_tf_tensor(logits)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(tf_group)
      res = session.run(tf_logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, VOCAB_SIZE))

  def testMtfTransformerModelParallel(self):
    hparams = mtf_transformer.mtf_transformer_single()

    model, features, hparams = get_model(hparams)
    hparams.mesh_shape = "all:2"
    hparams.layout = "length:all"
    mesh, mesh_impl = get_placement_mesh(hparams)

    logits, _ = model.mtf_model_fn(features, mesh)
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    tf_group = lowering.copy_masters_to_slices()
    tf_logits = lowering.export_to_tf_tensor(logits)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(tf_group)
      res = session.run(tf_logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, VOCAB_SIZE))

  def testMtfTransformerDataModelParallel(self):
    hparams = mtf_transformer.mtf_transformer_single()

    model, features, hparams = get_model(hparams)
    hparams.mesh_shape = "batch:2;model:2"
    hparams.layout = "batch:batch;vocab:model;d_ff:model;heads:model"
    mesh, mesh_impl = get_placement_mesh(hparams)

    logits, _ = model.mtf_model_fn(features, mesh)
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    tf_group = lowering.copy_masters_to_slices()
    tf_logits = lowering.export_to_tf_tensor(logits)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(tf_group)
      res = session.run(tf_logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, VOCAB_SIZE))


if __name__ == "__main__":
  tf.test.main()
