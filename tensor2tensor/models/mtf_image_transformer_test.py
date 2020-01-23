# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Tests for Image Transformer on Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf

import numpy as np
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import mtf_image_transformer

import tensorflow.compat.v1 as tf

# Constants shared between all functions.
BATCH_SIZE = 8
IMG_LENGTH = 8
VOCAB_SIZE = 256


def get_model(hparams=None,
              mode=tf.estimator.ModeKeys.TRAIN,
              model_cls=mtf_image_transformer.MtfImageTransformer):
  if hparams is None:
    hparams = mtf_image_transformer.mtf_image_transformer_single()
  hparams.max_length = IMG_LENGTH*IMG_LENGTH
  hparams.batch_size = BATCH_SIZE
  hparams.img_len = IMG_LENGTH
  hparams.num_channels = 1

  p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE,
                                                   VOCAB_SIZE,
                                                   hparams)
  del p_hparams.modality["inputs"]
  hparams.problem_hparams = p_hparams

  targets = np.random.randint(
      VOCAB_SIZE, size=(BATCH_SIZE, IMG_LENGTH, IMG_LENGTH, 1, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
  }

  return model_cls(hparams, mode, p_hparams), features, hparams


def get_placement_mesh(hparams):
  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  mesh_shape = mtf.convert_to_shape(hparams.mesh_shape)

  mesh_devices = [""] * mesh_shape.size
  mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      mesh_shape, hparams.layout, mesh_devices)
  return mesh, mesh_impl


class MtfImageTransformerTest(tf.test.TestCase):

  def testMtfImageTransformer(self):
    hparams = mtf_image_transformer.mtf_image_transformer_single()

    # need to know layout ahead of time for local attention.
    hparams.mesh_shape = ""
    hparams.layout = ""
    model, features, hparams = get_model(hparams)
    mesh, mesh_impl = get_placement_mesh(hparams)

    logits, _ = model.mtf_model_fn(features, mesh)
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    tf_group = lowering.copy_masters_to_slices()
    tf_logits = lowering.export_to_tf_tensor(logits)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(tf_group)
      res = session.run(tf_logits)
    self.assertEqual(res.shape,
                     (BATCH_SIZE, IMG_LENGTH, IMG_LENGTH,
                      hparams.num_channels, VOCAB_SIZE))

  def testMtfImageTransformerDataParallel(self):
    hparams = mtf_image_transformer.mtf_image_transformer_single()

    # need to know layout ahead of time for local attention.
    hparams.mesh_shape = "all:2"
    hparams.layout = "batch:all"
    model, features, hparams = get_model(hparams)
    mesh, mesh_impl = get_placement_mesh(hparams)

    logits, _ = model.mtf_model_fn(features, mesh)
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    tf_group = lowering.copy_masters_to_slices()
    tf_logits = lowering.export_to_tf_tensor(logits)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(tf_group)
      res = session.run(tf_logits)
    self.assertEqual(res.shape,
                     (BATCH_SIZE, IMG_LENGTH, IMG_LENGTH,
                      hparams.num_channels, VOCAB_SIZE))

  def testMtfImageTransformerModelParallel(self):
    hparams = mtf_image_transformer.mtf_image_transformer_single()

    # need to know layout ahead of time for local attention.
    hparams.mesh_shape = "all:2"
    hparams.layout = "length:all"
    model, features, hparams = get_model(hparams)
    mesh, mesh_impl = get_placement_mesh(hparams)

    logits, _ = model.mtf_model_fn(features, mesh)
    lowering = mtf.Lowering(mesh.graph, {mesh: mesh_impl})
    tf_group = lowering.copy_masters_to_slices()
    tf_logits = lowering.export_to_tf_tensor(logits)

    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      session.run(tf_group)
      res = session.run(tf_logits)
    self.assertEqual(
        res.shape,
        (BATCH_SIZE, IMG_LENGTH, IMG_LENGTH, hparams.num_channels, VOCAB_SIZE))

if __name__ == "__main__":
  tf.test.main()
