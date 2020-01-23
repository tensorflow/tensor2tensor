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

"""Tests for Revnet."""

from tensor2tensor.models import revnet
import tensorflow.compat.v1 as tf


class RevnetTest(tf.test.TestCase):

  def testH(self):
    rev_block_input = tf.random_uniform([1, 299, 299, 3])
    rev_block_output = revnet.downsample_bottleneck(rev_block_input, 256)
    self.assertEqual(rev_block_output.get_shape().as_list(), [1, 299, 299, 256])

  def testHStride(self):
    rev_block_input = tf.random_uniform([2, 299, 299, 256])
    rev_block_output = revnet.downsample_bottleneck(
        rev_block_input, 512, stride=2, scope='HStride')
    self.assertEqual(rev_block_output.get_shape().as_list(), [2, 150, 150, 512])

  def testInit(self):
    images = tf.random_uniform([1, 299, 299, 3])
    x1, x2 = revnet.init(images, 32)
    self.assertEqual(x1.get_shape().as_list(), [1, 74, 74, 16])
    self.assertEqual(x2.get_shape().as_list(), [1, 74, 74, 16])

  def testInit3D(self):
    images = tf.random_uniform([1, 299, 299, 299, 3])
    x1, x2 = revnet.init(images, 32, dim='3d', scope='init3d')
    self.assertEqual(x1.get_shape().as_list(), [1, 74, 74, 74, 16])
    self.assertEqual(x2.get_shape().as_list(), [1, 74, 74, 74, 16])

  def testUnit1(self):
    x1 = tf.random_uniform([4, 74, 74, 256])
    x2 = tf.random_uniform([4, 74, 74, 256])
    x1, x2 = revnet.unit(x1, x2, block_num=1, depth=64,
                         first_batch_norm=True, num_layers=1)
    self.assertEqual(x1.get_shape().as_list(), [4, 74, 74, 256])
    self.assertEqual(x2.get_shape().as_list(), [4, 74, 74, 256])

  def testUnit2(self):
    x1 = tf.random_uniform([4, 74, 74, 256])
    x2 = tf.random_uniform([4, 74, 74, 256])
    x1, x2 = revnet.unit(x1, x2, block_num=2, depth=128,
                         num_layers=1, stride=2)
    self.assertEqual(x1.get_shape().as_list(), [4, 37, 37, 512])
    self.assertEqual(x2.get_shape().as_list(), [4, 37, 37, 512])

  def testUnit3(self):
    x1 = tf.random_uniform([1, 37, 37, 512])
    x2 = tf.random_uniform([1, 37, 37, 512])
    x1, x2 = revnet.unit(x1, x2, block_num=3, depth=256,
                         num_layers=10, stride=2)
    self.assertEqual(x1.get_shape().as_list(), [1, 19, 19, 1024])
    self.assertEqual(x2.get_shape().as_list(), [1, 19, 19, 1024])

  def testUnit4(self):
    x1 = tf.random_uniform([1, 19, 19, 1024])
    x2 = tf.random_uniform([1, 19, 19, 1024])
    x1, x2 = revnet.unit(x1, x2, block_num=4, depth=416,
                         num_layers=1, stride=2)
    self.assertEqual(x1.get_shape().as_list(), [1, 10, 10, 1664])
    self.assertEqual(x2.get_shape().as_list(), [1, 10, 10, 1664])

  def testUnit3D(self):
    x1 = tf.random_uniform([4, 74, 74, 74, 256])
    x2 = tf.random_uniform([4, 74, 74, 74, 256])
    x1, x2 = revnet.unit(x1, x2, block_num=5, depth=128,
                         num_layers=1, dim='3d', stride=2)
    self.assertEqual(x1.get_shape().as_list(), [4, 37, 37, 37, 512])
    self.assertEqual(x2.get_shape().as_list(), [4, 37, 37, 37, 512])

  def testFinalBlock(self):
    x1 = tf.random_uniform([5, 10, 10, 1024])
    x2 = tf.random_uniform([5, 10, 10, 1024])
    logits = revnet.final_block(x1, x2)
    self.assertEqual(logits.shape, [5, 1, 1, 2048])

  def testFinalBlock3D(self):
    x1 = tf.random_uniform([5, 10, 10, 10, 1024])
    x2 = tf.random_uniform([5, 10, 10, 10, 1024])
    logits = revnet.final_block(x1, x2, dim='3d', scope='FinalBlock3D')
    self.assertEqual(logits.shape, [5, 1, 1, 1, 2048])

  def testEndToEnd(self):
    images = tf.random_uniform([1, 299, 299, 3])
    hparams = revnet.revnet_base()
    hparams.mode = tf.estimator.ModeKeys.TRAIN
    logits = revnet.revnet(images, hparams)
    self.assertEqual(logits.shape, [1, 1, 1, 3328])

  def testEndToEnd3D(self):
    images = tf.random_uniform([1, 299, 299, 299, 3])
    hparams = revnet.revnet_base()
    hparams.dim = '3d'
    hparams.mode = tf.estimator.ModeKeys.TRAIN
    logits = revnet.revnet(images, hparams)
    self.assertEqual(logits.shape, [1, 1, 1, 1, 3328])

if __name__ == '__main__':
  tf.test.main()
