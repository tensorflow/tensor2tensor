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

# Lint as: python2, python3
"""Tests for tensor2tensor.models.research.glow_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import numpy as np
from six.moves import range
from tensor2tensor import problems
from tensor2tensor.data_generators import cifar  # pylint: disable=unused-import
from tensor2tensor.models.research import glow
from tensor2tensor.utils import registry  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf

MODES = tf.estimator.ModeKeys


class GlowModelTest(tf.test.TestCase):

  def batch(self, one_shot_iterator, batch_size=16):
    x_batch, y_batch = [], []
    for _ in range(batch_size):
      curr = one_shot_iterator.get_next()
      x_batch.append(curr['inputs'])
      y_batch.append(curr['targets'])
    return tf.stack(x_batch), tf.stack(y_batch)

  def test_glow(self):
    with tf.Graph().as_default():
      hparams = glow.glow_hparams()
      hparams.depth = 15
      hparams.n_levels = 2
      hparams.init_batch_size = 256
      hparams.batch_size = 1
      hparams.data_dir = ''
      cifar_problem = problems.problem('image_cifar10_plain_random_shift')
      hparams.problem = cifar_problem
      model = glow.Glow(hparams, tf.estimator.ModeKeys.TRAIN)
      train_dataset = cifar_problem.dataset(MODES.TRAIN)
      one_shot = train_dataset.make_one_shot_iterator()
      x_batch, y_batch = self.batch(one_shot)
      features = {'inputs': x_batch, 'targets': y_batch}
      _, obj_dict = model.body(features)
      objective = obj_dict['training']
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Run initialization.
        init_op = tf.get_collection('glow_init_op')
        sess.run(init_op)

        # Run forward pass.
        obj_np = sess.run(objective)
        mean_obj = np.mean(obj_np)

        # Check that one forward-propagation does not NaN, i.e
        # initialization etc works as expected.
        self.assertTrue(mean_obj > 0 and mean_obj < 10.0)

  def test_glow_inference(self):
    hparams = glow.glow_hparams()
    hparams.depth = 15
    hparams.n_levels = 2
    hparams.data_dir = ''
    curr_dir = tempfile.mkdtemp()

    # Training pipeline
    with tf.Graph().as_default():
      cifar_problem = problems.problem('image_cifar10_plain_random_shift')
      hparams.problem = cifar_problem
      model = glow.Glow(hparams, tf.estimator.ModeKeys.TRAIN)
      train_dataset = cifar_problem.dataset(MODES.TRAIN)
      one_shot = train_dataset.make_one_shot_iterator()
      x_batch, y_batch = self.batch(one_shot)
      features = {'inputs': x_batch, 'targets': y_batch}
      model_path = os.path.join(curr_dir, 'model')
      model(features)

      with tf.Session() as session:
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())

        init_op = tf.get_collection('glow_init_op')
        session.run(init_op)
        z = session.run([model.z])
        mean_z = np.mean(z)
        is_undefined = np.isnan(mean_z) or np.isinf(mean_z)
        self.assertTrue(not is_undefined)
        saver.save(session, model_path)

    # Inference pipeline
    with tf.Graph().as_default():
      cifar_problem = problems.problem('image_cifar10_plain_random_shift')
      hparams.problem = cifar_problem
      model = glow.Glow(hparams, tf.estimator.ModeKeys.PREDICT)
      test_dataset = cifar_problem.dataset(MODES.EVAL)
      one_shot = test_dataset.make_one_shot_iterator()
      x_batch, y_batch = self.batch(one_shot)
      features = {'inputs': x_batch, 'targets': y_batch}
      model_path = os.path.join(curr_dir, 'model')

      predictions = model.infer(features)
      with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, model_path)
        predictions_np = session.run(predictions)
        self.assertTrue(np.all(predictions_np <= 255))
        self.assertTrue(np.all(predictions_np >= 0))

if __name__ == '__main__':
  tf.test.main()
