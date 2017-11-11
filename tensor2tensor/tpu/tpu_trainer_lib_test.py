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

"""Tests for tpu_trainer_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.tpu import tpu_trainer_lib as lib
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import trainer_utils_test

import tensorflow as tf


class TpuTrainerTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    trainer_utils_test.TrainerUtilsTest.setUpClass()

  def testSmoke(self):
    data_dir = trainer_utils_test.TrainerUtilsTest.data_dir
    problem_name = "tiny_algo"
    model_name = "transformer"
    hparams_set = "transformer_tpu"

    hparams = trainer_utils.create_hparams(hparams_set, data_dir)
    trainer_utils.add_problem_hparams(hparams, problem_name)

    model_fn = lib.get_model_fn(model_name, hparams, use_tpu=False)
    input_fn = lib.get_input_fn(tf.estimator.ModeKeys.TRAIN, hparams)

    params = {"batch_size": 16}
    config = tf.contrib.tpu.RunConfig(
        tpu_config=tf.contrib.tpu.TPUConfig(num_shards=2))
    features, targets = input_fn(params)
    with tf.variable_scope("training"):
      spec = model_fn(features, targets, tf.estimator.ModeKeys.TRAIN, params,
                      config)

    self.assertTrue(spec.loss is not None)
    self.assertTrue(spec.train_op is not None)

    with tf.variable_scope("eval"):
      spec = model_fn(features, targets, tf.estimator.ModeKeys.EVAL, params,
                      config)
    self.assertTrue(spec.eval_metric_ops is not None)


if __name__ == "__main__":
  tf.test.main()
