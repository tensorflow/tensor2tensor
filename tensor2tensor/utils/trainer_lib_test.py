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

"""Tests for trainer_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

# Dependency imports

from tensor2tensor import models  # pylint: disable=unused-import
from tensor2tensor.data_generators import algorithmic
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem as problem_lib
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


@registry.register_problem
class TinyAlgo(algorithmic.AlgorithmicIdentityBinary40):

  def generate_data(self, data_dir, _):
    identity_problem = algorithmic.AlgorithmicIdentityBinary40()
    generator_utils.generate_files(
        identity_problem.generator(self.num_symbols, 40, 100000),
        self.training_filepaths(data_dir, 1, shuffled=True), 100)
    generator_utils.generate_files(
        identity_problem.generator(self.num_symbols, 400, 10000),
        self.dev_filepaths(data_dir, 1, shuffled=True), 100)


class TrainerLibTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    tmp_dir = tf.test.get_temp_dir()
    shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    cls.data_dir = tmp_dir

    # Generate a small test dataset
    registry.problem("tiny_algo").generate_data(cls.data_dir, None)

  def testExperiment(self):
    exp_fn = trainer_lib.create_experiment_fn(
        "transformer",
        "tiny_algo",
        self.data_dir,
        train_steps=1,
        eval_steps=1,
        min_eval_frequency=1,
        use_tpu=False)
    run_config = trainer_lib.create_run_config(
        model_dir=self.data_dir, num_gpus=0, use_tpu=False)
    hparams = registry.hparams("transformer_tiny_tpu")()
    exp = exp_fn(run_config, hparams)
    exp.test()

  def testModel(self):
    # HParams
    hparams = trainer_lib.create_hparams("transformer_tiny",
                                         data_dir=self.data_dir,
                                         problem_name="tiny_algo")

    # Dataset
    problem = hparams.problem_instances[0]
    dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, self.data_dir)
    dataset = dataset.repeat(None).padded_batch(10, dataset.output_shapes)
    features = dataset.make_one_shot_iterator().get_next()
    features = problem_lib.standardize_shapes(features)

    # Model
    model = registry.model("transformer")(hparams, tf.estimator.ModeKeys.TRAIN)
    logits, losses = model(features)

    self.assertTrue("training" in losses)
    loss = losses["training"]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      logits_val, loss_val = sess.run([logits, loss])
      logits_shape = list(logits_val.shape)
      logits_shape[1] = None
      self.assertAllEqual(logits_shape, [10, None, 1, 1, 4])
      self.assertEqual(loss_val.shape, tuple())


if __name__ == "__main__":
  tf.test.main()
