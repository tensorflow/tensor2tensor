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
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils_test

import tensorflow as tf


class TpuTrainerTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    trainer_utils_test.TrainerUtilsTest.setUpClass()

  def testExperiment(self):
    exp_fn = lib.create_experiment_fn(
        "transformer",
        "tiny_algo",
        trainer_utils_test.TrainerUtilsTest.data_dir,
        train_steps=1,
        eval_steps=1,
        min_eval_frequency=1,
        use_tpu=False)
    run_config = lib.create_run_config(num_gpus=0, use_tpu=False)
    hparams = registry.hparams("transformer_tiny_tpu")()
    exp = exp_fn(run_config, hparams)
    exp.test()


if __name__ == "__main__":
  tf.test.main()
