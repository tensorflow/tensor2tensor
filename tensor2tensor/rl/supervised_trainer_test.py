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

"""Tests for supervised_trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.rl import supervised_trainer
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib_test

import tensorflow as tf


class TrainerNewTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    trainer_lib_test.TrainerLibTest.setUpClass()

  def testTrain(self):
    supervised_trainer.train(
        problem=registry.problem("tiny_algo"),
        model_name="transformer",
        hparams=registry.hparams("transformer_tiny"),
        train_steps=1,
        eval_steps=1,
        output_dir=tf.test.get_temp_dir(),
        data_dir=tf.test.get_temp_dir()
    )


if __name__ == "__main__":
  tf.test.main()
