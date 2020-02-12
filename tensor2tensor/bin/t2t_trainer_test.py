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

"""Tests for t2t_trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import trainer_lib_test

import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS


class TrainerTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    trainer_lib_test.TrainerLibTest.setUpClass()

  def testTrain(self):
    FLAGS.problem = "tiny_algo"
    FLAGS.model = "transformer"
    FLAGS.hparams_set = "transformer_tiny"
    FLAGS.train_steps = 1
    FLAGS.eval_steps = 1
    FLAGS.output_dir = tf.test.get_temp_dir()
    FLAGS.data_dir = tf.test.get_temp_dir()
    t2t_trainer.main(None)


if __name__ == "__main__":
  tf.test.main()
