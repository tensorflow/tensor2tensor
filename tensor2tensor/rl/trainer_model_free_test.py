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

"""Tests of basic flow of collecting trajectories and training PPO."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.rl import trainer_model_free
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS


class TrainTest(tf.test.TestCase):

  def _test_hparams_set(self, hparams_set):
    hparams = registry.hparams(hparams_set)
    FLAGS.output_dir = tf.test.get_temp_dir()
    trainer_model_free.train(hparams, FLAGS.output_dir,
                             env_problem_name=None)

  def test_train_pong(self):
    self._test_hparams_set("rlmf_tiny")

  def test_train_pong_dqn(self):
    self._test_hparams_set("rlmf_dqn_tiny")


if __name__ == "__main__":
  tf.test.main()
