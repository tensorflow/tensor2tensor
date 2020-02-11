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


class TrainerModelFreeTicTacToeTest(tf.test.TestCase):

  def test_train_tictactoe(self):
    hparams = registry.hparams("rlmf_tictactoe")
    hparams.batch_size = 2
    hparams.eval_sampling_temps = [0.0, 1.0]
    hparams.add_hparam("ppo_epochs_num", 2)
    hparams.add_hparam("ppo_epoch_length", 3)

    hparams.epochs_num = 100
    hparams.eval_every_epochs = 25

    FLAGS.output_dir = tf.test.get_temp_dir()
    FLAGS.env_problem_name = "tic_tac_toe_env_problem"
    trainer_model_free.train(hparams, FLAGS.output_dir, FLAGS.env_problem_name)


if __name__ == "__main__":
  tf.test.main()
