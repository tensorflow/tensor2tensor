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
"""Tests of basic flow of collecting trajectories and training PPO."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models.research.rl import feed_forward_cnn_small_categorical_fun
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.utils import registry  # pylint: disable=unused-import
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


class TrainTest(tf.test.TestCase):

  test_config = ("epochs_num=4,eval_every_epochs=3,video_during_eval=False,"
                 "num_agents=5,optimization_epochs=5,epoch_length=50")

  def test_no_crash_pendulum(self):
    hparams = trainer_lib.create_hparams(
        "ppo_continuous_action_base",
        TrainTest.test_config)

    rl_trainer_lib.train(hparams)

  def test_no_crash_cartpole(self):
    hparams = trainer_lib.create_hparams(
        "ppo_discrete_action_base",
        TrainTest.test_config)

    rl_trainer_lib.train(hparams)

  # This test should successfully train pong.
  # It should get train mean_score around 0 after 200 epoch
  # By default the test is disabled to avoid travis timeouts
  def test_train_pong(self):
    hparams = trainer_lib.create_hparams(
        "ppo_pong_base",
        TrainTest.test_config)
    hparams.epochs_num = 300
    hparams.eval_every_epochs = 10
    hparams.num_agents = 10
    hparams.optimization_epochs = 3
    hparams.epoch_length = 200
    hparams.entropy_loss_coef = 0.003
    hparams.learning_rate = 8e-05
    hparams.optimizer = "Adam"
    hparams.policy_network = feed_forward_cnn_small_categorical_fun
    hparams.gae_lambda = 0.985
    hparams.num_eval_agents = 1
    hparams.max_gradients_norm = 0.5
    hparams.gae_gamma = 0.985
    hparams.optimization_batch_size = 4
    hparams.clipping_coef = 0.2
    hparams.value_loss_coef = 1

    # TODO(lukaszkaiser): enable tests with Atari.
    # rl_trainer_lib.train(hparams)


if __name__ == "__main__":
  tf.test.main()
