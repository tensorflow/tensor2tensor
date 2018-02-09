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

import tensorflow as tf

from tensor2tensor.bin import t2t_rl_trainer


FLAGS = tf.app.flags.FLAGS


class TrainTest(tf.test.TestCase):

  def test_no_crash_pendulum(self):
    params = t2t_rl_trainer.example_params()
    params[2].epochs_num = 10
    FLAGS.event_dir = tf.test.get_temp_dir()
    t2t_rl_trainer.train(params)


if __name__ == '__main__':
  FLAGS.config = 'unused'
  tf.test.main()
