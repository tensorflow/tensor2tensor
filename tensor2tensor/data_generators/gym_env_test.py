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

"""Gym env tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import gym
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from tensor2tensor.data_generators import gym_env

import tensorflow as tf


class TestEnv(gym.Env):
  """Test environment.

  Odd frames are "done".
  """

  action_space = Discrete(1)
  observation_space = Box(
      low=0, high=255, shape=(2, 2, 1), dtype=np.uint8
  )

  def __init__(self):
    self._counter = 0

  def _generate_ob(self):
    return np.zeros(
        self.observation_space.shape, self.observation_space.dtype
    )

  def step(self, action):
    done = self._counter % 2 == 1
    self._counter += 1
    return (self._generate_ob(), 0, done, {})

  def reset(self):
    return self._generate_ob()


class GymEnvTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.out_dir = tf.test.get_temp_dir()
    shutil.rmtree(cls.out_dir)
    os.mkdir(cls.out_dir)

  def test_generates(self):
    env = gym_env.T2TGymEnv([TestEnv(), TestEnv()])
    env.reset()
    for _ in range(20):
      (_, _, dones) = env.step([0, 0])
      for (i, done) in enumerate(dones):
        if done:
          env.reset([i])
    env.generate_data(self.out_dir, tmp_dir=None)

    filenames = os.listdir(self.out_dir)
    self.assertTrue(filenames)
    path = os.path.join(self.out_dir, filenames[0])
    records = list(tf.python_io.tf_record_iterator(path))
    self.assertTrue(records)


if __name__ == "__main__":
  tf.test.main()
