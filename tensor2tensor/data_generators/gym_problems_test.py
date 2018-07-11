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
"""Gym generators tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from tensor2tensor.data_generators import gym_problems

import tensorflow as tf


class GymProblemsTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tmp_dir = tf.test.get_temp_dir()
    shutil.rmtree(cls.tmp_dir)
    os.mkdir(cls.tmp_dir)

  def testGymAtariBoots(self):
    problem = gym_problems.GymPongRandom()
    self.assertEqual(210, problem.frame_height)


if __name__ == "__main__":
  tf.test.main()
