# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Tests for tensor2tensor.trax.rl.fake_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax.rl.envs import fake_env
from tensorflow import test


class FakeEnvTest(test.TestCase):

  def test_done_action(self):
    env = fake_env.FakeEnv(input_shape=(2, 3),
                           n_actions=10,
                           done_time_step=None,
                           done_action=9)
    env.reset()

    # Actions 0 to 8
    for action in range(9):
      _, reward, done, _ = env.step(action)
      self.assertFalse(done)
      self.assertEqual(-1.0, reward)

    _, reward, done, _ = env.step(9)
    self.assertTrue(done)
    self.assertEqual(1.0, reward)

  def test_done_time_step(self):
    env = fake_env.FakeEnv(input_shape=(2, 3),
                           n_actions=10,
                           done_time_step=10,
                           done_action=None)
    env.reset()

    # Take 10 steps.
    for _ in range(10):
      _, reward, done, _ = env.step(0)
      self.assertFalse(done)
      self.assertEqual(-1.0, reward)

    # Take final time-step, this is the time-step numbered 10 since time-steps
    # are 0 indexed.
    _, reward, done, _ = env.step(0)
    self.assertTrue(done)
    self.assertEqual(1.0, reward)

if __name__ == '__main__':
  test.main()
