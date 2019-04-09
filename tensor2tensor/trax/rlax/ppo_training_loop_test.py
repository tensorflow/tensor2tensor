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

"""Tests for tensor2tensor.trax.rlax.ppo's training_loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import gym
from tensor2tensor.rl import gym_utils
from tensor2tensor.trax.rlax import ppo
from tensor2tensor.trax.stax import stax_base as stax
from tensorflow import test


class PpoTrainingLoopTest(test.TestCase):

  def test_training_loop(self):
    env = gym.make("CartPole-v0")
    # Usually gym envs are wrapped in TimeLimit wrapper.
    env = gym_utils.remove_time_limit_wrapper(env)
    # Limit this to a small number for tests.
    env = gym.wrappers.TimeLimit(env, max_episode_steps=2)
    num_epochs = 2
    batch_size = 2
    # Common bottom layer(s).
    bottom_layers = [stax.Dense(1)]
    # Run the training loop.
    _, rewards, val_losses, ppo_objectives = ppo.training_loop(
        env=env,
        epochs=num_epochs,
        policy_net_fun=functools.partial(
            ppo.policy_net, bottom_layers=bottom_layers),
        value_net_fun=functools.partial(
            ppo.value_net, bottom_layers=bottom_layers),
        batch_size=batch_size,
        num_optimizer_steps=1,
        random_seed=0)
    self.assertLen(rewards, num_epochs)
    self.assertLen(val_losses, num_epochs)
    self.assertLen(ppo_objectives, num_epochs)


if __name__ == "__main__":
  test.main()
