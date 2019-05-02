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
from tensor2tensor.trax import layers
from tensor2tensor.trax.rlax import ppo
from tensorflow import test


class PpoTrainingLoopTest(test.TestCase):

  def get_wrapped_env(self, name="CartPole-v0", max_episode_steps=2):
    env = gym.make(name)
    # Usually gym envs are wrapped in TimeLimit wrapper.
    env = gym_utils.remove_time_limit_wrapper(env)
    # Limit this to a small number for tests.
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

  def test_training_loop(self):
    env = self.get_wrapped_env("CartPole-v0", 2)
    num_epochs = 2
    batch_size = 2
    # Run the training loop.
    _, rewards, val_losses, ppo_objectives = ppo.training_loop(
        env=env,
        epochs=num_epochs,
        policy_net_fun=functools.partial(
            ppo.policy_net, bottom_layers=[layers.Dense(1)]),
        value_net_fun=functools.partial(
            ppo.value_net, bottom_layers=[layers.Dense(1)]),
        policy_optimizer_fun=ppo.optimizer_fun,
        value_optimizer_fun=ppo.optimizer_fun,
        batch_size=batch_size,
        num_optimizer_steps=1,
        random_seed=0)
    self.assertLen(rewards, num_epochs)
    self.assertLen(val_losses, num_epochs)
    self.assertLen(ppo_objectives, num_epochs)

  def test_training_loop_policy_and_value_function(self):
    env = self.get_wrapped_env("CartPole-v0", 2)
    num_epochs = 2
    batch_size = 2
    # Run the training loop.
    _, rewards, val_losses, ppo_objectives = ppo.training_loop(
        env=env,
        epochs=num_epochs,
        policy_and_value_net_fun=functools.partial(
            ppo.policy_and_value_net, bottom_layers=[layers.Dense(1)]),
        policy_and_value_optimizer_fun=ppo.optimizer_fun,
        batch_size=batch_size,
        num_optimizer_steps=1,
        random_seed=0)
    self.assertLen(rewards, num_epochs)
    self.assertLen(val_losses, num_epochs)
    self.assertLen(ppo_objectives, num_epochs)


if __name__ == "__main__":
  test.main()
