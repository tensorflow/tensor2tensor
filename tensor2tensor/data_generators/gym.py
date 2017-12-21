# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Data generators for Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

import gym

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import tensorflow as tf



class GymDiscreteProblem(problem.Problem):
  """Gym environment with discrete actions and rewards."""

  def __init__(self, *args, **kwargs):
    super(GymDiscreteProblem, self).__init__(*args, **kwargs)
    self._env = None

  @property
  def env_name(self):
    # This is the name of the Gym environment for this problem.
    raise NotImplementedError()

  @property
  def env(self):
    if self._env is None:
      self._env = gym.make(self.env_name)
    return self._env

  @property
  def num_actions(self):
    raise NotImplementedError()

  @property
  def num_rewards(self):
    raise NotImplementedError()

  @property
  def num_steps(self):
    raise NotImplementedError()

  @property
  def num_shards(self):
    return 10

  @property
  def num_dev_shards(self):
    return 1

  def get_action(self, observation=None):
    return self.env.action_space.sample()

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.input_modality = {"inputs": ("image:identity", 256),
                        "inputs_prev": ("image:identity", 256),
                        "reward": ("symbol:identity", self.num_rewards),
                        "action": ("symbol:identity", self.num_actions)}
    p.target_modality = ("image:identity", 256)
    p.input_space_id = problem.SpaceID.IMAGE
    p.target_space_id = problem.SpaceID.IMAGE

  def generator(self, data_dir, tmp_dir):
    self.env.reset()
    action = self.get_action()
    prev_observation, observation = None, None
    for _ in range(self.num_steps):
      prev_prev_observation = prev_observation
      prev_observation = observation
      observation, reward, done, _ = self.env.step(action)
      action = self.get_action(observation)
      if done:
        self.env.reset()
      def flatten(nparray):
        flat1 = [x for sublist in nparray.tolist() for x in sublist]
        return [x for sublist in flat1 for x in sublist]
      if prev_prev_observation is not None:
        yield {"inputs_prev": flatten(prev_prev_observation),
               "inputs": flatten(prev_observation),
               "action": [action],
               "done": [done],
               "reward": [reward],
               "targets": flatten(observation)}

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    all_paths = train_paths + dev_paths
    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir), all_paths)
    generator_utils.shuffle_dataset(all_paths)


@registry.register_problem
class GymPongRandom5k(GymDiscreteProblem):
  """Pong game, random actions."""

  @property
  def env_name(self):
    return "Pong-v0"

  @property
  def num_actions(self):
    return 4

  @property
  def num_rewards(self):
    return 2

  @property
  def num_steps(self):
    return 5000
