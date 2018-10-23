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

"""SimulatedBatchEnv in a Gym-like interface."""

#TODO(pm): do we really need these
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
import tensorflow as tf

from gym.spaces import Box
from gym.spaces import Discrete
from tensor2tensor.rl.envs.simulated_batch_env import SimulatedBatchEnv


class FlatBatchEnv:
  def __init__(self, batch_env):
    if batch_env.batch_size != 1:
      raise ValueError("Number of environments in batch must be equal to one")
    self.batch_env = batch_env
    self.action_space = self.batch_env.action_space
    # TODO(KC): replace this when removing _augment_observation()
    # self.observation_space = self.batch_env.observation_space
    self.observation_space = Box(low=0, high=255, shape=(84, 84, 3),
                                 dtype=np.uint8)
    #TODO(pm): make a sepearte wrapper to handle this
    self.game_over = False  # Dopamine needs it?

  def step(self, action):
    obs, rewards, dones = self.batch_env.step([action])
    return self._augment_observation(obs[0]), rewards[0], dones[0], {}

  def reset(self):
    ob = self.batch_env.reset()[0]
    return self._augment_observation(ob)

  def _augment_observation(self, ob):
    # TODO(KC): remove this
    dopamine_ob = np.zeros(shape=(84, 84, 3),
                         dtype=np.uint8)
    dopamine_ob[:80, :80, :] = ob[:80, :80, :]
    return dopamine_ob


class SimulatedBatchGymEnv:
  """
  SimulatedBatchEnv in a Gym-like interface.
  
  The environments are  batched.
  """

  def __init__(self, environment_spec, batch_size, timesteps_limit=100, sess=None):
    self.batch_size = batch_size
    self.timesteps_limit = timesteps_limit

    self.action_space = Discrete(2)
    # TODO: check sizes
    # self.observation_space = self._batch_env.observ_space
    self.observation_space = Box(
        low=0, high=255, shape=(84, 84, 3),
        dtype=np.uint8)
    self.res = None
    self.game_over = False

    with tf.Graph().as_default():
      self._batch_env = SimulatedBatchEnv(environment_spec,
                                          self.batch_size)

      self.action_space = self._batch_env.action_space

      self._sess = sess if sess is not None else tf.Session()
      self._actions_t = tf.placeholder(shape=(1,), dtype=tf.int32)
      self._rewards_t, self._dones_t = self._batch_env.simulate(self._actions_t)
      self._obs_t = self._batch_env.observ
      self._reset_op = self._batch_env.reset(tf.constant([0], dtype=tf.int32))

      environment_wrappers = environment_spec.wrappers
      wrappers = copy.copy(environment_wrappers) if environment_wrappers else []

      self._to_initialize = [self._batch_env]
      for w in wrappers:
        self._batch_env = w[0](self._batch_env, **w[1])
        self._to_initialize.append(self._batch_env)

      self._sess_initialized = False
      self._step_num = 0

  def _initialize_sess(self):
    self._sess.run(tf.global_variables_initializer())
    for _batch_env in self._to_initialize:
      _batch_env.initialize(self._sess)
    self._sess_initialized = True

  def render(self, mode="human"):
    raise NotImplementedError()

  def reset(self, indicies=None):
    if indicies:
      raise NotImplementedError()
    if not self._sess_initialized:
      self._initialize_sess()
    obs = self._sess.run(self._reset_op)
    # TODO(pmilos): remove if possible
    obs[:, 0, 0, 0] = 0
    obs[:, 0, 0, 1] = 255
    return obs

  def step(self, actions):
    self._step_num += 1
    obs, rewards, dones = self._sess.run(
      [self._obs_t, self._rewards_t, self._dones_t],
      feed_dict={self._actions_t: [actions]})

    if self._step_num >= 100:
      dones = [True] * self.batch_size

    return obs, rewards, dones
