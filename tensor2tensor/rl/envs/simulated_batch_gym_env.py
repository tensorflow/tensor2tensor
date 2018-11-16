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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym import Env
from tensor2tensor.rl.envs.simulated_batch_env import SimulatedBatchEnv

import tensorflow as tf


# TODO(koz4k): Unify interfaces of batch envs.
class SimulatedBatchGymEnv(Env):
  """SimulatedBatchEnv in a Gym-like interface, environments are  batched."""

  def __init__(self, *args, **kwargs):
    with tf.Graph().as_default():
      self._batch_env = SimulatedBatchEnv(*args, **kwargs)

      self._actions_t = tf.placeholder(shape=(self.batch_size,), dtype=tf.int32)
      self._rewards_t, self._dones_t = self._batch_env.simulate(self._actions_t)
      self._obs_t = self._batch_env.observ
      self._reset_op = self._batch_env.reset(
          tf.range(self.batch_size, dtype=tf.int32)
      )

      self._sess = tf.Session()
      self._sess.run(tf.global_variables_initializer())
      self._batch_env.initialize(self._sess)

  @property
  def batch_size(self):
    return self._batch_env.batch_size

  @property
  def observation_space(self):
    return self._batch_env.observ_space

  @property
  def action_space(self):
    return self._batch_env.action_space

  def render(self, mode="human"):
    raise NotImplementedError()

  def reset(self, indices=None):
    if indices:
      raise NotImplementedError()
    obs = self._sess.run(self._reset_op)
    # TODO(pmilos): remove if possible
    # obs[:, 0, 0, 0] = 0
    # obs[:, 0, 0, 1] = 255
    return obs

  def step(self, actions):
    obs, rewards, dones = self._sess.run(
        [self._obs_t, self._rewards_t, self._dones_t],
        feed_dict={self._actions_t: actions})
    return obs, rewards, dones

  def close(self):
    self._sess.close()
