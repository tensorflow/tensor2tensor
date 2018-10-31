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

import copy

from gym import Env
from tensor2tensor.rl.envs.simulated_batch_env import SimulatedBatchEnv
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


class SimulatedBatchGymEnv(Env):
  """SimulatedBatchEnv in a Gym-like interface, environments are  batched."""

  def __init__(self, environment_spec, batch_size,
               model_dir=None, sess=None):
    self.batch_size = batch_size

    with tf.Graph().as_default():
      self._batch_env = SimulatedBatchEnv(environment_spec,
                                          self.batch_size)

      self.action_space = self._batch_env.action_space
      # TODO(kc): check for the stack wrapper and correct number of channels in
      # observation_space
      self.observation_space = self._batch_env.observ_space
      self._sess = sess if sess is not None else tf.Session()
      self._to_initialize = [self._batch_env]

      environment_wrappers = environment_spec.wrappers
      wrappers = copy.copy(environment_wrappers) if environment_wrappers else []

      for w in wrappers:
        self._batch_env = w[0](self._batch_env, **w[1])
        self._to_initialize.append(self._batch_env)

      self._sess.run(tf.global_variables_initializer())
      for wrapped_env in self._to_initialize:
        wrapped_env.initialize(self._sess)

      self._actions_t = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
      self._rewards_t, self._dones_t = self._batch_env.simulate(self._actions_t)
      self._obs_t = self._batch_env.observ
      self._reset_op = self._batch_env.reset(
          tf.range(batch_size, dtype=tf.int32)
      )

      env_model_loader = tf.train.Saver(
          var_list=tf.global_variables(scope="next_frame*"))  # pylint:disable=unexpected-keyword-arg
      trainer_lib.restore_checkpoint(model_dir, saver=env_model_loader,
                                     sess=self._sess, must_restore=True)

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
