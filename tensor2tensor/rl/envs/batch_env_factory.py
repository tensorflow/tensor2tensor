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

"""Utilities for creating batched environments."""

# The code was based on Danijar Hafner's code from tf.agents:
# https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py
# https://github.com/tensorflow/agents/blob/master/agents/scripts/utility.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import gym_env
from tensor2tensor.rl.envs import py_func_batch_env
from tensor2tensor.rl.envs import simulated_batch_env

import tensorflow as tf


def batch_env_factory(environment_spec, num_agents, initial_frame_chooser=None):
  """Factory of batch envs."""
  # TODO(konradczechowski): this is temporary function handling both old and
  # new pipelines, refactor this when we move to the new pipeline.
  if environment_spec.simulated_env:
    cur_batch_env = _define_simulated_batch_env(
        environment_spec, num_agents, initial_frame_chooser)
  else:
    if 'batch_env' in environment_spec:
      assert not 'env_lambda' in environment_spec, \
          'Environment_spec should contain only one of (env_lambda, batch_env).'
      batch_env = environment_spec.batch_env
      assert batch_env.batch_size == num_agents
    else:
      batch_env = _define_batch_env(environment_spec, num_agents)
    cur_batch_env = py_func_batch_env.PyFuncBatchEnv(batch_env)
  return cur_batch_env


def _define_batch_env(environment_spec, num_agents):
  """Create environments and apply all desired wrappers."""

  with tf.variable_scope("environments"):
    envs = [
        environment_spec.env_lambda()
        for _ in range(num_agents)]
    env = gym_env.T2TGymEnv(envs)
    return env


def _define_simulated_batch_env(environment_spec, num_agents,
                                initial_frame_chooser):
  cur_batch_env = simulated_batch_env.SimulatedBatchEnv(
      environment_spec, num_agents, initial_frame_chooser
  )
  return cur_batch_env
