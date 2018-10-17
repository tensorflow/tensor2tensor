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

from tensor2tensor.rl.envs import py_func_batch_env
from tensor2tensor.rl.envs import simulated_batch_env


def batch_env_factory(environment_spec, num_agents):
  """Factory of batch envs."""
  if environment_spec.simulated_env:
    cur_batch_env = simulated_batch_env.SimulatedBatchEnv(
        environment_spec, num_agents
    )
  else:
    cur_batch_env = py_func_batch_env.PyFuncBatchEnv(environment_spec.env)
  return cur_batch_env
