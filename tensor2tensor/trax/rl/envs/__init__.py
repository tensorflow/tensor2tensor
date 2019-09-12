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

"""Environments defined in RL."""

import gin
from gym.envs.registration import register

from tensor2tensor.trax.rl.envs import online_tune_env


# Ginify and register in gym.
def configure_and_register_env(env_class):
  register(
      id="{}-v0".format(env_class.__name__),
      entry_point="tensor2tensor.trax.rl.envs:{}".format(env_class.__name__),
  )
  return gin.external_configurable(env_class, module="trax.rl.envs")


# pylint: disable=invalid-name
OnlineTuneEnv = configure_and_register_env(online_tune_env.OnlineTuneEnv)
