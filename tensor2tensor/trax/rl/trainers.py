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

"""Trainers defined in trax.rl."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from tensor2tensor.trax.rl import ppo_trainer
from tensor2tensor.trax.rl import simple_trainer


# Ginify
def trainer_configure(*args, **kwargs):
  kwargs["module"] = "trax.rl.trainers"
  kwargs["blacklist"] = ["train_env", "eval_env", "output_dir"]
  return gin.external_configurable(*args, **kwargs)


# pylint: disable=invalid-name
PPO = trainer_configure(ppo_trainer.PPO)
SimPLe = trainer_configure(simple_trainer.SimPLe)
