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

"""Example registrations for T2T."""
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry


@registry.register_hparams
def my_very_own_hparams():
  # Start with the base set
  hp = common_hparams.basic_params1()
  # Modify existing hparams
  hp.num_hidden_layers = 2
  # Add new hparams
  hp.add_hparam("filter_size", 2048)
  return hp

# Use register_model for a new T2TModel
# Use register_problem for a new Problem
