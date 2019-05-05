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

"""gin-configurable optimizers and learning rate functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

import jax
from jax.experimental import optimizers as opt
import numpy as onp


def opt_configure(*args, **kwargs):
  kwargs["module"] = "trax.optimizers"
  return gin.external_configurable(*args, **kwargs)

# Optimizers
sgd = opt_configure(opt.sgd)
adam = opt_configure(opt.adam)
momentum = opt_configure(opt.momentum)
rmsprop = opt_configure(opt.rmsprop)

# Learning rates
constant = opt_configure(opt.constant)
exponential_decay = opt_configure(opt.exponential_decay)
inverse_time_decay = opt_configure(opt.inverse_time_decay)
piecewise_constant = opt_configure(opt.piecewise_constant)


def parallelize(opt_maker):
  """Transform an optimizer maker into a parallel one with replicated params."""

  def parallel_opt_maker(*args, **kwargs):  # pylint:disable=missing-docstring
    init_fun, update_fun, get_params = opt_maker(*args, **kwargs)

    num_devices = jax.lib.xla_bridge.device_count()
    replicate_array = lambda x: onp.broadcast_to(x, (num_devices,) + x.shape)
    unreplicate_array = lambda x: x.mean(0)  # an alternative is x[0]

    def init_replicated(params):
      if num_devices > 1:
        params = jax.tree_util.tree_map(replicate_array, params)
      return init_fun(params)

    def get_params_unreplicated(opt_state):
      params = get_params(opt_state)
      if num_devices > 1:
        params = jax.tree_util.tree_map(unreplicate_array, params)
      return params

    return init_replicated, update_fun, get_params, get_params_unreplicated
  return parallel_opt_maker
