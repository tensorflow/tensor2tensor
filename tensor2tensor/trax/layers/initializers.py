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

"""Trax initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
from tensor2tensor.trax import backend


def RandomNormalInitializer(stddev=1e-2):
  """An initializer function for random normal coefficients."""
  def Init(shape, rng):
    return (stddev * backend.random.normal(rng, shape)).astype('float32')
  return Init


def GlorotNormalInitializer(out_dim=0, in_dim=1, scale=onp.sqrt(2)):
  """An initializer function for random Glorot-scaled coefficients."""
  def Init(shape, rng):
    fan_in, fan_out = shape[in_dim], shape[out_dim]
    size = onp.prod(onp.delete(shape, [in_dim, out_dim]))
    std = scale / backend.numpy.sqrt((fan_in + fan_out) / 2. * size)
    return (std * backend.random.normal(rng, shape)).astype('float32')
  return Init


def GlorotUniformInitializer(out_dim=0, in_dim=1):
  """An initializer function for random uniform Glorot-scaled coefficients."""
  def Init(shape, rng):
    fan_in, fan_out = shape[in_dim], shape[out_dim]
    std = backend.numpy.sqrt(2.0 / (fan_in + fan_out))
    a = backend.numpy.sqrt(3.0) * std
    return backend.random.uniform(rng, shape, minval=-a, maxval=a)
  return Init
