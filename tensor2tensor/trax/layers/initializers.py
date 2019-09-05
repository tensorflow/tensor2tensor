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


def _GetFans(shape, out_dim=-1, in_dim=-2):
  """Get the fan-in and fan-out sizes for the given shape and dims."""
  # Temporary fix until numpy.delete supports negative indices.
  if out_dim < 0:
    out_dim += len(shape)
  if in_dim < 0:
    in_dim += len(shape)

  receptive_field = backend.numpy.prod(onp.delete(shape, [in_dim, out_dim]))
  if len(shape) >= 2:
    fan_in, fan_out = shape[in_dim], shape[out_dim]
  elif len(shape) == 1:
    fan_in = shape[0]
    fan_out = shape[0]
  else:
    fan_in = 1.
    fan_out = 1.
    fan_in *= receptive_field
    fan_out *= receptive_field
  return fan_in, fan_out


def RandomNormalInitializer(stddev=1e-2):
  """An initializer function for random normal coefficients."""

  def Init(shape, rng):
    return (stddev * backend.random.normal(rng, shape)).astype('float32')

  return Init


def RandomUniformInitializer(lim=1.0):
  """An initializer function for random uniform coefficients."""

  def Init(shape, rng):
    return (backend.random.uniform(rng, shape, backend.numpy.float32, -lim,
                                   lim))

  return Init


def VarianceScalingInitializer(out_dim, in_dim, scale, mode, distribution):
  """Initializer capable of adapting its scale to the shape of weights."""
  if scale <= 0.:
    raise ValueError('scale must be positive float, {} given'.format(scale))
  if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
    raise ValueError(
        'Invalid mode argument:, {}, must be either fan_in, fan_out or fan_avg'
        .format(mode))

  def Init(shape, rng):
    """The initializer function."""
    fan_in, fan_out = _GetFans(shape, out_dim, in_dim)
    gain = scale
    if mode == 'fan_in':
      gain /= fan_in
    elif mode == 'fan_out':
      gain /= fan_out
    elif mode == 'fan_avg':
      gain /= (fan_in + fan_out) / 2
    if distribution == 'truncated_normal':
      # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = backend.numpy.sqrt(gain) / .87962566103423978
      return (backend.random.truncated_normal(rng, -2, 2, shape) *
              stddev).astype('float32')
    elif distribution == 'normal':
      return (backend.random.normal(rng, shape) *
              backend.numpy.sqrt(gain)).astype('float32')
    elif distribution == 'uniform':
      lim = backend.numpy.sqrt(3. * gain)
      return (backend.random.uniform(rng, shape, backend.numpy.float32, -lim,
                                     lim))
    else:
      raise ValueError('invalid distribution for variance scaling Initializer')

  return Init


def GlorotNormalInitializer(out_dim=-1, in_dim=-2, scale=1.):
  """An initializer function for random Glorot-scaled coefficients."""
  return VarianceScalingInitializer(out_dim, in_dim, scale, 'fan_avg', 'normal')


def GlorotUniformInitializer(out_dim=-1, in_dim=-2, scale=1.):
  """An initializer function for random uniform Glorot-scaled coefficients."""
  return VarianceScalingInitializer(out_dim, in_dim, scale, 'fan_avg',
                                    'uniform')


def LeCunNormalInitializer(out_dim=-1, in_dim=-2, scale=1.):
  """An initializer function for random LeCun-scaled coefficients."""
  return VarianceScalingInitializer(out_dim, in_dim, scale, 'fan_in', 'normal')


def LeCunUniformInitializer(out_dim=-1, in_dim=-2, scale=1.):
  """An initializer function for random uniform LeCun-scaled coefficients."""
  return VarianceScalingInitializer(out_dim, in_dim, scale, 'fan_in', 'uniform')


def KaimingNormalInitializer(out_dim=-1, in_dim=-2, param=0.):
  """An initializer function for random Kaiming-scaled coefficients."""
  return VarianceScalingInitializer(out_dim, in_dim,
                                    2.0 / backend.numpy.sqrt(1 + param**2),
                                    'fan_in', 'normal')


def KaimingUniformInitializer(out_dim=-1, in_dim=-2, param=0.):
  """An initializer function for random uniform Kaiming-scaled coefficients."""
  return VarianceScalingInitializer(out_dim, in_dim,
                                    2.0 / backend.numpy.sqrt(1 + param**2),
                                    'fan_in', 'uniform')
