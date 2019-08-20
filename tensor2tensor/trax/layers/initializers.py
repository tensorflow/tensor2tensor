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

def _get_fans(shape):
    receptive_field = np.prod(shape[:-2])
    if len(shape) >= 2:
        fan_in, fan_out = shape[-2], shape[-1]
    elif len(shape) == 1:
        fan_in, fan_out = shape[0]
    else:
        fan_in, fan_out = 1.
    fan_in *= receptive_field
    fan_out *= receptive_field
    return fan_in, fan_out


def RandomNormalInitializer(stddev=1e-2):
  """An initializer function for random normal coefficients."""
  def Init(shape, rng):
    return (stddev * backend.random.normal(rng, shape)).astype('float32')
  return Init


def VarianceScalingInitializer(scale, mode, distribution):
    if scale <= 0.:
        raise ValueError(f"scale must be positive float, {scale} given")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
        raise ValueError(f"Invalid mode argument: {mode}, must be either fan_in, fan_out or fan_avg")

    def Init(shape, rng):
        fan_in, fan_out = _get_fans(shape)
        gain = scale
        if mode == "fan_in":
            gain /= fan_in
        elif mode == "fan_out":
            gain /= fan_out
        elif mode == "fan_avg":
            gain /= (fan_in + fan_out) / 2
        if distribution == "truncated_normal":
            # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = backend.numpy.sqrt(gain) / .87962566103423978
            return (backend.random.truncated_normal(rng, -2, 2, shape) * stddev).astype('float32')
        elif distribution == "normal":
            return (backend.random.normal(rng, shape, dtype) * backend.numpy.sqrt(gain)).astype('float32')
        elif distribution == "uniform":
            lim = backend.numpy.sqrt(3. * gain)
            return backend.random.uniform(rng, shape, dtype, minval=-lim, maxval=lim).astype('float32')
        else:
            raise ValueError("invalid distribution for variance scaling Initializer")
    return Init


def GlorotNormalInitializer(out_dim=0, in_dim=1, scale=1.):
    """An initializer function for random Glorot-scaled coefficients."""
    return VarianceScalingInitializer(scale, "fan_avg", "truncated_normal")


def GlorotUniformInitializer(out_dim=0, in_dim=1, scale=1.):
    """An initializer function for random uniform Glorot-scaled coefficients."""
    return VarianceScalingInitializer(scale, "fan_avg", "uniform")    


def LeCunNormalInitializer(out_dim=0, in_dim=1, scale=1.):
    """An initializer function for random LeCun-scaled coefficients."""
    return VarianceScalingInitializer(scale, "fan_in", "truncated_normal")


def LeCunUniformInitializer(out_dim=0, in_dim=1):
    """An initializer function for random uniform LeCun-scaled coefficients."""
    return VarianceScalingInitializer(scale, "fan_in", "uniform")    


def KaimingNormalInitializer(out_dim=0, in_dim=1, param=0.):
    """An initializer function for random Kaiming-scaled coefficients."""
    return VarianceScalingInitializer(2.0 / backend.np.sqrt(1 + param**2),
                                      "fan_in", "truncated_normal")


def KaimingUniformInitializer(out_dim=0, in_dim=1, param=0.):
    """An initializer function for random uniform Kaiming-scaled coefficients."""
    return VarianceScalingInitializer((2.0 / backend.np.sqrt(1 + param**2),
                                      "fan_in", "uniform")
