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

"""Trax normalization layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base


# Batch normalization.
def BatchNormParams(input_shape, rng, axis=(0, 1, 2),
                    center=True, scale=True, **kwargs):
  """Helper to initialize batch norm params."""
  del rng, kwargs
  axis = (axis,) if np.isscalar(axis) else axis
  shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
  beta = np.zeros(shape, dtype='float32') if center else ()
  gamma = np.ones(shape, dtype='float32') if scale else ()
  return (beta, gamma)


@base.layer(new_parameters=BatchNormParams)
def BatchNorm(x, params, axis=(0, 1, 2), epsilon=1e-5,
              center=True, scale=True, **unused_kwargs):
  """Layer construction function for a batch normalization layer."""
  mean = np.mean(x, axis, keepdims=True)
  # Fast but less numerically-stable variance calculation than np.var.
  m1 = np.mean(x**2, axis, keepdims=True)
  var = m1 - mean**2
  z = (x - mean) / np.sqrt(var + epsilon)

  # Expand the parameters to have the right axes.
  beta, gamma = params
  # TODO(phawkins): np.expand_dims should accept an axis tuple.
  # (https://github.com/numpy/numpy/issues/12290)
  ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x)))
  beta = beta[ed]
  gamma = gamma[ed]

  # Return the z rescaled by the parameters if requested.
  if center and scale:
    return gamma * z + beta
  if center:
    return z + beta
  if scale:
    return gamma * z
  return z


# Layer normalization.
def LayerNormParams(input_shape, rng, epsilon=1e-6):
  """Helper: create layer norm parameters."""
  del rng, epsilon
  features = input_shape[-1]
  scale = np.ones(features)
  bias = np.zeros(features)
  return (scale, bias)


@base.layer(new_parameters=LayerNormParams)
def LayerNorm(x, params, epsilon=1e-6, **unused_kwargs):
  (scale, bias) = params
  mean = np.mean(x, axis=-1, keepdims=True)
  variance = np.mean((x - mean)**2, axis=-1, keepdims=True)
  norm_inputs = (x - mean) / np.sqrt(variance + epsilon)
  return norm_inputs * scale + bias
