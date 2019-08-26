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


class BatchNorm(base.Layer):
  """Batch normalization."""

  def __init__(self, axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True,
               momentum=0.999, mode='train'):
    super(BatchNorm, self).__init__()
    self._axis = axis
    self._epsilon = epsilon
    self._center = center
    self._scale = scale
    self._momentum = momentum
    self._mode = mode

  def new_parameters(self, input_shape, input_dtype, rng):
    """Helper to initialize batch norm params."""
    del input_dtype, rng
    axis = self._axis
    axis = (axis,) if np.isscalar(axis) else axis
    shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
    beta = np.zeros(shape, dtype='float32') if self._center else ()
    gamma = np.ones(shape, dtype='float32') if self._scale else ()
    def get_stats_axis(i, d):
      if i in axis:
        return 1
      else:
        return d
    stats_shape = tuple(get_stats_axis(i, d) for i, d in enumerate(input_shape))
    running_mean = np.zeros(stats_shape, dtype=np.float32)
    running_var = np.ones(stats_shape, dtype=np.float32)
    num_batches = np.zeros((), dtype=np.int32)
    return (beta, gamma), (running_mean, running_var, num_batches)

  def call(self, x, params, state, **unused_kwargs):
    """Layer construction function for a batch normalization layer."""

    running_mean, running_var, num_batches = state

    if self._mode == 'train':
      mean = np.mean(x, self._axis, keepdims=True)
      # Fast but less numerically-stable variance calculation than np.var.
      m1 = np.mean(x**2, self._axis, keepdims=True)
      var = m1 - mean**2
      num_batches = num_batches + 1
      def average(factor, new, old):
        return (factor * old + (1 - factor) * new).astype(old.dtype)
      running_mean = average(self._momentum, mean, running_mean)
      running_var = average(self._momentum, var, running_var)
      state = (running_mean, running_var, num_batches)
    else:
      mean = running_mean
      var = running_var

    z = (x - mean.astype(x.dtype)) / np.sqrt(var +
                                             self._epsilon).astype(x.dtype)

    # Expand the parameters to have the right axes.
    beta, gamma = params
    # TODO(phawkins): np.expand_dims should accept an axis tuple.
    # (https://github.com/numpy/numpy/issues/12290)
    ed = tuple(None if i in self._axis else slice(None)
               for i in range(np.ndim(x)))
    beta = beta[ed]
    gamma = gamma[ed]

    # Return the z rescaled by the parameters if requested.
    if self._center and self._scale:
      output = gamma * z + beta
    elif self._center:
      output = z + beta
    elif self._scale:
      output = gamma * z
    else:
      output = z
    assert output.dtype == x.dtype, ('The dtype of the output (%s) of batch '
                                     'norm is not the same as the input (%s). '
                                     'Batch norm should not change the dtype' %
                                     (output.dtype, x.dtype))
    return output, state


# Layer normalization.
def _layer_norm_params(input_shape, input_dtype, rng, epsilon=1e-6):
  """Helper: create layer norm parameters."""
  del input_dtype, rng, epsilon
  features = input_shape[-1]
  scale = np.ones(features)
  bias = np.zeros(features)
  return (scale, bias)


@base.layer(new_parameters=_layer_norm_params)
def LayerNorm(x, params, epsilon=1e-6, **unused_kwargs):  # pylint: disable=invalid-name
  (scale, bias) = params
  mean = np.mean(x, axis=-1, keepdims=True)
  variance = np.mean((x - mean)**2, axis=-1, keepdims=True)
  norm_inputs = (x - mean) / np.sqrt(variance + epsilon)
  return norm_inputs * scale + bias
