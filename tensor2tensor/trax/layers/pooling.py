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

"""Trax pooling layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import lax

from tensor2tensor.trax.backend import numpy as np
from tensor2tensor.trax.layers import base


def PoolingGeneral(inputs, reducer, init_val, rescaler=None,
                   pool_size=(2, 2), strides=None, padding='VALID'):
  """Helper: general pooling computation used in pooling layers later."""
  spatial_strides = strides or (1,) * len(pool_size)
  rescale = rescaler(pool_size, spatial_strides, padding) if rescaler else None
  dims = (1,) + pool_size + (1,)  # NHWC
  strides = (1,) + spatial_strides + (1,)
  out = lax.reduce_window(inputs, init_val, reducer, dims, strides, padding)
  return rescale(out, inputs) if rescale else out


@base.layer()
def MaxPool(x, params, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return PoolingGeneral(x, lax.max, -np.inf, pool_size=pool_size,
                        strides=strides, padding=padding)


@base.layer()
def SumPool(x, params, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return PoolingGeneral(x, lax.add, 0., pool_size=pool_size,
                        strides=strides, padding=padding)


def _normalize_by_window_size(dims, spatial_strides, padding):  # pylint: disable=invalid-name
  def Rescale(outputs, inputs):
    one = np.ones(inputs.shape[1:-1], dtype=inputs.dtype)
    window_sizes = lax.reduce_window(
        one, 0., lax.add, dims, spatial_strides, padding)
    return outputs / window_sizes[..., np.newaxis]
  return Rescale


@base.layer()
def AvgPool(x, params, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return PoolingGeneral(x, lax.add, 0., _normalize_by_window_size,
                        pool_size, strides=strides, padding=padding)
