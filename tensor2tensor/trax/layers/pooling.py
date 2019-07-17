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

from tensor2tensor.trax import backend
from tensor2tensor.trax.layers import base


@base.layer()
def MaxPool(x, params, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return backend.max_pool(x, pool_size=pool_size, strides=strides,
                          padding=padding)


@base.layer()
def SumPool(x, params, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return backend.sum_pool(x, pool_size=pool_size, strides=strides,
                          padding=padding)


@base.layer()
def AvgPool(x, params, pool_size=(2, 2), strides=None, padding='VALID', **kw):
  del params, kw
  return backend.avg_pool(x, pool_size=pool_size, strides=strides,
                          padding=padding)
