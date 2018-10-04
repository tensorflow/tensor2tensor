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
"""Common utilities for mesh tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import heapq

import tensorflow as tf
from tensorflow.python.framework import ops


@contextlib.contextmanager
def outside_all_rewrites():
  with ops.control_dependencies(None):
    yield


class BalancedVariablePlacer(object):
  """Place the variable on different device and blance the memory usage."""

  def __init__(self, devices, init_usage=None):
    init_usage = init_usage if init_usage else [0] * len(devices)
    assert len(devices) == len(init_usage)
    self._mem_device_heap = list(zip(init_usage, devices))
    heapq.heapify(self._mem_device_heap)
    self._last_device = devices[0]

  def device_function(self, var):
    """Choose a device for the input variable.

    Args:
      var: an Variable.

    Returns:
      The device for placing the var.
    """
    if var.type not in ('Variable', 'VariableV2', 'VarHandleOp'):
      tf.logging.info('Place {} on last device: {}.'.format(
          var.name, self._last_device))
      return self._last_device

    shape = tf.TensorShape(var.get_attr('shape'))
    assert shape.num_elements() is not None

    size = tf.DType(var.get_attr('dtype')).size
    mem, device = heapq.heappop(self._mem_device_heap)
    mem += shape.num_elements() * size
    heapq.heappush(self._mem_device_heap, (mem, device))
    tf.logging.info('Place variable {} on {} and consumes {} Bytes.'.format(
        var.name, device, mem))
    self._last_device = device

    return device
