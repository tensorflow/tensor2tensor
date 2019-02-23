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

"""J2J models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

from jax.experimental import stax


def ConvBlock(kernel_size, filters, strides):
  """ResNet convolutional striding block."""
  ks = kernel_size
  filters1, filters2, filters3 = filters
  main = stax.serial(
      stax.Conv(filters1, (1, 1), strides),
      stax.BatchNorm(), stax.Relu,
      stax.Conv(filters2, (ks, ks), padding='SAME'),
      stax.BatchNorm(), stax.Relu,
      stax.Conv(filters3, (1, 1)), stax.BatchNorm())
  shortcut = stax.serial(
      stax.Conv(filters3, (1, 1), strides),
      stax.BatchNorm())
  return stax.serial(
      stax.FanOut(2),
      stax.parallel(main, shortcut),
      stax.FanInSum, stax.Relu)


def IdentityBlock(kernel_size, filters):
  """ResNet identical size block."""
  ks = kernel_size
  filters1, filters2 = filters
  def MakeMain(input_shape):
    # the number of output channels depends on the number of input channels
    return stax.serial(
        stax.Conv(filters1, (1, 1)),
        stax.BatchNorm(), stax.Relu,
        stax.Conv(filters2, (ks, ks), padding='SAME'),
        stax.BatchNorm(), stax.Relu,
        stax.Conv(input_shape[3], (1, 1)), stax.BatchNorm())
  main = stax.shape_dependent(MakeMain)
  return stax.serial(
      stax.FanOut(2),
      stax.parallel(main, stax.Identity),
      stax.FanInSum, stax.Relu)


@gin.configurable()
def Resnet50(hidden_size=64, num_output_classes=1001):
  """ResNet.

  Args:
    hidden_size: the size of the first hidden layer (multiplied later).
    num_output_classes: how many classes to distinguish.

  Returns:
    The ResNet model with the given layer and output sizes.
  """
  return stax.serial(
      stax.Conv(hidden_size, (7, 7), (2, 2), 'SAME'),
      stax.BatchNorm(), stax.Relu,
      stax.MaxPool((3, 3), strides=(2, 2)),
      ConvBlock(3, [hidden_size, hidden_size, 4 * hidden_size], (1, 1)),
      IdentityBlock(3, [hidden_size, hidden_size]),
      IdentityBlock(3, [hidden_size, hidden_size]),
      ConvBlock(3, [2 * hidden_size, 2 * hidden_size, 8 * hidden_size], (2, 2)),
      IdentityBlock(3, [2 * hidden_size, 2 * hidden_size]),
      IdentityBlock(3, [2 * hidden_size, 2 * hidden_size]),
      IdentityBlock(3, [2 * hidden_size, 2 * hidden_size]),
      ConvBlock(3, [4 * hidden_size, 4 * hidden_size, 16*hidden_size], (2, 2)),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size]),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size]),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size]),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size]),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size]),
      ConvBlock(3, [8 * hidden_size, 8 * hidden_size, 32*hidden_size], (2, 2)),
      IdentityBlock(3, [8 * hidden_size, 8 * hidden_size]),
      IdentityBlock(3, [8 * hidden_size, 8 * hidden_size]),
      stax.AvgPool((7, 7)), stax.Flatten,
      stax.Dense(num_output_classes), stax.LogSoftmax)
