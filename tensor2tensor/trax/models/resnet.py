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

"""ResNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.trax import layers as tl


def ConvBlock(kernel_size, filters, strides):
  """ResNet convolutional striding block."""
  # TODO(jonni): Use good defaults so Resnet50 code is cleaner / less redundant.
  ks = kernel_size
  filters1, filters2, filters3 = filters
  main = [
      tl.Conv(filters1, (1, 1), strides),
      tl.BatchNorm(),
      tl.Relu(),
      tl.Conv(filters2, (ks, ks), padding='SAME'),
      tl.BatchNorm(),
      tl.Relu(),
      tl.Conv(filters3, (1, 1)),
      tl.BatchNorm(),
  ]
  shortcut = [
      tl.Conv(filters3, (1, 1), strides),
      tl.BatchNorm(),
  ]
  return [
      tl.Residual(main, shortcut=shortcut),
      tl.Relu(),
  ]


def IdentityBlock(kernel_size, filters):
  """ResNet identical size block."""
  # TODO(jonni): Use good defaults so Resnet50 code is cleaner / less redundant.
  ks = kernel_size
  filters1, filters2, filters3 = filters
  main = [
      tl.Conv(filters1, (1, 1)),
      tl.BatchNorm(),
      tl.Relu(),
      tl.Conv(filters2, (ks, ks), padding='SAME'),
      tl.BatchNorm(),
      tl.Relu(),
      tl.Conv(filters3, (1, 1)),
      tl.BatchNorm(),
  ]
  return [
      tl.Residual(main),
      tl.Relu(),
  ]


def Resnet50(d_hidden=64, n_output_classes=1001, mode='train'):
  """ResNet.

  Args:
    d_hidden: Dimensionality of the first hidden layer (multiplied later).
    n_output_classes: Number of distinct output classes.
    mode: Whether we are training or evaluating or doing inference.

  Returns:
    The list of layers comprising a ResNet model with the given parameters.
  """
  del mode
  return tl.Model(
      tl.ToFloat(),
      tl.Conv(d_hidden, (7, 7), (2, 2), 'SAME'),
      tl.BatchNorm(),
      tl.Relu(),
      tl.MaxPool(pool_size=(3, 3), strides=(2, 2)),
      ConvBlock(3, [d_hidden, d_hidden, 4 * d_hidden], (1, 1)),
      IdentityBlock(3, [d_hidden, d_hidden, 4 * d_hidden]),
      IdentityBlock(3, [d_hidden, d_hidden, 4 * d_hidden]),
      ConvBlock(3, [2 * d_hidden, 2 * d_hidden, 8 * d_hidden], (2, 2)),
      IdentityBlock(3, [2 * d_hidden, 2 * d_hidden, 8 * d_hidden]),
      IdentityBlock(3, [2 * d_hidden, 2 * d_hidden, 8 * d_hidden]),
      IdentityBlock(3, [2 * d_hidden, 2 * d_hidden, 8 * d_hidden]),
      ConvBlock(3, [4 * d_hidden, 4 * d_hidden, 16 * d_hidden], (2, 2)),
      IdentityBlock(3, [4 * d_hidden, 4 * d_hidden, 16 * d_hidden]),
      IdentityBlock(3, [4 * d_hidden, 4 * d_hidden, 16 * d_hidden]),
      IdentityBlock(3, [4 * d_hidden, 4 * d_hidden, 16 * d_hidden]),
      IdentityBlock(3, [4 * d_hidden, 4 * d_hidden, 16 * d_hidden]),
      IdentityBlock(3, [4 * d_hidden, 4 * d_hidden, 16 * d_hidden]),
      ConvBlock(3, [8 * d_hidden, 8 * d_hidden, 32 * d_hidden], (2, 2)),
      IdentityBlock(3, [8 * d_hidden, 8 * d_hidden, 32 * d_hidden]),
      IdentityBlock(3, [8 * d_hidden, 8 * d_hidden, 32 * d_hidden]),
      tl.AvgPool(pool_size=(7, 7)),
      tl.Flatten(),
      tl.Dense(n_output_classes),
      tl.LogSoftmax(),
  )


def WideResnetBlock(channels, strides=(1, 1)):
  """WideResnet convolutional block."""
  return [
      tl.BatchNorm(),
      tl.Relu(),
      tl.Conv(channels, (3, 3), strides, padding='SAME'),
      tl.BatchNorm(),
      tl.Relu(),
      tl.Conv(channels, (3, 3), padding='SAME'),
  ]


def WideResnetGroup(n, channels, strides=(1, 1)):
  shortcut = [
      tl.Conv(channels, (3, 3), strides, padding='SAME'),
  ]
  return [
      tl.Residual(WideResnetBlock(channels, strides), shortcut=shortcut),
      tl.Residual([WideResnetBlock(channels, (1, 1))
                   for _ in range(n - 1)]),
  ]


def WideResnet(n_blocks=3, d_hidden=64, n_output_classes=10,
               mode='train'):
  """WideResnet from https://arxiv.org/pdf/1605.07146.pdf.

  Args:
    n_blocks: int, number of blocks in a group.
    d_hidden: Dimensionality of the first hidden layer (multiplied later).
    n_output_classes: int, number of distinct output classes.
    mode: Whether we are training or evaluating or doing inference.

  Returns:
    The list of layers comprising a WideResnet model with the given parameters.
  """
  del mode
  return tl.Model(
      tl.ToFloat(),
      tl.Conv(d_hidden, (3, 3), padding='SAME'),
      WideResnetGroup(n_blocks, d_hidden),
      WideResnetGroup(n_blocks, d_hidden * 2, (2, 2)),
      WideResnetGroup(n_blocks, d_hidden * 4, (2, 2)),
      tl.BatchNorm(),
      tl.Relu(),
      tl.AvgPool(pool_size=(8, 8)),
      tl.Flatten(),
      tl.Dense(n_output_classes),
      tl.LogSoftmax(),
  )
