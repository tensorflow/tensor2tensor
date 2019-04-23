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

from tensor2tensor.trax import layers


def ConvBlock(kernel_size, filters, strides):
  """ResNet convolutional striding block."""
  ks = kernel_size
  filters1, filters2, filters3 = filters
  main = layers.Serial(
      layers.Conv(filters1, (1, 1), strides),
      layers.BatchNorm(), layers.Relu(),
      layers.Conv(filters2, (ks, ks), padding='SAME'),
      layers.BatchNorm(), layers.Relu(),
      layers.Conv(filters3, (1, 1)), layers.BatchNorm())
  shortcut = layers.Serial(
      layers.Conv(filters3, (1, 1), strides),
      layers.BatchNorm())
  return layers.Serial(
      layers.FanOut(),
      layers.Parallel(main, shortcut),
      layers.FanInSum(), layers.Relu())


def IdentityBlock(kernel_size, filters):
  """ResNet identical size block."""
  ks = kernel_size
  filters1, filters2, filters3 = filters
  main = layers.Serial(
      layers.Conv(filters1, (1, 1)),
      layers.BatchNorm(), layers.Relu(),
      layers.Conv(filters2, (ks, ks), padding='SAME'),
      layers.BatchNorm(), layers.Relu(),
      layers.Conv(filters3, (1, 1)), layers.BatchNorm())
  return layers.Serial(
      layers.FanOut(),
      layers.Parallel(main, layers.Identity()),
      layers.FanInSum(), layers.Relu())


def Resnet50(hidden_size=64, num_output_classes=1001, mode='train'):
  """ResNet.

  Args:
    hidden_size: the size of the first hidden layer (multiplied later).
    num_output_classes: how many classes to distinguish.
    mode: whether we are training or evaluating or doing inference.

  Returns:
    The ResNet model with the given layer and output sizes.
  """
  del mode
  return layers.Serial(
      layers.Conv(hidden_size, (7, 7), (2, 2), 'SAME'),
      layers.BatchNorm(), layers.Relu(),
      layers.MaxPool(pool_size=(3, 3), strides=(2, 2)),
      ConvBlock(3, [hidden_size, hidden_size, 4 * hidden_size], (1, 1)),
      IdentityBlock(3, [hidden_size, hidden_size, 4 * hidden_size]),
      IdentityBlock(3, [hidden_size, hidden_size, 4 * hidden_size]),
      ConvBlock(3, [2 * hidden_size, 2 * hidden_size, 8 * hidden_size], (2, 2)),
      IdentityBlock(3, [2 * hidden_size, 2 * hidden_size, 8 * hidden_size]),
      IdentityBlock(3, [2 * hidden_size, 2 * hidden_size, 8 * hidden_size]),
      IdentityBlock(3, [2 * hidden_size, 2 * hidden_size, 8 * hidden_size]),
      ConvBlock(3, [4 * hidden_size, 4 * hidden_size, 16*hidden_size], (2, 2)),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size, 16 * hidden_size]),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size, 16 * hidden_size]),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size, 16 * hidden_size]),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size, 16 * hidden_size]),
      IdentityBlock(3, [4 * hidden_size, 4 * hidden_size, 16 * hidden_size]),
      ConvBlock(3, [8 * hidden_size, 8 * hidden_size, 32*hidden_size], (2, 2)),
      IdentityBlock(3, [8 * hidden_size, 8 * hidden_size, 32 * hidden_size]),
      IdentityBlock(3, [8 * hidden_size, 8 * hidden_size, 32 * hidden_size]),
      layers.AvgPool(pool_size=(7, 7)), layers.Flatten(),
      layers.Dense(num_output_classes), layers.LogSoftmax())


def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
  """WideResnet convolutational block."""
  main = layers.Serial(layers.BatchNorm(), layers.Relu(),
                       layers.Conv(channels, (3, 3), strides, padding='SAME'),
                       layers.BatchNorm(), layers.Relu(),
                       layers.Conv(channels, (3, 3), padding='SAME'))
  shortcut = layers.Identity() if not channel_mismatch else layers.Conv(
      channels, (3, 3), strides, padding='SAME')
  return layers.Serial(
      layers.FanOut(), layers.Parallel(main, shortcut), layers.FanInSum())


def WideResnetGroup(n, channels, strides=(1, 1)):
  blocks = []
  blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
  for _ in range(n - 1):
    blocks += [WideResnetBlock(channels, (1, 1))]
  return layers.Serial(*blocks)


def WideResnet(num_blocks=3, hidden_size=64, num_output_classes=10,
               mode='train'):
  """WideResnet from https://arxiv.org/pdf/1605.07146.pdf.

  Args:
    num_blocks: int, number of blocks in a group.
    hidden_size: the size of the first hidden layer (multiplied later).
    num_output_classes: int, number of classes to distinguish.
    mode: is it training or eval.

  Returns:
    The WideResnet model with given layer and output sizes.
  """
  del mode
  return layers.Serial(
      layers.Conv(hidden_size, (3, 3), padding='SAME'),
      WideResnetGroup(num_blocks, hidden_size),
      WideResnetGroup(num_blocks, hidden_size * 2, (2, 2)),
      WideResnetGroup(num_blocks, hidden_size * 4, (2, 2)), layers.BatchNorm(),
      layers.Relu(), layers.AvgPool(pool_size=(8, 8)), layers.Flatten(),
      layers.Dense(num_output_classes), layers.LogSoftmax())
