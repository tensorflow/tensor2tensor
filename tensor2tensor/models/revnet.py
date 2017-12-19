# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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


"""Creates a RevNet with the bottleneck residual function.

Implements the following equations described in the RevNet paper:
y1 = x1 + f(x2)
y2 = x2 + g(y1)

However, in practice, the authors use the following equations to downsample
tensors inside a RevNet block:

y1 = h(x1) + f(x2)
y2 = h(x2) + g(y1)

In this case, h is the downsampling function used to change number of channels.

These modified equations are evident in the authors' code online:
https://github.com/renmengye/revnet-public

For reference, the original paper can be found here:
https://arxiv.org/pdf/1707.04585.pdf
"""

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import rev_block
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

CONFIG = {'2d': {'conv': tf.layers.conv2d,
                 'max_pool': tf.layers.max_pooling2d,
                 'avg_pool': tf.layers.average_pooling2d,
                 'split_axis': 3,
                 'reduction_dimensions': [1, 2]
                },
          '3d': {'conv': tf.layers.conv3d,
                 'max_pool': tf.layers.max_pooling3d,
                 'avg_pool': tf.layers.average_pooling2d,
                 'split_axis': 4,
                 'reduction_dimensions': [1, 2, 3]
                }
         }


def f(x, depth1, depth2, dim='2d', first_batch_norm=True, layer_stride=1,
      training=True, padding='SAME'):
  """Applies bottleneck residual function for 104-layer RevNet.

  Args:
    x: input tensor
    depth1: Number of output channels for the first and second conv layers.
    depth2: Number of output channels for the third conv layer.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    first_batch_norm: Whether to keep the first batch norm layer or not.
      Typically used in the first RevNet block.
    layer_stride: Stride for the first conv filter. Note that this particular
      104-layer RevNet architecture only varies the stride for the first conv
      filter. The stride for the second conv filter is always set to 1.
    training: True for train phase, False for eval phase.
    padding: Padding for each conv layer.

  Returns:
    Output tensor after applying residual function for 104-layer RevNet.
  """
  conv = CONFIG[dim]['conv']
  with tf.variable_scope('f'):
    if first_batch_norm:
      net = tf.layers.batch_normalization(x, training=training)
      net = tf.nn.relu(net)
    else:
      net = x
    net = conv(net, depth1, 1, strides=layer_stride,
               padding=padding, activation=None)

    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = conv(net, depth1, 3, strides=1,
               padding=padding, activation=None)

    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = conv(net, depth2, 1, strides=1,
               padding=padding, activation=None)
    return net


def h(x, output_channels, dim='2d', layer_stride=1, scope='h'):
  """Downsamples 'x' using a 1x1 convolution filter and a chosen stride.

  Args:
    x: input tensor of size [N, H, W, C]
    output_channels: Desired number of output channels.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    layer_stride: What stride to use. Usually 1 or 2.
    scope: Optional variable scope for the h function.

  This function uses a 1x1 convolution filter and a chosen stride to downsample
  the input tensor x.

  Returns:
    A downsampled tensor of size [N, H/2, W/2, output_channels] if layer_stride
    is 2, else returns a tensor of size [N, H, W, output_channels] if
    layer_stride is 1.
  """
  conv = CONFIG[dim]['conv']
  with tf.variable_scope(scope):
    x = conv(x, output_channels, 1, strides=layer_stride, padding='SAME',
             activation=None)
    return x


def init(images, num_channels, dim='2d', training=True, scope='init'):
  """Standard ResNet initial block used as first RevNet block.

  Args:
    images: [N, H, W, 3] tensor of input images to the model.
    num_channels: Output depth of convolutional layer in initial block.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    training: True for train phase, False for eval phase.
    scope: Optional scope for the init block.

  Returns:
    Two [N, H, W, C] output activations from input images.
  """
  conv = CONFIG[dim]['conv']
  pool = CONFIG[dim]['max_pool']
  with tf.variable_scope(scope):
    net = conv(images, num_channels, 7, strides=2,
               padding='SAME', activation=None)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = pool(net, pool_size=3, strides=2)
    x1, x2 = tf.split(net, 2, axis=CONFIG[dim]['split_axis'])
    return x1, x2


def unit(x1, x2, block_num, depth1, depth2, num_layers, dim='2d',
         first_batch_norm=True, stride=1, training=True):
  """Implements bottleneck RevNet unit from authors' RevNet-104 architecture.

  Args:
    x1: [N, H, W, C] tensor of network activations.
    x2: [N, H, W, C] tensor of network activations.
    block_num: integer ID of block
    depth1: First depth in bottleneck residual unit.
    depth2: Second depth in bottleneck residual unit.
    num_layers: Number of layers in the RevNet block.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    first_batch_norm: Whether to keep the first batch norm layer or not.
      Typically used in the first RevNet block.
    stride: Stride for the residual function.
    training: True for train phase, False for eval phase.

  Returns:
    Two [N, H, W, C] output activation tensors.
  """
  scope_name = 'unit_%d' % block_num
  with tf.variable_scope(scope_name):
    # Manual implementation of downsampling
    with tf.variable_scope('downsampling'):
      with tf.variable_scope('x1'):
        hx1 = h(x1, depth2, dim=dim, layer_stride=stride)
        fx2 = f(x2, depth1, depth2, dim=dim, layer_stride=stride,
                first_batch_norm=first_batch_norm, training=training)
        x1 = hx1 + fx2
      with tf.variable_scope('x2'):
        hx2 = h(x2, depth2, dim=dim, layer_stride=stride)
        fx1 = f(x1, depth1, depth2, dim=dim, training=training)
        x2 = hx2 + fx1

    # Full block using memory-efficient rev_block implementation.
    with tf.variable_scope('full_block'):
      residual_func = lambda x: f(x, depth1, depth2, dim=dim, training=training)
      x1, x2 = rev_block.rev_block(x1, x2,
                                   residual_func,
                                   residual_func,
                                   num_layers=num_layers)
      return x1, x2


def final_block(x1, x2, dim='2d', training=True, scope='final_block'):
  """Converts activations from last RevNet block to pre-logits.

  Args:
    x1: [NxHxWxC] tensor of network activations.
    x2: [NxHxWxC] tensor of network activations.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    training: True for train phase, False for eval phase.
    scope: Optional variable scope for the final block.

  Returns:
    [N, hidden_dim] pre-logits tensor from activations x1 and x2.
  """

  # Final batch norm and relu
  with tf.variable_scope(scope):
    y = tf.concat([x1, x2], axis=CONFIG[dim]['split_axis'])
    y = tf.layers.batch_normalization(y, training=training)
    y = tf.nn.relu(y)

    # Global average pooling
    net = tf.reduce_mean(y, CONFIG[dim]['reduction_dimensions'],
                         name='final_pool', keep_dims=True)

    return net


def revnet104(inputs, hparams, reuse=None):
  """Uses Tensor2Tensor memory optimized RevNet block to build a RevNet.

  Args:
    inputs: [NxHxWx3] tensor of input images to the model.
    hparams: HParams object that contains the following parameters,
      in addition to the parameters contained in the basic_params1() object in
      the common_hparams module:
        num_channels_first - A Python list where each element represents the
          depth of the first and third convolutional layers in the bottleneck
          residual unit for a given block.
        num_channels_second - A Python list where each element represents the
          depth of the second convolutional layer in the bottleneck residual
          unit for a given block.
        num_layers_per_block - A Python list containing the number of RevNet
          layers for each block.
        first_batch_norm - A Python list containing booleans representing the
          presence of a batch norm layer at the beginning of a given block.
        strides - A Python list containing integers representing the stride of
          the residual function for each block.
        num_channels_init_block - An integer representing the number of channels
          for the convolutional layer in the initial block.
        dimension - A string (either "2d" or "3d") that decides if the RevNet is
          2-dimensional or 3-dimensional.
    reuse: Whether to reuse the default variable scope.

  Returns:
    [batch_size, hidden_dim] pre-logits tensor from the bottleneck RevNet.
  """
  training = hparams.mode == tf.estimator.ModeKeys.TRAIN
  with tf.variable_scope('RevNet104', reuse=reuse):
    x1, x2 = init(inputs,
                  num_channels=hparams.num_channels_init_block,
                  dim=hparams.dim,
                  training=training)
    for block_num in range(1, len(hparams.num_layers_per_block)):
      block = {'depth1': hparams.num_channels_first[block_num],
               'depth2': hparams.num_channels_second[block_num],
               'num_layers': hparams.num_layers_per_block[block_num],
               'first_batch_norm': hparams.first_batch_norm[block_num],
               'stride': hparams.strides[block_num]}
      x1, x2 = unit(x1, x2, block_num, dim=hparams.dim, training=training,
                    **block)
    pre_logits = final_block(x1, x2, dim=hparams.dim, training=training)
    return pre_logits


@registry.register_model
class Revnet104(t2t_model.T2TModel):

  def body(self, features):
    return revnet104(features['inputs'], self.hparams)


@registry.register_hparams
def revnet_base():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.add_hparam('num_channels_first', [64, 128, 256, 416])
  hparams.add_hparam('num_channels_second', [256, 512, 1024, 1664])
  hparams.add_hparam('num_layers_per_block', [1, 1, 10, 1])
  hparams.add_hparam('first_batch_norm', [False, True, True, True])
  hparams.add_hparam('strides', [1, 2, 2, 2])
  hparams.add_hparam('num_channels_init_block', 32)
  hparams.add_hparam('dim', '2d')

  hparams.optimizer = 'Momentum'
  hparams.learning_rate = 0.01
  hparams.weight_decay = 1e-4
  # Can run with a batch size of 128 with Problem ImageImagenet224
  hparams.tpu_batch_size_per_shard = 128
  return hparams
