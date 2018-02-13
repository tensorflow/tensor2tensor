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

import functools

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import rev_block
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

conv_initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=2.0, mode='FAN_OUT')

CONFIG = {'2d': {'conv': functools.partial(
    tf.layers.conv2d, kernel_initializer=conv_initializer),
                 'max_pool': tf.layers.max_pooling2d,
                 'avg_pool': tf.layers.average_pooling2d,
                 'split_axis': 3,
                 'reduction_dimensions': [1, 2]
                },
          '3d': {'conv': functools.partial(
              tf.layers.conv3d, kernel_initializer=conv_initializer),
                 'max_pool': tf.layers.max_pooling3d,
                 'avg_pool': tf.layers.average_pooling2d,
                 'split_axis': 4,
                 'reduction_dimensions': [1, 2, 3]
                }
         }


def f(x, depth1, depth2, dim='2d', first_batch_norm=True, stride=1,
      training=True, bottleneck=True, padding='SAME'):
  """Applies residual function for RevNet.

  Args:
    x: input tensor
    depth1: Number of output channels for the first and second conv layers.
    depth2: Number of output channels for the third conv layer.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    first_batch_norm: Whether to keep the first batch norm layer or not.
      Typically used in the first RevNet block.
    stride: Stride for the first conv filter. Note that this particular
      RevNet architecture only varies the stride for the first conv
      filter. The stride for the second conv filter is always set to 1.
    training: True for train phase, False for eval phase.
    bottleneck: If true, apply bottleneck 1x1 down/up sampling.
    padding: Padding for each conv layer.

  Returns:
    Output tensor after applying residual function for RevNet.
  """
  conv = CONFIG[dim]['conv']
  with tf.variable_scope('f'):
    if first_batch_norm:
      net = tf.layers.batch_normalization(x, training=training)
      net = tf.nn.relu(net)
    else:
      net = x

    if bottleneck:
      net = conv(net, depth1, 1, strides=stride,
                 padding=padding, activation=None)

      net = tf.layers.batch_normalization(net, training=training)
      net = tf.nn.relu(net)
      net = conv(net, depth1, 3, strides=1,
                 padding=padding, activation=None)

      net = tf.layers.batch_normalization(net, training=training)
      net = tf.nn.relu(net)
      net = conv(net, depth2, 1, strides=1,
                 padding=padding, activation=None)
    else:
      net = conv(net, depth2, 3, strides=stride,
                 padding=padding, activation=None)
      net = tf.layers.batch_normalization(x, training=training)
      net = tf.nn.relu(net)
      net = conv(net, depth2, 3, strides=stride,
                 padding=padding, activation=None)

    return net


def downsample_bottleneck(x, output_channels, dim='2d', stride=1, scope='h'):
  """Downsamples 'x' by `stride` using a 1x1 convolution filter.

  Args:
    x: input tensor of size [N, H, W, C]
    output_channels: Desired number of output channels.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    stride: What stride to use. Usually 1 or 2.
    scope: Optional variable scope.

  Returns:
    A downsampled tensor of size [N, H/2, W/2, output_channels] if stride
    is 2, else returns a tensor of size [N, H, W, output_channels] if
    stride is 1.
  """
  conv = CONFIG[dim]['conv']
  with tf.variable_scope(scope):
    x = conv(x, output_channels, 1, strides=stride, padding='SAME',
             activation=None)
    return x


def downsample_residual(x, output_channels, dim='2d', stride=1, scope='h'):
  """Downsamples 'x' by `stride` using average pooling.

  Args:
    x: input tensor of size [N, H, W, C]
    output_channels: Desired number of output channels.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    stride: What stride to use. Usually 1 or 2.
    scope: Optional variable scope.

  Returns:
    A downsampled tensor of size [N, H/2, W/2, output_channels] if stride
    is 2, else returns a tensor of size [N, H, W, output_channels] if
    stride is 1.
  """
  with tf.variable_scope(scope):
    if stride > 1:
      avg_pool = CONFIG[dim]['avg_pool']
      x = avg_pool(x,
                   pool_size=(stride, stride),
                   strides=(stride, stride),
                   padding='VALID')

    input_channels = tf.shape(x)[3]
    diff = output_channels - input_channels
    x = tf.pad(
        x, [[0, 0], [0, 0], [0, 0],
            [diff // 2, diff // 2]])
    return x


def init(images, num_channels, dim='2d', stride=2,
         kernel_size=7, maxpool=True, training=True, scope='init'):
  """Standard ResNet initial block used as first RevNet block.

  Args:
    images: [N, H, W, 3] tensor of input images to the model.
    num_channels: Output depth of convolutional layer in initial block.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    stride: stride for the convolution and pool layer.
    kernel_size: Size of the initial convolution filter
    maxpool: If true, apply a maxpool after the convolution
    training: True for train phase, False for eval phase.
    scope: Optional scope for the init block.

  Returns:
    Two [N, H, W, C] output activations from input images.
  """
  conv = CONFIG[dim]['conv']
  pool = CONFIG[dim]['max_pool']
  with tf.variable_scope(scope):
    net = conv(images, num_channels, kernel_size, strides=stride,
               padding='SAME', activation=None)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    if maxpool:
      net = pool(net, pool_size=3, strides=stride)
    x1, x2 = tf.split(net, 2, axis=CONFIG[dim]['split_axis'])
    return x1, x2


def unit(x1, x2, block_num, depth, num_layers, dim='2d',
         bottleneck=True, first_batch_norm=True, stride=1, training=True):
  """Implements bottleneck RevNet unit from authors' RevNet architecture.

  Args:
    x1: [N, H, W, C] tensor of network activations.
    x2: [N, H, W, C] tensor of network activations.
    block_num: integer ID of block
    depth: First depth in bottleneck residual unit.
    num_layers: Number of layers in the RevNet block.
    dim: '2d' if 2-dimensional, '3d' if 3-dimensional.
    bottleneck: Should a bottleneck layer be used.
    first_batch_norm: Whether to keep the first batch norm layer or not.
      Typically used in the first RevNet block.
    stride: Stride for the residual function.
    training: True for train phase, False for eval phase.

  Returns:
    Two [N, H, W, C] output activation tensors.
  """
  scope_name = 'unit_%d' % block_num
  if bottleneck:
    depth1 = depth
    depth2 = depth * 4
  else:
    depth1 = depth2 = depth

  residual = functools.partial(f,
                               depth1=depth1, depth2=depth2, dim=dim,
                               training=training, bottleneck=bottleneck)

  with tf.variable_scope(scope_name):
    downsample = downsample_bottleneck if bottleneck else downsample_residual
    # Manual implementation of downsampling
    with tf.variable_scope('downsampling'):
      with tf.variable_scope('x1'):
        hx1 = downsample(x1, depth2, dim=dim, stride=stride)
        fx2 = residual(x2, stride=stride, first_batch_norm=first_batch_norm)
        x1 = hx1 + fx2
      with tf.variable_scope('x2'):
        hx2 = downsample(x2, depth2, dim=dim, stride=stride)
        fx1 = residual(x1)
        x2 = hx2 + fx1

    # Full block using memory-efficient rev_block implementation.
    with tf.variable_scope('full_block'):
      x1, x2 = rev_block.rev_block(x1, x2,
                                   residual,
                                   residual,
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


def revnet(inputs, hparams, reuse=None):
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
  with tf.variable_scope('RevNet', reuse=reuse):
    x1, x2 = init(inputs,
                  num_channels=hparams.num_channels_init_block,
                  dim=hparams.dim,
                  kernel_size=hparams.init_kernel_size,
                  maxpool=hparams.init_maxpool,
                  stride=hparams.init_stride,
                  training=training)
    for block_num in range(len(hparams.num_layers_per_block)):
      block = {'depth': hparams.num_channels[block_num],
               'num_layers': hparams.num_layers_per_block[block_num],
               'first_batch_norm': hparams.first_batch_norm[block_num],
               'stride': hparams.strides[block_num],
               'bottleneck': hparams.bottleneck}
      x1, x2 = unit(x1, x2, block_num, dim=hparams.dim, training=training,
                    **block)
    pre_logits = final_block(x1, x2, dim=hparams.dim, training=training)
    return pre_logits


@registry.register_model
class Revnet(t2t_model.T2TModel):

  def body(self, features):
    return revnet(features['inputs'], self.hparams)


def revnet_base():
  """Default hparams for Revnet."""
  hparams = common_hparams.basic_params1()
  hparams.add_hparam('num_channels', [64, 128, 256, 416])
  hparams.add_hparam('num_layers_per_block', [1, 1, 10, 1])
  hparams.add_hparam('bottleneck', True)
  hparams.add_hparam('first_batch_norm', [False, True, True, True])
  hparams.add_hparam('init_stride', 2)
  hparams.add_hparam('init_kernel_size', 7)
  hparams.add_hparam('init_maxpool', True)
  hparams.add_hparam('strides', [1, 2, 2, 2])
  hparams.add_hparam('num_channels_init_block', 64)
  hparams.add_hparam('dim', '2d')

  # Variable init
  hparams.initializer = 'normal_unit_scaling'
  hparams.initializer_gain = 2.

  # Optimization
  hparams.optimizer = 'Momentum'
  hparams.optimizer_momentum_momentum = 0.9
  hparams.optimizer_momentum_nesterov = True
  hparams.weight_decay = 1e-4
  hparams.clip_grad_norm = 0.0
  # (base_lr=0.1) * (batch_size=128*8 (on TPU, or 8 GPUs)=1024) / (256.)
  hparams.learning_rate = 0.4
  hparams.learning_rate_decay_scheme = 'cosine'
  # For image_imagenet224, 120k training steps, which effectively makes this a
  # cosine decay (i.e. no cycles).
  hparams.learning_rate_cosine_cycle_steps = 120000

  # Can run with a batch size of 128 with Problem ImageImagenet224
  hparams.batch_size = 128
  return hparams


@registry.register_hparams
def revnet_104():
  return revnet_base()


def revnet_cifar_base():
  """Tiny hparams suitable for CIFAR/etc."""
  hparams = revnet_base()
  hparams.num_channels_init_block = 32
  hparams.first_batch_norm = [False, True, True]
  hparams.init_stride = 1
  hparams.init_kernel_size = 3
  hparams.init_maxpool = False
  hparams.strides = [1, 2, 2]
  hparams.batch_size = 128
  hparams.weight_decay = 1e-4

  hparams.learning_rate = 0.1
  hparams.learning_rate_cosine_cycle_steps = 5000
  return hparams


@registry.register_hparams
def revnet_38_cifar():
  hparams = revnet_cifar_base()
  hparams.bottleneck = False
  hparams.num_channels = [16, 32, 56]
  hparams.num_layers_per_block = [2, 2, 2]
  hparams.initializer = 'normal_unit_scaling'
  hparams.initializer_gain = 1.5
  return hparams


@registry.register_hparams
def revnet_110_cifar():
  """Tiny hparams suitable for CIFAR/etc."""
  hparams = revnet_cifar_base()
  hparams.bottleneck = False
  hparams.num_channels = [16, 32, 64]
  hparams.num_layers_per_block = [8, 8, 8]
  return hparams


@registry.register_hparams
def revnet_164_cifar():
  """Tiny hparams suitable for CIFAR/etc."""
  hparams = revnet_cifar_base()
  hparams.bottleneck = True
  hparams.num_channels = [16, 32, 64]
  hparams.num_layers_per_block = [8, 8, 8]
  return hparams


@registry.register_ranged_hparams
def revnet_range(rhp):
  """Hyperparameters for tuning revnet."""
  rhp.set_float('learning_rate', 0.05, 0.2, scale=rhp.LOG_SCALE)
  rhp.set_float('weight_decay', 1e-5, 1e-3, scale=rhp.LOG_SCALE)
  rhp.set_discrete('num_channels_init_block', [64, 128])
  return rhp
