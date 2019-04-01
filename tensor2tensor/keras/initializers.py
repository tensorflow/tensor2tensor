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

"""Initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import six

from tensor2tensor.keras import constraints
from tensor2tensor.keras import regularizers
import tensorflow as tf
from tensorflow_probability import edward2 as ed


# From `tensorflow/python/ops/init_ops.py`
def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1.
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  if isinstance(fan_in, tf.Dimension):
    fan_in = fan_in.value
  if isinstance(fan_out, tf.Dimension):
    fan_out = fan_out.value
  return fan_in, fan_out


class ScaledNormalStdDev(tf.keras.initializers.VarianceScaling):
  """Initializer capable of adapting its scale to the shape of weights tensors.

  This initializes the standard deviation parameter of a Trainable Normal
  distribution with a scale based on the shape of the weights tensor.
  Additionally, A small amount of noise will be added to break weigh symmetry.

  With `distribution="truncated_normal" or "untruncated_normal"`, the standard
  deviation (after truncation, if used) is `stddev = sqrt(scale / n)`, where n
  is:
    - number of input units in the weight tensor, if mode = "fan_in"
    - number of output units, if mode = "fan_out"
    - average of the numbers of input and output units, if mode = "fan_avg"

  Args:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "truncated_normal", or
      "untruncated_normal".
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
      for behavior.
    dtype: The data type. Only floating point types are supported.

  Raises:
    ValueError: In case of an invalid value for the "scale", mode" or
      "distribution" arguments.
  """

  def __init__(self,
               scale=1.0,
               mode='fan_in',
               distribution='untruncated_normal',
               seed=None,
               dtype=tf.float32):
    distribution = distribution.lower()
    if distribution not in {'truncated_normal', 'untruncated_normal'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    super(ScaledNormalStdDev, self).__init__(scale=scale, mode=mode,
                                             distribution=distribution,
                                             seed=seed, dtype=dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    fan_in, fan_out = _compute_fans(scale_shape)
    if self.mode == 'fan_in':
      scale /= max(1., fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == 'truncated_normal':
      # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
    else:  # self.distribution == 'untruncated_normal':
      stddev = math.sqrt(scale)
    return tf.random.truncated_normal(shape, mean=stddev, stddev=stddev*0.1,
                                      dtype=dtype)


class TrainableNormal(tf.keras.layers.Layer):
  """Random normal op as an initializer with trainable mean and stddev."""

  def __init__(self,
               mean_initializer=tf.keras.initializers.truncated_normal(
                   stddev=1e-5),
               stddev_initializer='scaled_normal_std_dev',
               mean_regularizer=None,
               stddev_regularizer=None,
               mean_constraint=None,
               stddev_constraint='positive',
               seed=None,
               dtype=tf.float32,
               **kwargs):
    """Constructs the initializer."""
    super(TrainableNormal, self).__init__(dtype=dtype, **kwargs)
    self.mean_initializer = get(mean_initializer)
    self.stddev_initializer = get(stddev_initializer)
    self.mean_regularizer = regularizers.get(mean_regularizer)
    self.stddev_regularizer = regularizers.get(stddev_regularizer)
    self.mean_constraint = constraints.get(mean_constraint)
    self.stddev_constraint = constraints.get(stddev_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.mean = self.add_weight(
        'mean',
        shape=shape,
        initializer=self.mean_initializer,
        regularizer=self.mean_regularizer,
        constraint=self.mean_constraint,
        dtype=dtype,
        trainable=True)
    self.stddev = self.add_weight(
        'stddev',
        shape=shape,
        initializer=self.stddev_initializer,
        regularizer=self.stddev_regularizer,
        constraint=self.stddev_constraint,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None, partition_info=None):
    del partition_info  # unused arg
    if not self.built:
      self.build(shape, dtype)
    return ed.Independent(
        ed.Normal(loc=self.mean, scale=self.stddev).distribution,
        reinterpreted_batch_ndims=len(shape))

  def get_config(self):
    return {
        'mean_initializer':
            tf.keras.initializers.serialize(self.mean_initializer),
        'stddev_initializer':
            tf.keras.initializers.serialize(self.stddev_initializer),
        'mean_regularizer':
            tf.keras.regularizers.serialize(self.mean_regularizer),
        'stddev_regularizer':
            tf.keras.regularizers.serialize(self.stddev_regularizer),
        'mean_constraint':
            tf.keras.constraints.serialize(self.mean_constraint),
        'stddev_constraint':
            tf.keras.constraints.serialize(self.stddev_constraint),
        'seed': self.seed,
        'dtype': self.dtype,
    }


class TrainableHeNormal(TrainableNormal):
  """Trainable normal initialized per He et al. 2015, given a ReLU nonlinearity.

  The distribution is initialized to a Normal scaled by `sqrt(2 / fan_in)`,
  where `fan_in` is the number of input units. A ReLU nonlinearity is assumed
  for this initialization scheme.

  References:
    He K, Zhang X, Ren S, Sun J. Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification. In Proceedings of the
    IEEE international conference on computer vision 2015 (pp. 1026-1034).
    https://arxiv.org/abs/1502.01852
  """

  def __init__(self, seed=None, dtype=tf.float32):
    super(TrainableHeNormal, self).__init__(
        stddev_initializer=ScaledNormalStdDev(scale=2.0, seed=seed,
                                              dtype=dtype),
        seed=seed, dtype=dtype)

  def get_config(self):
    return {
        'seed': self.seed,
        'dtype': self.dtype,
    }


class TrainableGlorotNormal(TrainableNormal):
  """Trainable normal initialized per Glorot and Bengio, 2010.

  The distribution is initialized to a Normal scaled by `sqrt(2 / fan_in +
  fan_out)`, where `fan_in` is the number of input units and `fan_out` is the
  number of output units.

  References:
    Glorot X, Bengio Y. Understanding the difficulty of training deep
    feedforward neural networks. In Proceedings of the thirteenth international
    conference on artificial intelligence and statistics 2010 Mar 31 (pp.
    249-256). http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  """

  def __init__(self, seed=None, dtype=tf.float32):
    super(TrainableGlorotNormal, self).__init__(
        stddev_initializer=ScaledNormalStdDev(mode='fan_avg', seed=seed,
                                              dtype=dtype),
        seed=seed, dtype=dtype)

  def get_config(self):
    return {
        'seed': self.seed,
        'dtype': self.dtype
    }


# Compatibility aliases, following tf.keras

# pylint: disable=invalid-name
scaled_normal_std_dev = ScaledNormalStdDev
trainable_normal = TrainableNormal
trainable_he_normal = TrainableHeNormal
trainable_glorot_normal = TrainableGlorotNormal
# pylint: enable=invalid-name

# Utility functions, following tf.keras


def serialize(initializer):
  return tf.keras.utils.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return tf.keras.utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='initializers')


def get(identifier, value=None):
  """Getter for loading from strings; returns value if can't load."""
  if value is None:
    value = identifier
  if identifier is None:
    return None
  elif isinstance(identifier, dict):
    try:
      return deserialize(identifier)
    except ValueError:
      return value
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    try:
      return deserialize(config)
    except ValueError:
      return value
  elif callable(identifier):
    return identifier
  return value
