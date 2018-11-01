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

"""Bayesian layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Softplus(tf.keras.constraints.Constraint):
  """Softplus constraint."""

  def __init__(self, epsilon=tf.keras.backend.epsilon()):
    self.epsilon = epsilon

  def __call__(self, w):
    return tf.nn.softplus(w) + self.epsilon

  def get_config(self):
    return {'epsilon': self.epsilon}


def softplus():  # alias, following tf.keras.constraints
  return Softplus()


class TrainableInitializer(tf.keras.initializers.Initializer):
  """An initializer with trainable variables.

  In this implementation, a layer must call `build` before usage in order to
  capture the variables.
  """

  def __init__(self):
    self.built = False

  def build(self, shape, dtype=None, add_variable_fn=None):
    """Builds the initializer, with the variables captured by the caller."""
    raise NotImplementedError


class TrainableNormal(TrainableInitializer):
  """Random normal op as an initializer with trainable mean and stddev."""

  def __init__(self,
               mean_initializer=tf.random_normal_initializer(stddev=0.1),
               unconstrained_stddev_initializer=tf.random_normal_initializer(
                   mean=-3., stddev=0.1),
               mean_regularizer=None,
               unconstrained_stddev_regularizer=None,
               mean_constraint=None,
               unconstrained_stddev_constraint=softplus(),
               seed=None,
               dtype=tf.float32):
    """Constructs the initializer."""
    super(TrainableNormal, self).__init__()
    self.mean_initializer = mean_initializer
    self.unconstrained_stddev_initializer = unconstrained_stddev_initializer
    self.mean_regularizer = mean_regularizer
    self.unconstrained_stddev_regularizer = unconstrained_stddev_regularizer
    self.mean_constraint = mean_constraint
    self.unconstrained_stddev_constraint = unconstrained_stddev_constraint
    self.seed = seed
    self.dtype = tf.as_dtype(dtype)

  def build(self, shape, dtype=None, add_variable_fn=None):
    """Builds the initializer, with the variables captured by the caller."""
    if dtype is None:
      dtype = self.dtype
    self.shape = shape
    self.dtype = tf.as_dtype(dtype)

    self.mean = add_variable_fn(
        'mean',
        shape=shape,
        initializer=self.mean_initializer,
        regularizer=self.mean_regularizer,
        constraint=self.mean_constraint,
        dtype=dtype,
        trainable=True)
    self.stddev = add_variable_fn(
        'unconstrained_stddev',
        shape=shape,
        initializer=self.unconstrained_stddev_initializer,
        regularizer=self.unconstrained_stddev_regularizer,
        constraint=self.unconstrained_stddev_constraint,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape=None, dtype=None, partition_info=None):
    del shape, dtype, partition_info  # Unused in TrainableInitializers.
    # TODO(dusenberrymw): Restructure so that we can build as needed.
    if not self.built:
      raise ValueError('A TrainableInitializer must be built by a layer before '
                       'usage, and is currently only compatible with Bayesian '
                       'layers.')
    noise = tf.random_normal(self.shape, dtype=self.dtype, seed=self.seed)
    output = self.mean + self.stddev * noise
    # TODO(trandustin): Hack to store parameters so KL reg. can operate on them.
    output._parameters = (self.mean, self.stddev)  # pylint: disable=protected-access
    return output

  def get_config(self):
    return {
        'mean_initializer':
            tf.keras.initializers.serialize(self.mean_initializer),
        'unconstrained_stddev_initializer':
            tf.keras.initializers.serialize(
                self.unconstrained_stddev_initializer),
        'mean_regularizer':
            tf.keras.regularizers.serialize(self.mean_regularizer),
        'unconstrained_stddev_regularizer':
            tf.keras.regularizers.serialize(
                self.unconstrained_stddev_regularizer),
        'mean_constraint':
            tf.keras.constraints.serialize(self.mean_constraint),
        'unconstrained_stddev_constraint':
            tf.keras.constraints.serialize(
                self.unconstrained_stddev_constraint),
        'seed': self.seed,
        'dtype': self.dtype.name,
    }


def trainable_normal():  # alias, following tf.keras.initializers
  return TrainableNormal()


class NormalKLDivergence(tf.keras.regularizers.Regularizer):
  """KL divergence regularizer from one normal distribution to another."""

  def __init__(self, mean=0., stddev=1.):
    """Construct regularizer where default is a KL towards the std normal."""
    self.mean = mean
    self.stddev = stddev

  def __call__(self, x):
    mean, stddev = x._parameters  # pylint: disable=protected-access
    variance2 = tf.square(self.stddev)
    variance_ratio = tf.square(stddev) / variance2
    regularization = tf.square(mean - self.mean) / (2. * variance2)
    regularization += (variance_ratio - 1. - tf.log(variance_ratio)) / 2.
    return regularization

  def get_config(self):
    return {
        'mean': self.mean,
        'stddev': self.stddev,
    }


def normal_kl_divergence():  # alias, following tf.keras.regularizers
  return NormalKLDivergence()


class DenseReparameterization(tf.keras.layers.Dense):
  """Bayesian densely-connected layer estimated via reparameterization.

  The layer computes a variational Bayesian approximation to the distribution
  over densely-connected layers,

  ```
  p(outputs | inputs) = int dense(inputs; weights, bias) p(weights, bias)
    dweights dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. Gradients with respect to the
  distributions' learnable parameters backpropagate via reparameterization.
  Minimizing cross-entropy plus the layer's losses performs variational
  minimum description length, i.e., it minimizes an upper bound to the negative
  marginal likelihood.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer='zero',
               kernel_regularizer=normal_kl_divergence(),
               bias_regularizer=None,
               activity_regularizer=None,
               **kwargs):
    if not kernel_initializer:
      kernel_initializer = trainable_normal()
    if not bias_initializer:
      bias_initializer = trainable_normal()
    super(DenseReparameterization, self).__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        **kwargs)

  @property
  def kernel(self):
    if isinstance(self.kernel_initializer, TrainableInitializer):
      return self.kernel_initializer()
    else:
      return self._kernel

  @property
  def bias(self):
    if isinstance(self.bias_initializer, TrainableInitializer):
      return self.bias_initializer()
    else:
      return self._bias

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = input_shape[-1]
    if isinstance(last_dim, tf.Dimension):
      last_dim = last_dim.value
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = tf.layers.InputSpec(min_ndim=2, axes={-1: last_dim})

    if isinstance(self.kernel_initializer, TrainableInitializer):
      self.kernel_initializer.build([last_dim, self.units],
                                    self.dtype,
                                    self.add_weight)
    else:
      self._kernel = self.add_weight(
          'kernel',
          shape=[last_dim, self.units],
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint,
          dtype=self.dtype,
          trainable=True)

    if self.kernel_regularizer is not None:
      self._handle_weight_regularization('kernel',
                                         self.kernel,
                                         self.kernel_regularizer)

    if self.use_bias:
      if isinstance(self.bias_initializer, TrainableInitializer):
        self.bias_initializer.build([self.units], self.dtype, self.add_weight)
      else:
        self._bias = self.add_weight(
            'bias',
            shape=[self.units],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)

      if self.bias_regularizer is not None:
        self._handle_weight_regularization('bias',
                                           self.bias,
                                           self.bias_regularizer)
    else:
      self.bias = None
    self.built = True
