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

from tensorflow_probability import edward2 as ed


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


# TODO(dusenberrymw): Restructure the implementation of a trainable initializer
# such that callers do not need to have type-conditional logic.
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
               stddev_initializer=tf.random_uniform_initializer(
                   minval=1e-5, maxval=0.1),
               mean_regularizer=None,
               stddev_regularizer=None,
               mean_constraint=None,
               stddev_constraint=softplus(),
               seed=None,
               dtype=tf.float32):
    """Constructs the initializer."""
    super(TrainableNormal, self).__init__()
    self.mean_initializer = mean_initializer
    self.stddev_initializer = stddev_initializer
    self.mean_regularizer = mean_regularizer
    self.stddev_regularizer = stddev_regularizer
    self.mean_constraint = mean_constraint
    self.stddev_constraint = stddev_constraint
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
        'stddev',
        shape=shape,
        initializer=self.stddev_initializer,
        regularizer=self.stddev_regularizer,
        constraint=self.stddev_constraint,
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
    return ed.Normal(loc=self.mean, scale=self.stddev)

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
    """Computes regularization given an ed.Normal random variable as input."""
    if not isinstance(x, ed.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable.')
    random_variable = ed.Normal(loc=self.mean, scale=self.stddev)
    return random_variable.distribution.kl_divergence(x.distribution)

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
      if self.kernel_regularizer is not None:
        self.add_loss(create_regularization_loss_fn(
            'kernel', lambda: self.kernel, self.kernel_regularizer))

    else:
      self._kernel = self.add_weight(
          'kernel',
          shape=[last_dim, self.units],
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint,
          dtype=self.dtype,
          trainable=True)

    if self.use_bias:
      if isinstance(self.bias_initializer, TrainableInitializer):
        self.bias_initializer.build([self.units], self.dtype, self.add_weight)
        if self.bias_regularizer is not None:
          self.add_loss(create_regularization_loss_fn(
              'bias', lambda: self.bias, self.bias_regularizer))
      else:
        self._bias = self.add_weight(
            'bias',
            shape=[self.units],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)

    else:
      self._bias = None
    self.built = True


class LSTMCellReparameterization(tf.keras.layers.LSTMCell):
  """Bayesian LSTM cell class estimated via reparameterization.

  The layer computes a variational Bayesian approximation to the distribution
  over LSTM cell functions,

  ```
  p(outputs | inputs) = int lstm_cell(inputs; weights, bias) p(weights, bias)
    dweights dbias,
  ```

  where the weights consist of both input and recurrent weights.

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel, recurrent kernel, and bias. Gradients with
  respect to the distributions' learnable parameters backpropagate via
  reparameterization.  Minimizing cross-entropy plus the layer's losses performs
  variational minimum description length, i.e., it minimizes an upper bound to
  the negative marginal likelihood.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer=None,
               recurrent_initializer=None,
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=normal_kl_divergence(),
               recurrent_regularizer=normal_kl_divergence(),
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):
    if not kernel_initializer:
      kernel_initializer = trainable_normal()
    if not recurrent_initializer:
      recurrent_initializer = trainable_normal()
    if not bias_initializer:
      bias_initializer = trainable_normal()
    super(LSTMCellReparameterization, self).__init__(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        **kwargs)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    if isinstance(input_dim, tf.Dimension):
      input_dim = input_dim.value

    if isinstance(self.kernel_initializer, TrainableInitializer):
      self.kernel_initializer.build(
          [input_dim, self.units * 4], self.dtype, self.add_weight)
      self.kernel = self.kernel_initializer()
      if self.kernel_regularizer is not None:
        self.add_loss(create_regularization_loss_fn(
            # Can't use the kernel directly because we actually need to create a
            # new Edward RV.  The Dense layer already does this.
            # Also note that the initializer is a callable.
            'kernel', self.kernel_initializer, self.kernel_regularizer))

    else:
      self.kernel = self.add_weight(
          shape=(input_dim, self.units * 4),
          name='kernel',
          initializer=self.kernel_initializer,
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

    if isinstance(self.recurrent_initializer, TrainableInitializer):
      self.recurrent_initializer.build(
          [self.units, self.units * 4], self.dtype, self.add_weight)
      self.recurrent_kernel = self.recurrent_initializer()
      if self.recurrent_regularizer is not None:
        self.add_loss(create_regularization_loss_fn(
            # Can't use the kernel directly because we actually need to create a
            # new Edward RV.  The Dense layer already does this.
            # Also note that the initializer is a callable.
            'recurrent_kernel', self.recurrent_initializer,
            self.recurrent_regularizer))

    else:
      self.recurrent_kernel = self.add_weight(
          shape=(self.units, self.units * 4),
          name='recurrent_kernel',
          initializer=self.recurrent_initializer,
          regularizer=self.recurrent_regularizer,
          constraint=self.recurrent_constraint)

    if self.use_bias:
      if isinstance(self.bias_initializer, TrainableInitializer):
        self.bias_initializer.build(
            [self.units * 4], self.dtype, self.add_weight)
        self.bias = self.bias_initializer()
        if self.bias_regularizer is not None:
          self.add_loss(create_regularization_loss_fn(
              # Can't use the bias directly because we actually need to create a
              # new Edward RV.  The Dense layer already does this.
              # Also note that the initializer is a callable.
              'bias', self.bias_initializer, self.bias_regularizer))
      else:
        if self.unit_forget_bias:

          def bias_initializer(_, *args, **kwargs):
            return tf.keras.backend.concatenate([
                self.bias_initializer((self.units,), *args, **kwargs),
                tf.keras.initializers.Ones()((self.units,), *args, **kwargs),
                self.bias_initializer((self.units * 2,), *args, **kwargs),
            ])
        else:
          bias_initializer = self.bias_initializer
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            name='bias',
            initializer=bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def sample_weights(self):
    if isinstance(self.kernel_initializer, TrainableInitializer):
      self.kernel = self.kernel_initializer()
    if isinstance(self.recurrent_initializer, TrainableInitializer):
      self.recurrent_kernel = self.recurrent_initializer()
    if isinstance(self.bias_initializer, TrainableInitializer):
      self.bias = self.bias_initializer()

  # NOTE: This will not be called in TF < 1.11.
  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Get the initial state and side-effect sampling of stochastic weights."""
    if self.built:
      self.sample_weights()
    return super(LSTMCellReparameterization, self).get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)


def create_regularization_loss_fn(name, variable_fn, regularizer_fn):
  """Create a regularization loss function.

  The callable representing the variable allows for use with Bayesian Layers.

  Args:
    name: String name scope prefix.
    variable_fn: Callable that returns a TF Variable or ed.RandomVariable.
    regularizer_fn: Callable that returns a loss tensor when called with a TF
      Variable or ed.RandomVariable.

  Returns:
    A callable that returns a regularization loss tensor when called.
  """
  def loss_fn():
    """Creates a regularization loss `Tensor`."""
    with tf.name_scope(name + '/Regularizer'):
      regularization = regularizer_fn(variable_fn())
    return regularization

  return loss_fn


class MixtureLogistic(tf.keras.layers.Layer):
  """Stochastic output layer, distributed as a mixture of logistics."""

  def __init__(self, num_components, **kwargs):
    super(MixtureLogistic, self).__init__(**kwargs)
    self.num_components = num_components
    self.layer = tf.keras.layers.Dense(num_components * 3)

  def build(self, input_shape=None):
    self.layer.build(input_shape)
    self.built = True

  def call(self, inputs):
    net = self.layer(inputs)
    logits, loc, unconstrained_scale = tf.split(net, 3, axis=-1)
    scale = tf.nn.softplus(unconstrained_scale) + tf.keras.backend.epsilon()
    return ed.MixtureSameFamily(
        mixture_distribution=ed.Categorical(logits=logits).distribution,
        components_distribution=ed.Logistic(loc=loc, scale=scale).distribution)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(input_shape)[:-1]

  def get_config(self):
    config = {'num_components': self.num_components}
    base_config = super(MixtureLogistic, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
