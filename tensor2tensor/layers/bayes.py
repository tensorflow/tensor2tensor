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
import tensorflow_probability as tfp

from tensorflow_probability import edward2 as ed


class Positive(tf.keras.constraints.Constraint):
  """Positive constraint."""

  def __init__(self, epsilon=tf.keras.backend.epsilon()):
    self.epsilon = epsilon

  def __call__(self, w):
    return tf.maximum(w, self.epsilon)

  def get_config(self):
    return {'epsilon': self.epsilon}


def positive():  # alias, following tf.keras.constraints
  return Positive()


class Zeros(object):
  """Function returning zeros tensor of same shape excluding the last dim."""

  def __call__(self, inputs):
    return tf.zeros(tf.shape(inputs)[:-1], inputs.dtype)

  def get_config(self):
    return {}


class ExponentiatedQuadratic(object):
  """Exponentiated quadratic kernel."""

  def __init__(self, variance, lengthscale):
    self.variance = variance
    self.lengthscale = lengthscale

  def __call__(self, x1, x2):
    """Computes exponentiated quadratic over all pairs of inputs.

    Args:
      x1: Tensor of shape [batch_x1, ...]. Slices along the batch axis denote an
        individual input to be passed to the kernel. It is computed pairwise
        with each input sliced from x2.
      x2: Tensor of shape [batch_x2, ...]. Slices along the batch axis denote an
        individual input passed to the kernel function. It is computed pairwise
        with each input sliced from x1.

    Returns:
      Tensor of shape [batch_x1, batch_x2].
    """
    size = tf.convert_to_tensor(x1).shape.ndims
    if size > 2:
      raise NotImplementedError('Multiple feature dimensions is not yet '
                                'supported.')
    x1 = x1 / self.lengthscale
    x2 = x2 / self.lengthscale
    x1_squared = tf.reduce_sum(tf.square(x1), list(range(1, len(x1.shape))))
    x2_squared = tf.reduce_sum(tf.square(x2), list(range(1, len(x2.shape))))
    square = (x1_squared[:, tf.newaxis] +
              x2_squared[tf.newaxis, :] -
              2 * tf.matmul(x1, x2, transpose_b=True))
    return self.variance * tf.exp(-square / 2)

  def get_config(self):
    return {'variance': self.variance, 'lengthscale': self.lengthscale}


class LinearKernel(object):
  """Linear kernel, optionally on top of a feature extractor (e.g., encoder)."""

  def __init__(self, variance, bias, encoder=tf.identity):
    self.variance = variance
    self.bias = bias
    self.encoder = encoder

  def __call__(self, x1, x2):
    """Computes scaled dot product of over all pairs of encoded inputs.

    Args:
      x1: Tensor of shape [batch_x1] + encoder domain. Slices along the batch
        axis denote an individual input to be passed to the kernel. It is
        computed pairwise with each input sliced from x2.
      x2: Tensor of shape [batch_x2] + encoder domain. Slices along the batch
        axis denote an individual input to be passed to the kernel. It is
        computed pairwise with each input sliced from x1.

    Returns:
      Tensor of shape [batch_x1, batch_x2].
    """
    encoded_x1 = self.encoder(x1)
    encoded_x2 = self.encoder(x2)
    dot_product = tf.matmul(encoded_x1, encoded_x2, transpose_b=True)
    return tf.sqrt(self.variance) * dot_product + self.bias

  def get_config(self):
    return {
        'variance': self.variance,
        'bias': self.bias,
        'encoder': tf.keras.utils.serialize_keras_object(self.encoder),
    }


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
               stddev_constraint=positive(),
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
    return ed.Independent(
        ed.Normal(loc=self.mean, scale=self.stddev).distribution,
        reinterpreted_batch_ndims=len(self.shape))

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
    random_variable = ed.Independent(
        ed.Normal(
            loc=tf.broadcast_to(self.mean, x.distribution.event_shape),
            scale=tf.broadcast_to(self.stddev, x.distribution.event_shape)
        ).distribution,
        reinterpreted_batch_ndims=len(x.distribution.event_shape))
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


class GaussianProcess(tf.keras.layers.Layer):
  r"""Gaussian process layer.

  The layer represents a distribution over functions, where a
  stochastic forward pass appears as

  ```none
  f ~ GP(f | conditional_inputs, conditional_outputs; mean_fn, covariance_fn)
  outputs = f(inputs)
  ```

  The optional arguments `conditional_inputs` and `conditional_outputs`
  capture data that the GP "memorizes", i.e., it forms a posterior predictive
  distribution. If left unspecified, the GP posits a prior predictive.

  Given a call to `inputs`, an equivalent formulation in terms of function
  outputs is

  ```none
  outputs ~ \prod_{unit=1}^{units} MultivariateNormal(output[:, unit] |
      mean = mean_fn(inputs) + Knm Kmm^{-1} (conditional_outputs[:, unit]-mean),
      covariance = Knn - Knm Kmm^{-1} Kmn)
  ```

  where Knm is the covariance function evaluated between all `inputs` and
  `conditional_inputs`; Knn is between all `inputs`; Kmm is between all
  `conditional_inputs`; and mean is the mean function evaluated on
  `conditional_inputs`. The multivariate normal is correlated across input
  dimensions and is independent across output dimensions.
  """

  def __init__(
      self,
      units,
      mean_fn=Zeros(),
      covariance_fn=ExponentiatedQuadratic(variance=1., lengthscale=1.),
      conditional_inputs=None,
      conditional_outputs=None,
      **kwargs):
    """Constructs layer.

    Args:
      units: integer, dimensionality of layer.
      mean_fn: Mean function, a callable taking an inputs Tensor of shape
        [batch, ...] and returning a Tensor of shape [batch].
      covariance_fn: Covariance function, a callable taking two input Tensors
        of shape [batch_x1, ...] and [batch_x2, ...] respectively, and returning
        a positive semi-definite matrix of shape [batch_x1, batch_x2].
      conditional_inputs: Tensor of shape [batch, ...], where batch must be the
        same as conditional_outputs', and ellipses must match layer inputs.
      conditional_outputs: Tensor of shape [batch, units], where batch must be
        the same as conditional_inputs' and units is the layer's units size.
      **kwargs: kwargs passed to parent class.
    """
    super(GaussianProcess, self).__init__(**kwargs)
    self.units = int(units)
    self.mean_fn = mean_fn
    self.covariance_fn = covariance_fn
    self.conditional_inputs = conditional_inputs
    self.conditional_outputs = conditional_outputs

    self.supports_masking = True
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape=None):
    # Don't track trainable variables such as in the kernel. The user should
    # refer to any via, e.g., self.covariance_fn or the user environment.
    self.built = True

  def call(self, inputs):
    if self.conditional_inputs is None and self.conditional_outputs is None:
      covariance_matrix = self.covariance_fn(inputs, inputs)
      # Tile locations so output has shape [units, batch_size]. Covariance will
      # broadcast to [units, batch_size, batch_size], and we perform
      # shape manipulations to get a random variable over [batch_size, units].
      loc = self.mean_fn(inputs)
      loc = tf.tile(loc[tf.newaxis], [self.units] + [1] * len(loc.shape))
    else:
      knn = self.covariance_fn(inputs, inputs)
      knm = self.covariance_fn(inputs, self.conditional_inputs)
      kmm = self.covariance_fn(self.conditional_inputs, self.conditional_inputs)
      kmm = tf.matrix_set_diag(
          kmm, tf.matrix_diag_part(kmm) + tf.keras.backend.epsilon())
      kmm_tril = tf.linalg.cholesky(kmm)
      kmm_tril_operator = tf.linalg.LinearOperatorLowerTriangular(kmm_tril)
      knm_operator = tf.linalg.LinearOperatorFullMatrix(knm)

      # TODO(trandustin): Vectorize linear algebra for multiple outputs. For
      # now, we do each separately and stack to obtain a locations Tensor of
      # shape [units, batch_size].
      loc = []
      for conditional_outputs_unit in tf.unstack(self.conditional_outputs,
                                                 axis=-1):
        center = conditional_outputs_unit - self.mean_fn(
            self.conditional_inputs)
        loc_unit = knm_operator.matvec(
            kmm_tril_operator.solvevec(kmm_tril_operator.solvevec(center),
                                       adjoint=True))
        loc.append(loc_unit)
      loc = tf.stack(loc) + self.mean_fn(inputs)[tf.newaxis]

      covariance_matrix = knn
      covariance_matrix -= knm_operator.matmul(
          kmm_tril_operator.solve(
              kmm_tril_operator.solve(knm, adjoint_arg=True), adjoint=True))

    covariance_matrix = tf.matrix_set_diag(
        covariance_matrix,
        tf.matrix_diag_part(covariance_matrix) + tf.keras.backend.epsilon())

    # Form a multivariate normal random variable with batch_shape units and
    # event_shape batch_size. Then make it be independent across the units
    # dimension. Then transpose its dimensions so it is [batch_size, units].
    random_variable = ed.MultivariateNormalFullCovariance(
        loc=loc, covariance_matrix=covariance_matrix)
    random_variable = ed.Independent(random_variable.distribution,
                                     reinterpreted_batch_ndims=1)
    bijector = tfp.bijectors.Inline(
        forward_fn=lambda x: tf.transpose(x, [1, 0]),
        inverse_fn=lambda y: tf.transpose(y, [1, 0]),
        forward_event_shape_fn=lambda input_shape: input_shape[::-1],
        forward_event_shape_tensor_fn=lambda input_shape: input_shape[::-1],
        inverse_log_det_jacobian_fn=lambda y: tf.cast(0, y.dtype),
        forward_min_event_ndims=2)
    random_variable = ed.TransformedDistribution(random_variable.distribution,
                                                 bijector=bijector)
    return random_variable

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    input_dim = input_shape[-1]
    if isinstance(input_dim, tf.Dimension):
      input_dim = input_dim.value
    if input_dim is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'mean_fn': tf.keras.utils.serialize_keras_object(self.mean_fn),
        'covariance_fn': tf.keras.utils.serialize_keras_object(
            self.covariance_fn),
        'conditional_inputs': None,  # don't serialize as it can be large
        'conditional_outputs': None,  # don't serialize as it can be large
    }
    base_config = super(GaussianProcess, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


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


class BayesianLinearModel(tf.keras.Model):
  r"""Bayesian linear model with standard normal prior over its coefficients.

  A forward pass computes the mean of the exact predictive distribution

  ```none
  p(outputs | inputs) = \int Normal(outputs | coeffs * inputs, noise_variance)
                             Normal(coeffs | 0, 1) dweights dbias.
  ```

  It takes a Tensor of shape [batch_size, input_dim] as input and returns a
  Normal random variable of shape [batch_size] representing its outputs.
  After `fit()`, the forward pass computes the exact posterior predictive
  distribution.
  """

  def __init__(self, noise_variance, **kwargs):
    super(BayesianLinearModel, self).__init__(**kwargs)
    self.noise_variance = noise_variance
    self.coeffs_precision_tril_op = None
    self.coeffs_mean = None

  def call(self, inputs):
    if self.coeffs_mean is None and self.coeffs_precision_tril_op is None:
      # p(mean(ynew) | xnew) = Normal(ynew | mean = 0, variance = xnew xnew^T)
      predictive_mean = 0.
      predictive_variance = tf.reduce_sum(tf.square(inputs), -1)
    else:
      # p(mean(ynew) | xnew, x, y) = Normal(ynew |
      #   mean = xnew (1/noise_variance) (1/noise_variance x^T x + I)^{-1}x^T y,
      #   variance = xnew (1/noise_variance x^T x + I)^{-1} xnew^T)
      predictive_mean = tf.einsum('nm,m->n', inputs, self.coeffs_mean)
      predictive_covariance = tf.matmul(
          inputs,
          self.coeffs_precision_tril_op.solve(
              self.coeffs_precision_tril_op.solve(inputs, adjoint_arg=True),
              adjoint=True))
      predictive_variance = tf.diag_part(predictive_covariance)
    return ed.Normal(loc=predictive_mean, scale=tf.sqrt(predictive_variance))

  def fit(self, x=None, y=None):
    # p(coeffs | x, y) = Normal(coeffs |
    #   mean = (1/noise_variance) (1/noise_variance x^T x + I)^{-1} x^T y,
    #   covariance = (1/noise_variance x^T x + I)^{-1})
    # TODO(trandustin): We newly fit the data at each call. Extend to do
    # Bayesian updating.
    kernel_matrix = tf.matmul(x, x, transpose_a=True) / self.noise_variance
    coeffs_precision = tf.matrix_set_diag(
        kernel_matrix, tf.matrix_diag_part(kernel_matrix) + 1.)
    coeffs_precision_tril = tf.linalg.cholesky(coeffs_precision)
    self.coeffs_precision_tril_op = tf.linalg.LinearOperatorLowerTriangular(
        coeffs_precision_tril)
    self.coeffs_mean = self.coeffs_precision_tril_op.solvevec(
        self.coeffs_precision_tril_op.solvevec(tf.einsum('nm,n->m', x, y)),
        adjoint=True) / self.noise_variance
    # TODO(trandustin): To be fully Keras-compatible, return History object.
    return


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
