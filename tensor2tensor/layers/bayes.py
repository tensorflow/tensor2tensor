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

"""Bayesian layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math

import six
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
    return self.variance * dot_product + self.bias

  def get_config(self):
    return {
        'variance': self.variance,
        'bias': self.bias,
        'encoder': tf.keras.utils.serialize_keras_object(self.encoder),
    }


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
    self.mean_regularizer = get(mean_regularizer)
    self.stddev_regularizer = get(stddev_regularizer)
    self.mean_constraint = get(mean_constraint)
    self.stddev_constraint = get(stddev_constraint)
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


def add_weight(cls):
  """Decorator for Layers, overriding add_weight for trainable initializers."""
  @functools.wraps(cls.add_weight)
  def _add_weight(self,
                  name=None,
                  shape=None,
                  dtype=None,
                  initializer=None,
                  regularizer=None,
                  **kwargs):
    """Adds weight."""
    if isinstance(initializer, tf.keras.layers.Layer):
      weight = initializer(shape, dtype)
      self._trainable_weights.extend(initializer.trainable_weights)  # pylint: disable=protected-access
      self._non_trainable_weights.extend(initializer.non_trainable_weights)  # pylint: disable=protected-access
      if regularizer is not None:
        # TODO(trandustin): Replace need for this with
        # Layer._handle_weight_regularization. For Eager compatibility, random
        # variable __init__s cannot apply TF ops (cl/220898007).
        def loss_fn():
          """Creates a regularization loss `Tensor`."""
          with tf.name_scope(name + '/Regularizer'):
            return regularizer(initializer(shape, dtype))
        self.add_loss(loss_fn)
      return weight
    return super(cls, self).add_weight(name=name,
                                       shape=shape,
                                       dtype=dtype,
                                       initializer=initializer,
                                       regularizer=regularizer,
                                       **kwargs)
  cls.add_weight = _add_weight
  return cls


@add_weight
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
               kernel_initializer='trainable_normal',
               bias_initializer='zero',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               **kwargs):
    super(DenseReparameterization, self).__init__(
        units=units,
        activation=get(activation),
        use_bias=use_bias,
        kernel_initializer=get(kernel_initializer),
        bias_initializer=get(bias_initializer),
        kernel_regularizer=get(kernel_regularizer),
        bias_regularizer=get(bias_regularizer),
        activity_regularizer=get(activity_regularizer),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, *args, **kwargs):
    self.call_weights()
    return super(DenseReparameterization, self).call(*args, **kwargs)


@add_weight
class Conv2DReparameterization(tf.keras.layers.Conv2D):
  """2D convolution layer (e.g. spatial convolution over images).

  The layer computes a variational Bayesian approximation to the distribution
  over convolutional layers,

  ```
  p(outputs | inputs) = int conv2d(inputs; weights, bias) p(weights, bias)
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
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zeros',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv2DReparameterization, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=get(activation),
        use_bias=use_bias,
        kernel_initializer=get(kernel_initializer),
        bias_initializer=get(bias_initializer),
        kernel_regularizer=get(kernel_regularizer),
        bias_regularizer=get(bias_regularizer),
        activity_regularizer=get(activity_regularizer),
        kernel_constraint=get(kernel_constraint),
        bias_constraint=get(bias_constraint),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, *args, **kwargs):
    self.call_weights()
    return super(Conv2DReparameterization, self).call(*args, **kwargs)


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


@add_weight
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
               kernel_initializer='trainable_normal',
               recurrent_initializer='trainable_normal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer='normal_kl_divergence',
               recurrent_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):
    super(LSTMCellReparameterization, self).__init__(
        units=units,
        activation=get(activation),
        recurrent_activation=get(recurrent_activation),
        use_bias=use_bias,
        kernel_initializer=get(kernel_initializer),
        recurrent_initializer=get(recurrent_initializer),
        bias_initializer=get(bias_initializer),
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=get(kernel_regularizer),
        recurrent_regularizer=get(recurrent_regularizer),
        bias_regularizer=get(bias_regularizer),
        kernel_constraint=get(kernel_constraint),
        recurrent_constraint=get(recurrent_constraint),
        bias_constraint=get(bias_constraint),
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        **kwargs)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    if isinstance(input_dim, tf.Dimension):
      input_dim = input_dim.value
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if self.unit_forget_bias:
        if isinstance(self.bias_initializer, tf.keras.layers.Layer):
          def bias_mean_initializer(_, *args, **kwargs):
            return tf.concat([
                tf.keras.initializers.truncated_normal(
                    stddev=1e-5)((self.units,), *args, **kwargs),
                tf.keras.initializers.truncated_normal(
                    mean=1., stddev=1e-5)((self.units,), *args, **kwargs),
                tf.keras.initializers.truncated_normal(
                    stddev=1e-5)((self.units * 2,), *args, **kwargs),
            ], axis=0)
          bias_initializer = TrainableNormal(
              mean_initializer=bias_mean_initializer)
        else:
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

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.recurrent_initializer, tf.keras.layers.Layer):
      self.recurrent_kernel = self.recurrent_initializer(
          self.recurrent_kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  # NOTE: This will not be called in TF < 1.11.
  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Get the initial state and side-effect sampling of stochastic weights."""
    if self.built:
      self.call_weights()
    return super(LSTMCellReparameterization, self).get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)


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


# Compatibility aliases, following tf.keras

# pylint: disable=invalid-name
positive = Positive
scaled_normal_std_dev = ScaledNormalStdDev
trainable_normal = TrainableNormal
trainable_he_normal = TrainableHeNormal
trainable_glorot_normal = TrainableGlorotNormal
normal_kl_divergence = NormalKLDivergence
# pylint: enable=invalid-name

# Utility functions, following tf.keras


def deserialize(config, custom_objects=None):
  return tf.keras.utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='bayes')


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
