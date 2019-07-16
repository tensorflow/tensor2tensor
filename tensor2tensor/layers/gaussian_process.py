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

"""Gaussian process layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.keras import constraints
from tensor2tensor.keras import initializers
from tensor2tensor.keras import regularizers
from tensor2tensor.layers import bayes

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed


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


@bayes.add_weight
class SparseGaussianProcess(GaussianProcess):
  r"""Gaussian process layer with inducing input and output variables.

  The layer represents a distribution over functions, where a
  stochastic forward pass appears as

  ```none
  f ~ GP(f | inducing_inputs, inducing_outputs; mean_fn, covariance_fn)
  outputs = f(inputs)
  ```

  The arguments `inducing_inputs` and `inducing_outputs`
  capture data that the GP "memorizes", i.e., it forms a posterior predictive
  distribution. Typically in a variational inference scheme (and by default),
  the inducing outputs are normally distributed with learnable location and
  scale parameters, and the inducing inputs are learnable parameters.

  Given a call to `inputs` with these defaults, an equivalent formulation in
  terms of function outputs is

  ```none
  inducing_outputs ~ Normal(inducing_outputs | mean, stddev)
  outputs ~ \prod_{unit=1}^{units} MultivariateNormal(output[:, unit] |
      mean = mean_fn(inputs) + Knm Kmm^{-1} (inducing_outputs[:, unit]-mean),
      covariance = Knn - Knm Kmm^{-1} Kmn)
  ```

  where Knm is the covariance function evaluated between all `inputs` and
  `inducing_inputs`; Knn is between all `inputs`; Kmm is between all
  `inducing_inputs`; and mean is the mean function evaluated on
  `inducing_inputs`. The multivariate normal is correlated across input
  dimensions and is independent across output dimensions.

  #### Examples

  We demonstrate a three-layer deep GP with variational inference (Salimbeni and
  Deisenroth, 2017; Damianou and Lawrence, 2013). The code snippet mirrors
  Figure 5 of Bayesian Layers. We apply it for regression given batches of
  spatial inputs and vector-valued outputs. We flatten inputs to use the
  default squared exponential kernel; this naturally extends to pass in a
  more sophisticated kernel function.

  ```python
  from tensor2tensor.layers import bayes

  batch_size = 256
  dataset_size = 10000
  features, labels = load_spatial_data(batch_size)

  model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    layers.SparseGaussianProcess(256, num_inducing=512),
    layers.SparseGaussianProcess(256, num_inducing=512),
    layers.SparseGaussianProcess(10, num_inducing=512),
  ])
  predictions = model(features)
  nll = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
  kl = sum(model.losses) / dataset_size
  loss = nll + kl
  train_op = tf.train.AdamOptimizer().minimize(loss)
  ```
  """

  def __init__(
      self,
      units,
      num_inducing,
      mean_fn=Zeros(),
      covariance_fn=ExponentiatedQuadratic(variance=1., lengthscale=1.),
      inducing_inputs_initializer='random_normal',
      inducing_outputs_initializer='trainable_normal',
      inducing_inputs_regularizer=None,
      inducing_outputs_regularizer='normal_kl_divergence',
      inducing_inputs_constraint=None,
      inducing_outputs_constraint=None,
      **kwargs):
    """Constructs layer.

    Args:
      units: integer, dimensionality of layer.
      num_inducing: integer, number of inducing points for the approximation.
      mean_fn: Mean function, a callable taking an inputs Tensor of shape
        [batch, ...] and returning a Tensor of shape [batch].
      covariance_fn: Covariance function, a callable taking two input Tensors
        of shape [batch_x1, ...] and [batch_x2, ...] respectively, and returning
        a positive semi-definite matrix of shape [batch_x1, batch_x2].
      inducing_inputs_initializer: Initializer for the inducing inputs.
      inducing_outputs_initializer: Initializer for the inducing outputs.
      inducing_inputs_regularizer: Regularizer function applied to the inducing
        inputs.
      inducing_outputs_regularizer: Regularizer function applied to the inducing
        outputs.
      inducing_inputs_constraint: Constraint function applied to the inducing
        inputs.
      inducing_outputs_constraint: Constraint function applied to the inducing
        outputs.
      **kwargs: kwargs passed to parent class.
    """
    super(SparseGaussianProcess, self).__init__(
        units=units,
        mean_fn=mean_fn,
        covariance_fn=covariance_fn,
        conditional_inputs=None,
        conditional_outputs=None,
        **kwargs)
    self.num_inducing = num_inducing
    self.inducing_inputs_initializer = initializers.get(
        inducing_inputs_initializer)
    self.inducing_outputs_initializer = initializers.get(
        inducing_outputs_initializer)
    self.inducing_inputs_regularizer = regularizers.get(
        inducing_inputs_regularizer)
    self.inducing_outputs_regularizer = regularizers.get(
        inducing_outputs_regularizer)
    self.inducing_inputs_constraint = constraints.get(
        inducing_inputs_constraint)
    self.inducing_outputs_constraint = constraints.get(
        inducing_outputs_constraint)

  def build(self, input_shape=None):
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    if isinstance(input_dim, tf.Dimension):
      input_dim = input_dim.value
    self.conditional_inputs = self.add_weight(
        shape=(self.num_inducing, input_dim),
        name='inducing_inputs',
        initializer=self.inducing_inputs_initializer,
        regularizer=self.inducing_inputs_regularizer,
        constraint=self.inducing_inputs_constraint)
    self.conditional_outputs = self.add_weight(
        shape=(self.num_inducing, self.units),
        name='inducing_outputs',
        initializer=self.inducing_outputs_initializer,
        regularizer=self.inducing_outputs_regularizer,
        constraint=self.inducing_outputs_constraint)
    super(SparseGaussianProcess, self).build(input_shape)


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


def batch_mlp(inputs, hidden_sizes):
  """Apply MLP to the final axis of a 3D tensor.

  Args:
    inputs: input Tensor of shape [batch_size, n, d_in].
    hidden_sizes: An iterable containing the hidden layer sizes of the MLP.

  Returns:
    Tensor of shape [batch_size, n, d_out] where d_out = output_sizes[-1].
  """
  batch_size, _, filter_size = inputs.shape.as_list()
  hidden = tf.reshape(inputs, (-1, filter_size))

  for size in hidden_sizes[:-1]:
    hidden = tf.keras.layers.Dense(size, activation=tf.nn.relu)(hidden)

  output = tf.keras.layers.Dense(hidden_sizes[-1], activation=None)(hidden)
  output = tf.reshape(output, (batch_size, -1, hidden_sizes[-1]))
  return output


# TODO(adityagrover): Reimplement using preexisting attention routines in T2T
def uniform_attention(q, v):
  """Computes uniform attention. Equivalent to neural process.

  Args:
    q: queries. Tensor of shape [batch_size, m, d_k].
    v: values. Tensor of shape [batch_size, n, d_v].

  Returns:
    Tensor of shape [batch_size, m, d_v].
  """
  total_points = tf.shape(q)[1]
  rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [batch_size, 1, d_v]
  rep = tf.tile(rep, [1, total_points, 1])
  return rep


def laplace_attention(q, k, v, scale, normalise):
  """Computes laplace exponential attention.

  Args:
    q: queries. Tensor of shape [batch_size, m, d_k].
    k: keys. Tensor of shape [batch_size, n, d_k].
    v: values. Tensor of shape [batch_size, n, d_v].
    scale: float that scales the L1 distance.
    normalise: Boolean that determines whether weights sum to 1.

  Returns:
    Tensor of shape [batch_size, m, d_v].
  """
  k = tf.expand_dims(k, axis=1)  # [batch_size, 1, n, d_k]
  q = tf.expand_dims(q, axis=2)  # [batch_size, m, 1, d_k]
  unnorm_weights = - tf.abs((k - q) / scale)  # [batch_size, m, n, d_k]
  unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [batch_size, m, n]
  if normalise:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = lambda x: 1 + tf.tanh(x)
  weights = weight_fn(unnorm_weights)  # [batch_size, m, n]
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [batch_size, m, d_v]
  return rep


def dot_product_attention(q, k, v, normalise):
  """Computes dot product attention.

  Args:
    q: queries. Tensor of  shape [batch_size, m, d_k].
    k: keys. Tensor of shape [batch_size, n, d_k].
    v: values. Tensor of shape [batch_size, n, d_v].
    normalise: Boolean that determines whether weights sum to 1.

  Returns:
    Tensor of shape [batch_size, m, d_v].
  """
  d_k = tf.shape(q)[-1]
  scale = tf.sqrt(tf.cast(d_k, tf.float32))
  unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [batch_size,m,n]
  if normalise:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = tf.sigmoid
  weights = weight_fn(unnorm_weights)  # [batch_size,m,n]
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [batch_size,m,d_v]
  return rep


def multihead_attention(q, k, v, num_heads=8):
  """Computes multi-head attention.

  Args:
    q: queries. Tensor of  shape [batch_size, m, d_k].
    k: keys. Tensor of shape [batch_size, n, d_k].
    v: values. Tensor of shape [batch_size, n, d_v].
    num_heads: number of heads. Should divide d_v.

  Returns:
    Tensor of shape [batch_size, m, d_v].
  """
  d_k = q.get_shape().as_list()[-1]
  d_v = v.get_shape().as_list()[-1]
  head_size = int(d_v / num_heads)
  key_initializer = tf.random_normal_initializer(stddev=d_k**-0.5)
  value_initializer = tf.random_normal_initializer(stddev=d_v**-0.5)
  rep = tf.constant(0.0)
  for h in range(num_heads):
    o = dot_product_attention(
        tf.keras.layers.Conv1D(
            head_size, 1, kernel_initializer=key_initializer,
            name='wq%d' % h, use_bias=False, padding='VALID')(q),
        tf.keras.layers.Conv1D(
            head_size, 1, kernel_initializer=key_initializer,
            name='wk%d' % h, use_bias=False, padding='VALID')(k),
        tf.keras.layers.Conv1D(
            head_size, 1, kernel_initializer=key_initializer,
            name='wv%d' % h, use_bias=False, padding='VALID')(v),
        normalise=True)
    rep += tf.keras.layers.Conv1D(d_v, 1, kernel_initializer=value_initializer,
                                  name='wo%d' % h, use_bias=False,
                                  padding='VALID')(o)
  return rep


# TODO(adityagrover): Implement via T2T.
class Attention(object):
  """The Attention module."""

  def __init__(self, rep, output_sizes, att_type, scale=1., normalise=True,
               num_heads=8):
    """Creates a attention module.

    Takes in context inputs, target inputs and
    representations of each context input/output pair
    to output an aggregated representation of the context data.

    Args:
      rep: transformation to apply to contexts before computing attention.
          One of: ['identity', 'mlp'].
      output_sizes: list of number of hidden units per layer of mlp.
          Used only if rep == 'mlp'.
      att_type: type of attention. One of the following:
          ['uniform', 'laplace', 'dot_product', 'multihead']
      scale: scale of attention.
      normalise: Boolean determining whether to:
          1. apply softmax to weights so they sum to 1 across context pts or
          2. apply custom transformation to have weights in [0, 1].
      num_heads: number of heads for multihead.
    """
    self._rep = rep
    self._output_sizes = output_sizes
    self._type = att_type
    self._scale = scale
    self._normalise = normalise
    if self._type == 'multihead':
      self._num_heads = num_heads

  def __call__(self, x1, x2, r):
    """Applies attention to create aggregated representation of r.

    Args:
      x1: Tensor of shape [B ,n1, d_x].
      x2: Tensor of shape [batch_size, n2, d_x].
      r: Tensor of shape [batch_size, n1, d].

    Returns:
      Tensor of shape [batch_size, n2, d]

    Raises:
      NameError: The argument for rep/type was invalid.
    """
    if self._rep == 'identity':
      k, q = (x1, x2)
    elif self._rep == 'mlp':
      k = batch_mlp(x1, self._output_sizes)
      q = batch_mlp(x2, self._output_sizes)
    else:
      raise NameError("'rep' not among ['identity', 'mlp']")

    if self._type == 'uniform':
      rep = uniform_attention(q, r)
    elif self._type == 'laplace':
      rep = laplace_attention(q, k, r, self._scale, self._normalise)
    elif self._type == 'dot_product':
      rep = dot_product_attention(q, k, r, self._normalise)
    elif self._type == 'multihead':
      rep = multihead_attention(q, k, r, self._num_heads)
    else:
      raise NameError(("'att_type' not among ['uniform', 'laplace', "
                       "'dot_product', 'multihead']"))

    return rep


# TODO(adityagrover): Make the encoder and decoder configurable.
class NeuralProcess(tf.keras.Model):
  """Attentive Neural Process (Kim et al., 2019; Garnelo et al., 2018)."""

  def __init__(self,
               latent_encoder_sizes,
               num_latents,
               decoder_sizes,
               use_deterministic_path=True,
               deterministic_encoder_sizes=None,
               attention_wrapper=None):
    """Initializes the Neural Process model.

    Args:
      latent_encoder_sizes: (list of ints) Hidden layer sizes for latent
          encoder.
      num_latents: (int) Dimensionality of global latent variable.
      decoder_sizes: (list of ints) Hidden layer sizes for decoder
      use_deterministic_path: (bool) Uses deterministic encoder as well if True.
      deterministic_encoder_sizes: (list of ints) Hidden layer sizes for
          deterministic encoder.
      attention_wrapper: Instance of Attention class to apply for
          determinitic encoder embedding.
    """
    super(NeuralProcess, self).__init__()
    self._num_latents = num_latents
    self._latent_encoder_sizes = latent_encoder_sizes
    self._deterministic_encoder_sizes = deterministic_encoder_sizes
    self._decoder_sizes = decoder_sizes
    self._use_deterministic_path = use_deterministic_path
    self._attention = attention_wrapper

  def latent_encoder(self, x, y):
    """Encodes the inputs into one representation.

    Args:
      x: Tensor of shape [batch_size, observations, d_x]. For the prior, these
         are context x-values. For the posterior, these are target x-values.
      y: Tensor of shape [batch_size, observations, d_y]. For the prior, these
         are context y-values. For the posterior, these are target y-values.

    Returns:
      A normal distribution over tensors of shape [batch_size, num_latents].
    """
    encoder_input = tf.concat([x, y], axis=-1)
    per_example_embedding = batch_mlp(
        encoder_input, self._latent_encoder_sizes)
    dataset_embedding = tf.reduce_mean(per_example_embedding, axis=1)
    hidden = tf.keras.layers.Dense(
        (self._latent_encoder_sizes[-1] + self._num_latents)//2,
        activation=tf.nn.relu)(dataset_embedding)
    loc = tf.keras.layers.Dense(self._num_latents, activation=None)(hidden)
    untransformed_scale = tf.keras.layers.Dense(self._num_latents,
                                                activation=None)(hidden)
    # Constraint scale following Garnelo et al. (2018).
    scale_diag = 0.1 + 0.9 * tf.sigmoid(untransformed_scale)
    return ed.MultivariateNormalDiag(loc=loc,
                                     scale_diag=scale_diag)

  def deterministic_encoder(self, context_x, context_y, target_x):
    """Encodes the inputs into one representation.

    Args:
      context_x: Tensor of shape [batch_size, observations, d_x].
        Observed x-values.
      context_y: Tensor of shape [batch_size, observations, d_y].
        Observed y-values.
      target_x: Tensor of shape [batch_size, target_observations, d_x].
        Target x-values.

    Returns:
      Encodings. Tensor of shape [batch_size, target_observations, d].
    """
    encoder_input = tf.concat([context_x, context_y], axis=-1)
    per_example_embedding = batch_mlp(encoder_input,
                                      self._deterministic_encoder_sizes)
    per_target_embedding = self._attention(context_x,
                                           target_x,
                                           per_example_embedding)
    return per_target_embedding

  def decoder(self, representation, target_x):
    """Decodes the individual targets.

    Args:
      representation: The representation of the context for target predictions.
          Tensor of shape [batch_size, target_observations, ?].
      target_x: The x locations for the target query.
          Tensor of shape [batch_size, target_observations, d_x].

    Returns:
      dist: A multivariate Gaussian over the target points. A distribution over
          tensors of shape [batch_size, target_observations, d_y].
    """
    decoder_input = tf.concat([representation, target_x], axis=-1)
    hidden = batch_mlp(decoder_input, self._decoder_sizes)
    loc, untransformed_scale = tf.split(hidden, 2, axis=-1)
    scale_diag = 0.1 + 0.9 * tf.nn.softplus(untransformed_scale)
    return tfp.distributions.MultivariateNormalDiag(loc=loc,
                                                    scale_diag=scale_diag)

  def __call__(self, query, target_y=None):
    """Returns the predicted mean and variance at the target points.

    Args:
      query: Nested tuple containing ((context_x, context_y), target_x) where:
              context_x is Tensor of shape [batch_size, num_contexts, d_x].
                  Contains the x values of the context points.
              context_y is Tensor of shape [batch_size, num_contexts, d_y].
                  Contains the y values of the context points.
              target_x is Tensor of shape [batch_size, num_targets, d_x].
                  Contains the x values of the target points.
      target_y: The ground truth y values of the target y.
          Tensor of shape [batch_size, num_targets, d_y].

    Returns:
      predictive_dist: Predictive posterior distribution over the predicted y.
    """

    (context_x, context_y), target_x = query
    num_targets = tf.shape(target_x)[1]
    prior = self.latent_encoder(context_x, context_y)

    # For training, when target_y is available, use targets for latent encoder.
    # Note that targets contain contexts by design.
    # For testing, when target_y unavailable, use contexts for latent encoder.
    if target_y is None:
      latent_rep = prior
    else:
      posterior = self.latent_encoder(target_x, target_y)
      latent_rep = posterior
    latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                         [1, num_targets, 1])
    if self._use_deterministic_path:
      deterministic_rep = self.deterministic_encoder(context_x,
                                                     context_y,
                                                     target_x)
      representation = tf.concat([deterministic_rep, latent_rep], axis=-1)
    else:
      representation = latent_rep

    predictive_dist = self.decoder(representation, target_x)

    if target_y is not None:
      kl = tf.expand_dims(
          posterior.distribution.kl_divergence(prior.distribution),
          -1)
      self.add_loss(lambda: kl)

    return predictive_dist

  call = __call__



