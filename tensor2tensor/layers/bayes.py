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
from tensor2tensor.keras import constraints
from tensor2tensor.keras import initializers
from tensor2tensor.keras import regularizers

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed


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
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, *args, **kwargs):
    self.call_weights()
    kwargs.pop('training', None)
    return super(Conv2DReparameterization, self).call(*args, **kwargs)


class Conv2DFlipout(Conv2DReparameterization):
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

  This layer uses the Flipout estimator (Wen et al., 2018) for integrating with
  respect to the `kernel`. Namely, it applies
  pseudo-independent weight perturbations via independent sign flips for each
  example, enabling variance reduction over independent weight perturbations.
  For this estimator to work, the `kernel` random variable must be able
  to decompose as a sum of its mean and a perturbation distribution; the
  perturbation distribution must be independent across weight elements and
  symmetric around zero (for example, a fully factorized Gaussian).
  """

  def call(self, inputs):
    self.call_weights()
    if not isinstance(self.kernel, ed.RandomVariable):
      return super(Conv2DFlipout, self).call(inputs)
    input_shape = tf.shape(inputs)
    batch_dim = input_shape[0]
    if self.data_format == 'channels_first':
      channels = input_shape[1]
      sign_input_shape = [batch_dim, channels, 1, 1]
      sign_output_shape = [batch_dim, self.filters, 1, 1]
    else:
      channels = input_shape[-1]
      sign_input_shape = [batch_dim, 1, 1, channels]
      sign_output_shape = [batch_dim, 1, 1, self.filters]
    sign_input = 2 * tf.random.uniform(sign_input_shape,
                                       minval=0,
                                       maxval=2,
                                       dtype=inputs.dtype) - 1
    sign_output = 2 * tf.random.uniform(sign_output_shape,
                                        minval=0,
                                        maxval=2,
                                        dtype=inputs.dtype) - 1
    kernel_mean = self.kernel.distribution.mean()
    perturbation = self.kernel - kernel_mean
    outputs = self._convolution_op(inputs, kernel_mean)
    outputs += self._convolution_op(inputs * sign_input,
                                    perturbation) * sign_output
    if self.use_bias:
      if self.data_format == 'channels_first':
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs


@add_weight
class Conv2DVariationalDropout(tf.keras.layers.Conv2D):
  """2D convolution layer with variational dropout (Kingma et al., 2015).

  Implementation follows the additive parameterization of
  Molchanov et al. (2017).
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
               kernel_regularizer='log_uniform_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv2DVariationalDropout, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, inputs, training=None):
    self.call_weights()
    if training is None:
      training = tf.keras.backend.learning_phase()

    def dropped_inputs():
      """Forward pass with dropout."""
      # Clip magnitude of dropout rate, where we get the dropout rate alpha from
      # the additive parameterization (Molchanov et al., 2017): for weight ~
      # Normal(mu, sigma**2), the variance `sigma**2 = alpha * mu**2`.
      mean = self.kernel.distribution.mean()
      log_variance = tf.log(self.kernel.distribution.variance())
      log_alpha = log_variance - tf.log(tf.square(mean) +
                                        tf.keras.backend.epsilon())
      log_alpha = tf.clip_by_value(log_alpha, -8., 8.)
      log_variance = log_alpha + tf.log(tf.square(mean) +
                                        tf.keras.backend.epsilon())

      means = self._convolution_op(inputs, mean)
      stddevs = tf.sqrt(
          self._convolution_op(tf.square(inputs), tf.exp(log_variance)) +
          tf.keras.backend.epsilon())
      outputs = means + stddevs * tf.random_normal(tf.shape(stddevs))
      if self.use_bias:
        if self.data_format == 'channels_first':
          outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
        else:
          outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')
      if self.activation is not None:
        outputs = self.activation(outputs)
      return outputs

    # Following tf.keras.Dropout, only apply variational dropout if training
    # flag is True. The kernel must also be a random variable.
    training_value = smart_constant_value(training)
    if training_value is not None:
      if training_value and isinstance(self.kernel, ed.RandomVariable):
        return dropped_inputs()
      else:
        return super(Conv2DVariationalDropout, self).call(inputs)
    else:
      return tf.cond(tf.logical_and(training,
                                    isinstance(self.kernel, ed.RandomVariable)),
                     dropped_inputs,
                     lambda: super(Conv2DVariationalDropout, self).call(inputs))


# From `tensorflow/python/framework/smart_cond.py`
def smart_constant_value(pred):
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Arguments:
    pred: A scalar, either a Python bool or tensor.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Tensor or bool.
  """
  if pred in {0, 1}:  # Accept 1/0 as valid boolean values
    pred_value = bool(pred)
  elif isinstance(pred, bool):
    pred_value = pred
  elif isinstance(pred, tf.Tensor):
    pred_value = tf.contrib.util.constant_value(pred)
  else:
    raise TypeError('`pred` must be a Tensor, or a Python bool, or 1 or 0. '
                    'Found instead: %s' % pred)
  return pred_value


@add_weight
class DenseDVI(tf.keras.layers.Dense):
  """Densely-connected layer with deterministic VI (Wu et al., 2018).

  This layer computes a variational inference approximation via first and second
  moments. It is accurate if the kernel and bias initializers return factorized
  normal random variables and the number of units is sufficiently large. The
  advantage is that the forward pass is deterministic, reducing variance of
  gradients during training. The disadvantage is an O(features^2*units) compute
  and O(features^2 + features*units) memory complexity. In comparison,
  DenseReparameterization has O(features*units) compute and memory complexity.

  #### Examples

  Below implements deterministic variational inference for Bayesian
  feedforward network regression. We use the exact expected log-likelihood from
  Wu et al. (2018), Eq. 8. Assume 2-D real-valued tensors of `features` and
  `labels` of shapes `[batch_size, num_features]` and `[batch_size, 1]`
  respectively.

  ```python
  from tensor2tensor.layers import bayes

  model = tf.keras.Sequential([
      bayes.DenseDVI(256, activation=tf.nn.relu),
      bayes.DenseDVI(256, activation=tf.nn.relu),
      bayes.DenseDVI(1, activation=None),
  ])
  locs = model(features)
  nll = 0.5 * tf.reduce_mean(locs.distribution.variance() +
                             (labels - locs.distribution.mean())**2)
  kl = sum(model.losses) / total_dataset_size
  loss = nll + kl
  train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
  ```

  For evaluation, feed in data and use, e.g., `predictions.distribution.mean()`
  to make predictions via the posterior predictive distribution.

  ```python
  predictions = ed.Normal(loc=locs.distribution.mean(),
                          scale=locs.distribution.variance() + 1.)
  ```
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
    super(DenseDVI, self).__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, inputs):
    self.call_weights()
    if (not isinstance(inputs, ed.RandomVariable) and
        not isinstance(self.kernel, ed.RandomVariable) and
        not isinstance(self.bias, ed.RandomVariable)):
      return super(DenseDVI, self).call(inputs)
    inputs_mean, inputs_variance, inputs_covariance = get_moments(inputs)
    kernel_mean, kernel_variance, _ = get_moments(self.kernel)
    if self.use_bias:
      bias_mean, _, bias_covariance = get_moments(self.bias)

    # E[outputs] = E[inputs] * E[kernel] + E[bias]
    mean = tf.tensordot(inputs_mean, kernel_mean, [[-1], [0]])
    if self.use_bias:
      mean = tf.nn.bias_add(mean, bias_mean)

    # Cov = E[inputs**2] Cov(kernel) + E[W]^T Cov(inputs) E[W] + Cov(bias)
    # For first term, assume Cov(kernel) = 0 on off-diagonals so we only
    # compute diagonal term.
    covariance_diag = tf.tensordot(inputs_variance + inputs_mean**2,
                                   kernel_variance, [[-1], [0]])
    # Compute quadratic form E[W]^T Cov E[W] from right-to-left. First is
    #  [..., features, features], [features, units] -> [..., features, units].
    cov_w = tf.tensordot(inputs_covariance, kernel_mean, [[-1], [0]])
    # Next is [..., features, units], [features, units] -> [..., units, units].
    w_cov_w = tf.tensordot(cov_w, kernel_mean, [[-2], [0]])
    covariance = w_cov_w
    if self.use_bias:
      covariance += bias_covariance
    covariance = tf.matrix_set_diag(
        covariance, tf.matrix_diag_part(covariance) + covariance_diag)

    if self.activation in (tf.keras.activations.relu, tf.nn.relu):
      # Compute activation's moments with variable names from Wu et al. (2018).
      variance = tf.matrix_diag_part(covariance)
      scale = tf.sqrt(variance)
      mu = mean / (scale + tf.keras.backend.epsilon())
      mean = scale * soft_relu(mu)

      pairwise_variances = (tf.expand_dims(variance, -1) *
                            tf.expand_dims(variance, -2))  # [..., units, units]
      rho = covariance / tf.sqrt(pairwise_variances +
                                 tf.keras.backend.epsilon())
      rho = tf.clip_by_value(rho,
                             -1. / (1. + tf.keras.backend.epsilon()),
                             1. / (1. + tf.keras.backend.epsilon()))
      s = covariance / (rho + tf.keras.backend.epsilon())
      mu1 = tf.expand_dims(mu, -1)  # [..., units, 1]
      mu2 = tf.matrix_transpose(mu1)  # [..., 1, units]
      a = (soft_relu(mu1) * soft_relu(mu2) +
           rho * tfp.distributions.Normal(0., 1.).cdf(mu1) *
           tfp.distributions.Normal(0., 1.).cdf(mu2))
      gh = tf.asinh(rho)
      bar_rho = tf.sqrt(1. - rho**2)
      gr = gh + rho / (1. + bar_rho)
      # Include numerically stable versions of gr and rho when multiplying or
      # dividing them. The sign of gr*rho and rho/gr is always positive.
      safe_gr = tf.abs(gr) + 0.5 * tf.keras.backend.epsilon()
      safe_rho = tf.abs(rho) + tf.keras.backend.epsilon()
      exp_negative_q = gr / (2. * math.pi) * tf.exp(
          -safe_rho / (2. * safe_gr * (1 + bar_rho)) +
          (gh - rho) / (safe_gr * safe_rho) * mu1 * mu2)
      covariance = s * (a + exp_negative_q)
    elif self.activation not in (tf.keras.activations.linear, None):
      raise NotImplementedError('Activation is {}. Deterministic variational '
                                'inference is only available if activation is '
                                'ReLU or None.'.format(self.activation))

    return ed.MultivariateNormalFullCovariance(mean, covariance)


def get_moments(x):
  """Gets first and second moments of input."""
  if isinstance(x, ed.RandomVariable):
    mean = x.distribution.mean()
    variance = x.distribution.variance()
    try:
      covariance = x.distribution.covariance()
    except NotImplementedError:
      covariance = tf.zeros(x.shape.concatenate(x.shape[-1]), dtype=x.dtype)
      covariance = tf.matrix_set_diag(covariance, variance)
  else:
    mean = x
    variance = tf.zeros_like(x)
    covariance = tf.zeros(x.shape.concatenate(x.shape[-1]), dtype=x.dtype)
  return mean, variance, covariance


def soft_relu(x):
  return (tfp.distributions.Normal(0., 1.).prob(x) +
          x * tfp.distributions.Normal(0., 1.).cdf(x))


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
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, *args, **kwargs):
    self.call_weights()
    kwargs.pop('training', None)
    return super(DenseReparameterization, self).call(*args, **kwargs)


class DenseFlipout(DenseReparameterization):
  """Bayesian densely-connected layer estimated via Flipout (Wen et al., 2018).

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

  This layer uses the Flipout estimator (Wen et al., 2018) for integrating with
  respect to the `kernel`. Namely, it applies
  pseudo-independent weight perturbations via independent sign flips for each
  example, enabling variance reduction over independent weight perturbations.
  For this estimator to work, the `kernel` random variable must be able
  to decompose as a sum of its mean and a perturbation distribution; the
  perturbation distribution must be independent across weight elements and
  symmetric around zero (for example, a fully factorized Gaussian).
  """

  def call(self, inputs):
    self.call_weights()
    if not isinstance(self.kernel, ed.RandomVariable):
      return super(DenseFlipout, self).call(inputs)
    input_shape = tf.shape(inputs)
    sign_input = 2 * tf.random.uniform(input_shape,
                                       minval=0,
                                       maxval=2,
                                       dtype=inputs.dtype) - 1
    sign_output = 2 * tf.random.uniform(tf.concat([input_shape[:-1],
                                                   [self.units]], 0),
                                        minval=0,
                                        maxval=2,
                                        dtype=inputs.dtype) - 1
    kernel_mean = self.kernel.distribution.mean()
    perturbation = self.kernel - kernel_mean
    if inputs.shape.ndims <= 2:
      outputs = tf.matmul(inputs, kernel_mean)
      outputs += tf.matmul(inputs * sign_input, perturbation) * sign_output
    else:
      outputs = tf.tensordot(inputs, kernel_mean, [[-1], [0]])
      outputs += tf.tensordot(inputs * sign_input,
                              perturbation,
                              [[-1], [0]]) * sign_output
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs


@add_weight
class DenseVariationalDropout(tf.keras.layers.Dense):
  """Densely-connected layer with variational dropout (Kingma et al., 2015).

  Implementation follows the additive parameterization of
  Molchanov et al. (2017).
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zero',
               kernel_regularizer='log_uniform_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               **kwargs):
    super(DenseVariationalDropout, self).__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, inputs, training=None):
    self.call_weights()
    if training is None:
      training = tf.keras.backend.learning_phase()

    def dropped_inputs():
      """Forward pass with dropout."""
      # Clip magnitude of dropout rate, where we get the dropout rate alpha from
      # the additive parameterization (Molchanov et al., 2017): for weight ~
      # Normal(mu, sigma**2), the variance `sigma**2 = alpha * mu**2`.
      mean = self.kernel.distribution.mean()
      log_variance = tf.log(self.kernel.distribution.variance())
      log_alpha = log_variance - tf.log(tf.square(mean) +
                                        tf.keras.backend.epsilon())
      log_alpha = tf.clip_by_value(log_alpha, -8., 8.)
      log_variance = log_alpha + tf.log(tf.square(mean) +
                                        tf.keras.backend.epsilon())

      if inputs.shape.ndims <= 2:
        means = tf.matmul(inputs, mean)
        stddevs = tf.sqrt(
            tf.matmul(tf.square(inputs), tf.exp(log_variance)) +
            tf.keras.backend.epsilon())
      else:
        means = tf.tensordot(inputs, mean, [[-1], [0]])
        stddevs = tf.sqrt(
            tf.tensordot(tf.square(inputs), tf.exp(log_variance), [[-1], [0]]) +
            tf.keras.backend.epsilon())
      outputs = means + stddevs * tf.random_normal(tf.shape(stddevs))
      if self.use_bias:
        outputs = tf.nn.bias_add(outputs, self.bias)
      if self.activation is not None:
        outputs = self.activation(outputs)
      return outputs

    # Following tf.keras.Dropout, only apply variational dropout if training
    # flag is True. The kernel must also be a random variable.
    training_value = smart_constant_value(training)
    if training_value is not None:
      if training_value and isinstance(self.kernel, ed.RandomVariable):
        return dropped_inputs()
      else:
        return super(DenseVariationalDropout, self).call(inputs)
    else:
      return tf.cond(tf.logical_and(training,
                                    isinstance(self.kernel, ed.RandomVariable)),
                     dropped_inputs,
                     lambda: super(DenseVariationalDropout, self).call(inputs))


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
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        recurrent_initializer=initializers.get(recurrent_initializer),
        bias_initializer=initializers.get(bias_initializer),
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=regularizers.get(kernel_regularizer),
        recurrent_regularizer=regularizers.get(recurrent_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        recurrent_constraint=constraints.get(recurrent_constraint),
        bias_constraint=constraints.get(bias_constraint),
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
          bias_initializer = initializers.TrainableNormal(
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

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Get the initial state and side-effect sampling of stochastic weights."""
    if self.built:
      self.call_weights()
    return super(LSTMCellReparameterization, self).get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)


class LSTMCellFlipout(LSTMCellReparameterization):
  """Bayesian LSTM cell class estimated via Flipout (Wen et al., 2018).

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

  This layer uses the Flipout estimator (Wen et al., 2018) for integrating with
  respect to the `kernel` and `recurrent_kernel`. Namely, it applies
  pseudo-independent weight perturbations via independent sign flips for each
  example, enabling variance reduction over independent weight perturbations.
  For this estimator to work, the `kernel` and `recurrent_kernel` random
  variable must be able to decompose as a sum of its mean and a perturbation
  distribution; the perturbation distribution must be independent across weight
  elements and symmetric around zero (for example, a fully factorized Gaussian).
  """

  def _call_sign_flips(self, inputs=None, batch_size=None, dtype=None):
    """Builds per-example sign flips for pseudo-independent perturbations."""
    # TODO(trandustin): We add and call this method separately from build().
    # This is because build() operates on a static input_shape. We need dynamic
    # input shapes as we operate on the batch size which is often dynamic.
    if inputs is not None:
      batch_size = tf.shape(inputs)[0]
      dtype = inputs.dtype
    input_dim = tf.shape(self.kernel)[0]
    self.sign_input = 2 * tf.random.uniform(
        [batch_size, 4 * input_dim], minval=0, maxval=2, dtype=dtype) - 1
    self.sign_output = 2 * tf.random.uniform(
        [batch_size, 4 * self.units], minval=0, maxval=2, dtype=dtype) - 1
    self.recurrent_sign_input = 2 * tf.random.uniform(
        [batch_size, 4 * self.units], minval=0, maxval=2, dtype=dtype) - 1
    self.recurrent_sign_output = 2 * tf.random.uniform(
        [batch_size, 4 * self.units], minval=0, maxval=2, dtype=dtype) - 1

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Get the initial state and side-effect sampling of stochastic weights."""
    if self.built:
      self._call_sign_flips(inputs, batch_size, dtype)
    return super(LSTMCellFlipout, self).get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    if not isinstance(self.recurrent_kernel, ed.RandomVariable):
      return super(LSTMCellFlipout, self)._compute_carry_and_output(x,
                                                                    h_tm1,
                                                                    c_tm1)
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    kernel_mean = self.recurrent_kernel.distribution.mean()
    perturbation = self.recurrent_kernel - kernel_mean
    k_i, k_f, k_c, k_o = tf.split(kernel_mean, num_or_size_splits=4, axis=1)
    p_i, p_f, p_c, p_o = tf.split(perturbation, num_or_size_splits=4, axis=1)
    si_i, si_f, si_c, si_o = tf.split(self.recurrent_sign_input,
                                      num_or_size_splits=4, axis=1)
    so_i, so_f, so_c, so_o = tf.split(self.recurrent_sign_output,
                                      num_or_size_splits=4, axis=1)
    z0 = (x_i + tf.keras.backend.dot(h_tm1_i, k_i) +
          tf.keras.backend.dot(h_tm1_i * si_i, p_i) * so_i)
    z1 = (x_f + tf.keras.backend.dot(h_tm1_f, k_f) +
          tf.keras.backend.dot(h_tm1_f * si_f, p_f) * so_f)
    z2 = (x_c + tf.keras.backend.dot(h_tm1_c, k_c) +
          tf.keras.backend.dot(h_tm1_c * si_c, p_c) * so_c)
    z3 = (x_o + tf.keras.backend.dot(h_tm1_o, k_o) +
          tf.keras.backend.dot(h_tm1_o * si_o, p_o) * so_o)
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states, training=None):
    # TODO(trandustin): Enable option for Flipout on only the kernel or
    # recurrent_kernel. If only one is a random variable, we currently default
    # to weight reparameterization.
    if (not isinstance(self.kernel, ed.RandomVariable) or
        not isinstance(self.recurrent_kernel, ed.RandomVariable)):
      return super(LSTMCellFlipout, self).call(inputs, states, training)
    if not hasattr(self, 'sign_input'):
      self._call_sign_flips(inputs)
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
      kernel_mean = self.kernel.distribution.mean()
      perturbation = self.kernel - kernel_mean
      k_i, k_f, k_c, k_o = tf.split(kernel_mean, num_or_size_splits=4, axis=1)
      p_i, p_f, p_c, p_o = tf.split(perturbation, num_or_size_splits=4, axis=1)
      si_i, si_f, si_c, si_o = tf.split(self.sign_input,
                                        num_or_size_splits=4, axis=1)
      so_i, so_f, so_c, so_o = tf.split(self.sign_output,
                                        num_or_size_splits=4, axis=1)
      x_i = (tf.keras.backend.dot(inputs_i, k_i) +
             tf.keras.backend.dot(inputs_i * si_i, p_i) * so_i)
      x_f = (tf.keras.backend.dot(inputs_f, k_f) +
             tf.keras.backend.dot(inputs_f * si_f, p_f) * so_f)
      x_c = (tf.keras.backend.dot(inputs_c, k_c) +
             tf.keras.backend.dot(inputs_c * si_c, p_c) * so_c)
      x_o = (tf.keras.backend.dot(inputs_o, k_o) +
             tf.keras.backend.dot(inputs_o * si_o, p_o) * so_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = tf.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = tf.keras.backend.bias_add(x_i, b_i)
        x_f = tf.keras.backend.bias_add(x_f, b_f)
        x_c = tf.keras.backend.bias_add(x_c, b_c)
        x_o = tf.keras.backend.bias_add(x_o, b_o)

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
      kernel_mean = self.kernel.distribution.mean()
      perturbation = self.kernel - kernel_mean
      z = tf.keras.backend.dot(inputs, kernel_mean)
      z += tf.keras.backend.dot(inputs * self.sign_input,
                                perturbation) * self.sign_output
      if 0. < self.recurrent_dropout < 1.:
        h_tm1 = h_tm1 * rec_dp_mask[0]
      recurrent_kernel_mean = self.recurrent_kernel.distribution.mean()
      perturbation = self.recurrent_kernel - recurrent_kernel_mean
      z += tf.keras.backend.dot(h_tm1, recurrent_kernel_mean)
      z += tf.keras.backend.dot(h_tm1 * self.recurrent_sign_input,
                                perturbation) * self.recurrent_sign_output
      if self.use_bias:
        z = tf.keras.backend.bias_add(z, self.bias)

      z = tf.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]


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
