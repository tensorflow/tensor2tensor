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

"""Bayesian neural network layers."""

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
    if not isinstance(self.kernel, ed.RandomVariable):
      return super(Conv2DFlipout, self).call(inputs)
    self.call_weights()
    outputs = self._apply_kernel(inputs)
    if self.use_bias:
      if self.data_format == 'channels_first':
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def _apply_kernel(self, inputs):
    input_shape = tf.shape(inputs)
    batch_dim = input_shape[0]
    if self._convolution_op is None:
      padding = self.padding
      if self.padding == 'causal':
        padding = 'valid'
      if not isinstance(padding, (list, tuple)):
        padding = padding.upper()
      self._convolution_op = functools.partial(
          tf.nn.convolution,
          strides=self.strides,
          padding=padding,
          data_format='NHWC' if self.data_format == 'channels_last' else 'NCHW',
          dilations=self.dilation_rate)

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
    return outputs


class Conv2DHierarchical(Conv2DFlipout):
  """2D convolution layer with hierarchical distributions.

  The layer computes a variational Bayesian approximation to the distribution
  over convolutional layers, and where the distribution over weights
  involves a hierarchical distribution with hidden unit noise coupling vectors
  of the kernel weight matrix (Louizos et al., 2017),

  ```
  p(outputs | inputs) = int conv2d(inputs; new_kernel, bias) p(kernel,
    local_scales, global_scale, bias) dkernel dlocal_scales dglobal_scale dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. The kernel is written in non-centered
  parameterization where

  ```
  new_kernel[i, j] = kernel[i, j] * local_scale[j] * global_scale.
  ```

  That is, there is "local" multiplicative noise which couples weights for each
  output filter. There is also a "global" multiplicative noise which couples the
  entire weight matrix. By default, the weights are normally distributed and the
  local and global noises are half-Cauchy distributed; this makes the kernel a
  horseshoe distribution (Carvalho et al., 2009; Polson and Scott, 2012).

  The estimation uses Flipout for variance reduction with respect to sampling
  the full weights. Gradients with respect to the distributions' learnable
  parameters backpropagate via reparameterization. Minimizing cross-entropy
  plus the layer's losses performs variational minimum description length,
  i.e., it minimizes an upper bound to the negative marginal likelihood.
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
               local_scale_initializer='trainable_half_cauchy',
               global_scale_initializer='trainable_half_cauchy',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               local_scale_regularizer='half_cauchy_kl_divergence',
               global_scale_regularizer=regularizers.HalfCauchyKLDivergence(
                   scale=1e-5),
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               local_scale_constraint='positive',
               global_scale_constraint='positive',
               **kwargs):
    self.local_scale_initializer = initializers.get(local_scale_initializer)
    self.global_scale_initializer = initializers.get(global_scale_initializer)
    self.local_scale_regularizer = regularizers.get(local_scale_regularizer)
    self.global_scale_regularizer = regularizers.get(global_scale_regularizer)
    self.local_scale_constraint = constraints.get(local_scale_constraint)
    self.global_scale_constraint = constraints.get(global_scale_constraint)
    super(Conv2DHierarchical, self).__init__(
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

  def build(self, input_shape):
    self.local_scale = self.add_weight(
        shape=(self.filters,),
        name='local_scale',
        initializer=self.local_scale_initializer,
        regularizer=self.local_scale_regularizer,
        constraint=self.local_scale_constraint)
    self.global_scale = self.add_weight(
        shape=(),
        name='global_scale',
        initializer=self.global_scale_initializer,
        regularizer=self.global_scale_regularizer,
        constraint=self.global_scale_constraint)
    super(Conv2DHierarchical, self).build(input_shape)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.local_scale_initializer, tf.keras.layers.Layer):
      self.local_scale = self.local_scale_initializer(self.local_scale.shape,
                                                      self.dtype)
    if isinstance(self.global_scale_initializer, tf.keras.layers.Layer):
      self.global_scale = self.global_scale_initializer(self.global_scale.shape,
                                                        self.dtype)
    super(Conv2DHierarchical, self).call_weights()

  def _apply_kernel(self, inputs):
    outputs = super(Conv2DHierarchical, self)._apply_kernel(inputs)
    if self.data_format == 'channels_first':
      local_scale = tf.reshape(self.local_scale, [1, -1, 1, 1])
    else:
      local_scale = tf.reshape(self.local_scale, [1, 1, 1, -1])
    # TODO(trandustin): Figure out what to set local/global scales to at test
    # time. Means don't exist for Half-Cauchy approximate posteriors.
    outputs *= local_scale * self.global_scale
    return outputs


class Conv2DVariationalDropout(Conv2DReparameterization):
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

  def call(self, inputs, training=None):
    if not isinstance(self.kernel, ed.RandomVariable):
      return super(Conv2DVariationalDropout, self).call(inputs)
    self.call_weights()
    if training is None:
      training = tf.keras.backend.learning_phase()
    if self._convolution_op is None:
      padding = self.padding
      if self.padding == 'causal':
        padding = 'valid'
      if not isinstance(padding, (list, tuple)):
        padding = padding.upper()
      self._convolution_op = functools.partial(
          tf.nn.convolution,
          strides=self.strides,
          padding=padding,
          data_format='NHWC' if self.data_format == 'channels_last' else 'NCHW',
          dilations=self.dilation_rate)

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
      if self.use_bias:
        if self.data_format == 'channels_first':
          means = tf.nn.bias_add(means, self.bias, data_format='NCHW')
        else:
          means = tf.nn.bias_add(means, self.bias, data_format='NHWC')
      outputs = ed.Normal(loc=means, scale=stddevs)
      if self.activation is not None:
        outputs = self.activation(outputs)
      return outputs

    # Following tf.keras.Dropout, only apply variational dropout if training
    # flag is True.
    training_value = smart_constant_value(training)
    if training_value is not None:
      if training_value:
        return dropped_inputs()
      else:
        return super(Conv2DVariationalDropout, self).call(inputs)
    return tf.cond(training,
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


class DenseDVI(DenseReparameterization):
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

  def call(self, inputs):
    if (not isinstance(inputs, ed.RandomVariable) and
        not isinstance(self.kernel, ed.RandomVariable) and
        not isinstance(self.bias, ed.RandomVariable)):
      return super(DenseDVI, self).call(inputs)
    self.call_weights()
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
    if not isinstance(self.kernel, ed.RandomVariable):
      return super(DenseFlipout, self).call(inputs)
    self.call_weights()
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


class DenseVariationalDropout(DenseReparameterization):
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

  def call(self, inputs, training=None):
    if not isinstance(self.kernel, ed.RandomVariable):
      return super(DenseVariationalDropout, self).call(inputs)
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
      if self.use_bias:
        means = tf.nn.bias_add(means, self.bias)
      outputs = ed.Normal(loc=means, scale=stddevs)
      if self.activation is not None:
        outputs = self.activation(outputs)
      return outputs

    # Following tf.keras.Dropout, only apply variational dropout if training
    # flag is True.
    training_value = smart_constant_value(training)
    if training_value is not None:
      if training_value:
        return dropped_inputs()
      else:
        return super(DenseVariationalDropout, self).call(inputs)
    return tf.cond(training,
                   dropped_inputs,
                   lambda: super(DenseVariationalDropout, self).call(inputs))


class DenseHierarchical(DenseVariationalDropout):
  """Bayesian densely-connected layer with hierarchical distributions.

  The layer computes a variational Bayesian approximation to the distribution
  over densely-connected layers, and where the distribution over weights
  involves a hierarchical distribution with hidden unit noise coupling vectors
  of the kernel weight matrix (Louizos et al., 2017),

  ```
  p(outputs | inputs) = int dense(inputs; new_kernel, bias) p(kernel,
    local_scales, global_scale, bias) dkernel dlocal_scales dglobal_scale dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. The kernel is written in non-centered
  parameterization where

  ```
  new_kernel[i, j] = kernel[i, j] * local_scale[i] * global_scale.
  ```

  That is, there is "local" multiplicative noise which couples weights for each
  input neuron. There is also a "global" multiplicative noise which couples the
  entire weight matrix. By default, the weights are normally distributed and the
  local and global noises are half-Cauchy distributed; this makes the kernel a
  horseshoe distribution (Carvalho et al., 2009; Polson and Scott, 2012).

  The estimation uses local reparameterization to avoid sampling the full
  weights. Gradients with respect to the distributions' learnable parameters
  backpropagate via reparameterization. Minimizing cross-entropy plus the
  layer's losses performs variational minimum description length, i.e., it
  minimizes an upper bound to the negative marginal likelihood.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zero',
               local_scale_initializer='trainable_half_cauchy',
               global_scale_initializer='trainable_half_cauchy',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               local_scale_regularizer='half_cauchy_kl_divergence',
               global_scale_regularizer=regularizers.HalfCauchyKLDivergence(
                   scale=1e-5),
               activity_regularizer=None,
               local_scale_constraint='positive',
               global_scale_constraint='positive',
               **kwargs):
    self.local_scale_initializer = initializers.get(local_scale_initializer)
    self.global_scale_initializer = initializers.get(global_scale_initializer)
    self.local_scale_regularizer = regularizers.get(local_scale_regularizer)
    self.global_scale_regularizer = regularizers.get(global_scale_regularizer)
    self.local_scale_constraint = constraints.get(local_scale_constraint)
    self.global_scale_constraint = constraints.get(global_scale_constraint)
    super(DenseHierarchical, self).__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    if isinstance(input_dim, tf.Dimension):
      input_dim = input_dim.value
    self.local_scale = self.add_weight(
        shape=(input_dim,),
        name='local_scale',
        initializer=self.local_scale_initializer,
        regularizer=self.local_scale_regularizer,
        constraint=self.local_scale_constraint)
    self.global_scale = self.add_weight(
        shape=(),
        name='global_scale',
        initializer=self.global_scale_initializer,
        regularizer=self.global_scale_regularizer,
        constraint=self.global_scale_constraint)
    super(DenseHierarchical, self).build(input_shape)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.local_scale_initializer, tf.keras.layers.Layer):
      self.local_scale = self.local_scale_initializer(self.local_scale.shape,
                                                      self.dtype)
    if isinstance(self.global_scale_initializer, tf.keras.layers.Layer):
      self.global_scale = self.global_scale_initializer(self.global_scale.shape,
                                                        self.dtype)
    super(DenseHierarchical, self).call_weights()

  def call(self, inputs, training=None):
    self.call_weights()
    # TODO(trandustin): Figure out what to set local/global scales to at test
    # time. Means don't exist for Half-Cauchy approximate posteriors.
    inputs *= self.local_scale[tf.newaxis, :] * self.global_scale
    return super(DenseHierarchical, self).call(inputs, training=training)


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


class NCPNormalPerturb(tf.keras.layers.Layer):
  """Noise contrastive prior for continuous inputs (Hafner et al., 2018).

  The layer doubles the inputs' batch size and adds a random normal perturbation
  to the concatenated second batch. This acts an input prior to be used in
  combination with an output prior. The output prior reduces the second batch
  (reverting to the inputs' original shape) and computes a regularizer that
  matches the second batch towards some output (e.g., uniform distribution).
  This layer implementation is inspired by the Aboleth library.

  #### Examples

  Below implements neural network regression with heteroskedastic noise,
  noise contrastive priors, and being Bayesian only at the mean's output layer.

  ```python
  from tensor2tensor.layers import bayes

  batch_size, dataset_size = 128, 1000
  features, labels = get_some_dataset()

  inputs = keras.Input(shape=(25,))
  x = bayes.NCPNormalPerturb()(inputs)  # double input batch
  x = layers.Dense(64, activation='relu')(x)
  x = layers.Dense(64, activation='relu')(x)
  means = bayes.DenseVariationalDropout(1, activation=None)(x)  # get mean dist.
  means = bayes.NCPNormalOutput(labels)(means)  # halve input batch
  stddevs = tf.keras.layers.Dense(1, activation='softplus')(x[:batch_size])
  outputs = tf.keras.layers.Lambda(lambda x: ed.Normal(x[0], x[1]))([means,
                                                                     stddevs])
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  predictions = model(features)
  loss = tf.reduce_mean(predictions.distribution.log_prob(labels))
  loss += model.losses[0] / dataset_size  # KL regularizer for output layer
  loss += model.losses[-1]
  train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
  ```

  The network applies `bayes.NCPNormalPerturb()` to double the input batch
  size and add Gaussian noise to the second half; then feedforward layers; then
  `bayes.DenseVariational` to be Bayesian about the output density's mean; then
  `bayes.NCPNormalOutput` centered at the labels to revert to the batch size
  and compute a loss on the second half; then parameterize the output density's
  standard deviations; then compute the total loss function as the sum of the
  model's negative log-likelihood, KL divergence for the Bayesian mean layer,
  and NCP loss.
  """

  def __init__(self, mean=0., stddev=1., seed=None, **kwargs):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    super(NCPNormalPerturb, self).__init__(**kwargs)

  def call(self, inputs):
    noise = tf.random.normal(tf.shape(inputs),
                             mean=self.mean,
                             stddev=self.stddev,
                             dtype=inputs.dtype,
                             seed=self.seed)
    perturbed_inputs = inputs + noise
    return tf.concat([inputs, perturbed_inputs], 0)


class NCPCategoricalPerturb(tf.keras.layers.Layer):
  """Noise contrastive prior for discrete inputs (Hafner et al., 2018).

  The layer doubles the inputs' batch size and randomly flips categories
  for the concatenated second batch (all features must be integer-valued). This
  acts an input prior to be used in combination with an output prior. The output
  prior reduces the second batch (reverting to the inputs' original shape) and
  computes a regularizer that matches the second batch towards some output
  (e.g., uniform distribution). This layer implementation is inspired by the
  Aboleth library.

  #### Examples

  Below implements neural network regression with heteroskedastic noise,
  noise contrastive priors, and being Bayesian only at the mean's output layer.

  ```python
  from tensor2tensor.layers import bayes

  batch_size, dataset_size = 128, 1000
  features, labels = get_some_dataset()

  inputs = keras.Input(shape=(25,))
  x = bayes.NCPCategoricalPerturb(10)(inputs)  # double input batch
  x = layers.Dense(64, activation='relu')(x)
  x = layers.Dense(64, activation='relu')(x)
  means = bayes.DenseVariationalDropout(1, activation=None)(x)  # get mean dist.
  means = bayes.NCPNormalOutput(labels)(means)  # halve input batch
  stddevs = tf.keras.layers.Dense(1, activation='softplus')(x[:batch_size])
  outputs = tf.keras.layers.Lambda(lambda x: ed.Normal(x[0], x[1]))([means,
                                                                     stddevs])
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  predictions = model(features)
  loss = tf.reduce_mean(predictions.distribution.log_prob(labels))
  loss += model.losses[0] / dataset_size  # KL regularizer for output layer
  loss += model.losses[-1]
  train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
  ```

  The network applies `bayes.NCPCategoricalPerturb()` to double the input batch
  size and flip categories for the second half; then feedforward layers; then
  `bayes.DenseVariational` to be Bayesian about the output density's mean; then
  `bayes.NCPNormalOutput` centered at the labels to revert to the batch size
  and compute a loss on the second half; then parameterize the output density's
  standard deviations; then compute the total loss function as the sum of the
  model's negative log-likelihood, KL divergence for the Bayesian mean layer,
  and NCP loss.
  """

  def __init__(self, input_dim, probs=0.1, **kwargs):
    """Creates layer.

    Args:
      input_dim: int > 0. Size of the category, i.e. maximum integer index + 1.
      probs: Probability that a category is randomly flipped.
      **kwargs: kwargs to parent class.
    """
    self.input_dim = input_dim
    self.probs = probs
    super(NCPCategoricalPerturb, self).__init__(**kwargs)

  def call(self, inputs):
    mask = tf.cast(tf.random.uniform(tf.shape(inputs)) <= self.probs,
                   inputs.dtype)
    flips = tf.random.uniform(
        tf.shape(inputs), minval=0, maxval=self.input_dim, dtype=inputs.dtype)
    flipped_inputs = mask * flips + (1 - mask) * inputs
    return tf.concat([inputs, flipped_inputs], 0)


class NCPNormalOutput(tf.keras.layers.Layer):
  """Noise contrastive prior for continuous outputs (Hafner et al., 2018).

  The layer returns the first half of the inputs' batch. It computes a KL
  regularizer as a side-effect, which matches the inputs' second half towards a
  normal distribution (the output prior), and averaged over the number of inputs
  in the second half. This layer is typically in combination with an input prior
  which doubles the batch. This layer implementation is inspired by the Aboleth
  library.

  The layer computes the exact KL divergence from a normal distribution to
  the input RandomVariable. It is an unbiased estimate if the input
  RandomVariable has random parameters. If the input is a Tensor, then it
  assumes its density is `ed.Normal(input, 1.)`, i.e., mean squared error loss.

  #### Examples

  Below implements neural network regression with heteroskedastic noise,
  noise contrastive priors, and being Bayesian only at the mean's output layer.

  ```python
  from tensor2tensor.layers import bayes

  batch_size, dataset_size = 128, 1000
  features, labels = get_some_dataset()

  inputs = keras.Input(shape=(25,))
  x = bayes.NCPNormalPerturb()(inputs)  # double input batch
  x = layers.Dense(64, activation='relu')(x)
  x = layers.Dense(64, activation='relu')(x)
  means = bayes.DenseVariationalDropout(1, activation=None)(x)  # get mean dist.
  means = bayes.NCPNormalOutput(labels)(means)  # halve input batch
  stddevs = tf.keras.layers.Dense(1, activation='softplus')(x[:batch_size])
  outputs = tf.keras.layers.Lambda(lambda x: ed.Normal(x[0], x[1]))([means,
                                                                     stddevs])
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  predictions = model(features)
  loss = tf.reduce_mean(predictions.distribution.log_prob(labels))
  loss += model.losses[0] / dataset_size  # KL regularizer for output layer
  loss += model.losses[-1]
  train_op = tf.train.AdamOptimizer(0.1).minimize(loss)
  ```

  The network applies `bayes.NCPNormalPerturb()` to double the input batch
  size and add Gaussian noise to the second half; then feedforward layers; then
  `bayes.DenseVariational` to be Bayesian about the output density's mean; then
  `bayes.NCPNormalOutput` centered at the labels to revert to the batch size
  and compute a loss on the second half; then parameterize the output density's
  standard deviations; then compute the total loss function as the sum of the
  model's negative log-likelihood, KL divergence for the Bayesian mean layer,
  and NCP loss.
  """

  def __init__(self, mean=0., stddev=1., **kwargs):
    self.mean = mean
    self.stddev = stddev
    super(NCPNormalOutput, self).__init__(**kwargs)

  def call(self, inputs):
    if not isinstance(inputs, ed.RandomVariable):
      # Default to a unit normal, i.e., derived from mean squared error loss.
      inputs = ed.Normal(loc=inputs, scale=1.)
    batch_size = tf.shape(inputs)[0] // 2
    # TODO(trandustin): Depend on github's ed2 for indexing RVs. This is a hack.
    # _, _ = inputs[:batch_size], inputs[batch_size:]
    original_inputs = ed.RandomVariable(inputs.distribution[:batch_size],
                                        value=inputs.value[:batch_size])
    perturbed_inputs = ed.RandomVariable(inputs.distribution[batch_size:],
                                         value=inputs.value[batch_size:])
    loss = tf.reduce_sum(
        tfp.distributions.Normal(self.mean, self.stddev).kl_divergence(
            perturbed_inputs.distribution)) / tf.to_float(batch_size)
    self.add_loss(loss)
    return original_inputs


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
