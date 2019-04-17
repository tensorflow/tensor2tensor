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

"""Regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from tensorflow_probability import edward2 as ed


class LogUniformKLDivergence(tf.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to the log-uniform distribution."""

  def __call__(self, x):
    """Computes regularization given an ed.Normal random variable as input."""
    if not isinstance(x, ed.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable (for correct math, '
                       'an ed.Normal random variable).')
    # Clip magnitude of dropout rate, where we get the dropout rate alpha from
    # the additive parameterization (Molchanov et al., 2017): for weight ~
    # Normal(mu, sigma**2), the variance `sigma**2 = alpha * mu**2`.
    mean = x.distribution.mean()
    log_variance = tf.log(x.distribution.variance())
    log_alpha = log_variance - tf.log(tf.square(mean) +
                                      tf.keras.backend.epsilon())
    log_alpha = tf.clip_by_value(log_alpha, -8., 8.)

    # Set magic numbers for cubic polynomial approx. (Molchanov et al., 2017).
    k1 = 0.63576
    k2 = 1.8732
    k3 = 1.48695
    c = -k1
    output = tf.reduce_sum(k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) +
                           -0.5 * tf.log1p(tf.exp(-log_alpha)) + c)
    return output

  def get_config(self):
    return {}


class NormalKLDivergence(tf.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to the normal distribution."""

  def __init__(self, mean=0., stddev=1.):
    """Constructs regularizer where default is a KL towards the std normal."""
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


# Compatibility aliases, following tf.keras

# pylint: disable=invalid-name
log_uniform_kl_divergence = LogUniformKLDivergence
normal_kl_divergence = NormalKLDivergence
# pylint: enable=invalid-name

# Utility functions, following tf.keras


def serialize(initializer):
  return tf.keras.utils.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return tf.keras.utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='regularizers')


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
