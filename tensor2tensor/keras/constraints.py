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

"""Constraints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf


class Positive(tf.keras.constraints.Constraint):
  """Positive constraint."""

  def __init__(self, epsilon=tf.keras.backend.epsilon()):
    self.epsilon = epsilon

  def __call__(self, w):
    return tf.maximum(w, self.epsilon)

  def get_config(self):
    return {'epsilon': self.epsilon}


# Compatibility aliases, following tf.keras


positive = Positive  # pylint: disable=invalid-name

# Utility functions, following tf.keras


def serialize(initializer):
  return tf.keras.utils.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return tf.keras.utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='constraints')


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
