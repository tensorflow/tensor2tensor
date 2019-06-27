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

"""Utilities to use TF v1 layers with Keras and TF v2 easily."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FunctionLayer(tf.compat.v2.keras.layers.Layer):
  """Layer made of a function. Stores all variables."""

  def __init__(self, function, name=None):
    if name is None:
      name = function.__name__
    super(FunctionLayer, self).__init__(name=name)
    self._template = tf.compat.v1.make_template(name, function)
    self._was_called = False

  @property
  def losses(self):
    return []

  def compute_mask(self, inputs, previous_mask):
    return previous_mask

  @tf.function
  def _template_call(self, *args, **kwargs):
    """Call to template but made in graph mode for better speed."""
    return self._template(*args, **kwargs)

  def call(self, *args, **kwargs):
    if not self._was_called:  # Create variables at first call.
      return self._template(*args, **kwargs)
    return self._template_call(*args, **kwargs)
