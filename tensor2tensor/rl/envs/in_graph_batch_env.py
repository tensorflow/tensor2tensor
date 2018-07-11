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
"""Batch of environments inside the TensorFlow graph."""

# The code was based on Danijar Hafner's code from tf.agents:
# https://github.com/tensorflow/agents/blob/master/agents/tools/in_graph_batch_env.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class InGraphBatchEnv(object):
  """Abstract class for batch of environments inside the TensorFlow graph.
  """

  def __getattr__(self, name):
    """Forward unimplemented attributes to one of the original environments.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name in one of the original environments.
    """
    return getattr(self._batch_env, name)

  def __len__(self):
    """Number of combined environments."""
    return len(self._batch_env)

  def __getitem__(self, index):
    """Access an underlying environment by index."""
    return self._batch_env[index]

  def simulate(self, action):
    """Step the batch of environments.

    The results of the step can be accessed from the variables defined below.

    Args:
      action: Tensor holding the batch of actions to apply.

    Returns:
      Operation.
    """
    raise NotImplementedError

  def reset(self, indices=None):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset.

    Returns:
      Batch tensor of the new observations.
    """
    return tf.cond(
        tf.cast(tf.reduce_sum(indices + 1), tf.bool),
        lambda: self._reset_non_empty(indices), lambda: 0.0)

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  def close(self):
    """Send close messages to the external process and join them."""
    self._batch_env.close()
