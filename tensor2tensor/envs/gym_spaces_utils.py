# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Few utility functions to deal with gym spaces.

gym.spaces.Box and gym.spaces.Discrete are easiest to support.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Box
from gym.spaces import Discrete

import numpy as np
import tensorflow.compat.v1 as tf


def box_space_spec(box_space, tf_dtype):
  return tf.FixedLenFeature(box_space.shape, tf_dtype)


def discrete_space_spec(discrete_space, tf_dtype):
  del discrete_space  # this is not needed.
  return tf.FixedLenFeature((1,), tf_dtype)


def gym_space_spec(gym_space):
  """Returns a reading spec of a gym space.

  NOTE: Only implemented currently for Box and Discrete.

  Args:
    gym_space: instance of gym.spaces whose spec we want.

  Returns:
    Reading spec for that space.

  Raises:
    NotImplementedError: For spaces whose reading spec we haven't implemented.
  """
  # First try to determine the type.
  try:
    tf_dtype = tf.as_dtype(gym_space.dtype)
  except TypeError as e:
    tf.logging.error("Cannot convert space's type [%s] to tf.dtype",
                     gym_space.dtype)
    raise e

  # Now hand it over to the specialized functions.
  if isinstance(gym_space, Box):
    return box_space_spec(gym_space, tf_dtype)
  elif isinstance(gym_space, Discrete):
    return discrete_space_spec(gym_space, tf_dtype)
  else:
    raise NotImplementedError


def gym_space_encode(gym_space, observation):
  # We should return something that generator_utils.to_example can consume.
  if isinstance(gym_space, Discrete):
    return [observation]

  if isinstance(gym_space, Box):
    return observation.reshape(-1).tolist()

  raise NotImplementedError


def cardinality(gym_space):
  """Number of elements that can be represented by the space.

  Makes the most sense for Discrete or Box type with integral dtype, ex: number
  of actions in an action space.

  Args:
    gym_space: The gym space.

  Returns:
    np.int64 number of observations that can be represented by this space, or
    returns None when this doesn't make sense, i.e. float boxes etc.

  Raises:
    NotImplementedError when a space's cardinality makes sense but we haven't
    implemented it.
  """

  if (gym_space.dtype == np.float32) or (gym_space.dtype == np.float64):
    tf.logging.warn("Returning None for a float gym space's cardinality: %s",
                    gym_space)
    return None

  if isinstance(gym_space, Discrete):
    return gym_space.n

  if isinstance(gym_space, Box):
    # Construct a box with all possible values in this box and take a product.
    return np.prod(gym_space.high - gym_space.low + 1)

  raise NotImplementedError
