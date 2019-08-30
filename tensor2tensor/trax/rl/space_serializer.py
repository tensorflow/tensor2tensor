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

"""Serialization of elements of Gym spaces into discrete sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from absl import logging
import gin
import gym
import numpy as np


class SpaceSerializer(object):
  """Base class for Gym space serializers.

  Attrs:
    space_type: (type) Gym space class that this SpaceSerializer corresponds
      to. Should be defined in subclasses.
    representation_length: (int) Number of symbols in the representation of
      every element of the space.
  """

  space_type = None
  representation_length = None

  def __init__(self, space, vocab_size):
    """Creates a SpaceSerializer.

    Subclasses should retain the signature.

    Args:
      space: (gym.Space) Gym space of type self.space_type.
      vocab_size: (int) Number of symbols in the vocabulary.
    """
    assert isinstance(space, self.space_type)
    self._space = space
    self._vocab_size = vocab_size

  def serialize(self, data):
    """Serializes a batch of space elements into a discrete sequences.

    Should be defined in subclasses.

    Args:
      data: A batch of batch_size elements of the Gym space to be serialized.

    Returns:
      int32 array of shape (batch_size, self.representation_length).
    """
    raise NotImplementedError

  def deserialize(self, representation):
    """Deserializes a batch of discrete sequences into space elements.

    Should be defined in subclasses.

    Args:
      representation: int32 Numpy array of shape
        (batch_size, self.representation_length) to be deserialized.

    Returns:
      A batch of batch_size deserialized elements of the Gym space.
    """
    raise NotImplementedError


def create(space, vocab_size):
  """Creates a SpaceSerializer for the given Gym space."""
  return {
      gym.spaces.Box: BoxSpaceSerializer,
      gym.spaces.Discrete: DiscreteSpaceSerializer,
  }[type(space)](space, vocab_size)


@gin.configurable(blacklist=["space", "vocab_size"])
class BoxSpaceSerializer(SpaceSerializer):
  """Serializer for gym.spaces.Box.

  Assumes that the space is bounded. Internally rescales it to the [0, 1]
  interval and uses a fixed-precision encoding.
  """

  space_type = gym.spaces.Box

  def __init__(self, space, vocab_size, precision=2, max_range=(-100.0, 100.0)):
    self._precision = precision

    # Some gym envs (e.g. CartPole) have unreasonably high bounds for
    # observations. We clip so we can represent them.
    bounded_space = copy.copy(space)
    (min_low, max_high) = max_range
    bounded_space.low = np.maximum(space.low, min_low)
    bounded_space.high = np.minimum(space.high, max_high)
    if (not np.allclose(bounded_space.low, space.low) or
        not np.allclose(bounded_space.high, space.high)):
      logging.warning(
          "Space limits %s, %s out of bounds %s. Clipping to %s, %s.",
          str(space.low), str(space.high), str(max_range),
          str(bounded_space.low), str(bounded_space.high)
      )

    super(BoxSpaceSerializer, self).__init__(bounded_space, vocab_size)

  def serialize(self, data):
    array = data
    batch_size = array.shape[0]
    array = (array - self._space.low) / (self._space.high - self._space.low)
    digits = []
    for digit_index in range(-1, -self._precision - 1, -1):
      threshold = self._vocab_size ** digit_index
      digit = np.array(array / threshold).astype(np.int32) % self._vocab_size
      digits.append(digit)
      array -= digit * threshold
    digits = np.stack(digits, axis=-1)
    return np.reshape(digits, (batch_size, -1))

  def deserialize(self, representation):
    digits = representation
    batch_size = digits.shape[0]
    digits = np.reshape(digits, (batch_size, -1, self._precision))
    array = np.zeros(digits.shape[:-1])
    for digit_index_in_seq in range(self._precision):
      digit_index = -digit_index_in_seq - 1
      array += self._vocab_size ** digit_index * digits[..., digit_index_in_seq]
    array = np.reshape(array, (batch_size,) + self._space.shape)
    return array * (self._space.high - self._space.low) + self._space.low

  @property
  def representation_length(self):
    return self._precision * self._space.low.size


class DiscreteSpaceSerializer(SpaceSerializer):
  """Serializer for gym.spaces.Discrete.

  Assumes that the size of the space fits in the number of symbols.
  """

  space_type = gym.spaces.Discrete
  representation_length = 1

  def __init__(self, space, vocab_size):
    super(DiscreteSpaceSerializer, self).__init__(space, vocab_size)
    assert space.n <= vocab_size, (
        "Discrete space size should fit in the number of symbols.")

  def serialize(self, data):
    return np.reshape(data, (-1, 1))

  def deserialize(self, representation):
    return np.reshape(representation, -1)
