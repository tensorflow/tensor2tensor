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

"""Tests for tensor2tensor.trax.rl.serialization_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import gym
import numpy as np

from tensor2tensor.trax.rl import serialization_utils
from tensor2tensor.trax.rl import space_serializer
from tensorflow import test


class SerializationTest(test.TestCase):

  def setUp(self):
    super(SerializationTest, self).setUp()
    self._serializer = space_serializer.create(
        gym.spaces.Discrete(2), vocab_size=2
    )
    self._repr_length = 100
    self._serialization_utils_kwargs = {
        "observation_serializer": self._serializer,
        "action_serializer": self._serializer,
        "representation_length": self._repr_length,
    }

  def test_serializes_observations_and_actions(self):
    (reprs, mask) = serialization_utils.serialize_observations_and_actions(
        observations=np.array([[0, 1]]),
        actions=np.array([[0]]),
        mask=np.array([[1]]),
        **self._serialization_utils_kwargs
    )
    self.assertEqual(reprs.shape, (1, self._repr_length))
    self.assertEqual(mask.shape, (1, self._repr_length))
    self.assertGreater(np.sum(mask), 0)
    self.assertEqual(np.max(mask), 1)

  def test_masks_length(self):
    (reprs, mask) = serialization_utils.serialize_observations_and_actions(
        observations=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]]),
        actions=np.array([[0, 0], [0, 1], [0, 0]]),
        mask=np.array([[1, 0], [1, 1], [1, 1]]),
        **self._serialization_utils_kwargs
    )
    # Trajectories 1 and 2 are longer than 0.
    self.assertGreater(np.sum(mask[1]), np.sum(mask[0]))
    self.assertGreater(np.sum(mask[2]), np.sum(mask[0]))
    # Trajectory 0 is a common prefix of 1 and 2. 1 and 2 are different.
    np.testing.assert_array_equal(reprs[0] * mask[0], reprs[1] * mask[0])
    np.testing.assert_array_equal(reprs[0] * mask[0], reprs[2] * mask[0])
    self.assertFalse(np.array_equal(reprs[1] * mask[1], reprs[2] * mask[2]))
    # Trajectories should be padded with 0s.
    np.testing.assert_array_equal(
        reprs * (1 - mask), np.zeros((3, self._repr_length))
    )

  def test_observation_and_action_masks_are_valid_and_complementary(self):
    obs_mask = serialization_utils.observation_mask(
        **self._serialization_utils_kwargs
    )
    self.assertEqual(obs_mask.shape, (self._repr_length,))
    self.assertEqual(np.min(obs_mask), 0)
    self.assertEqual(np.max(obs_mask), 1)

    act_mask = serialization_utils.action_mask(
        **self._serialization_utils_kwargs
    )
    self.assertEqual(act_mask.shape, (self._repr_length,))
    self.assertEqual(np.min(act_mask), 0)
    self.assertEqual(np.max(act_mask), 1)

    np.testing.assert_array_equal(
        obs_mask + act_mask, np.ones(self._repr_length)
    )

  def test_masks_observations(self):
    (reprs, _) = serialization_utils.serialize_observations_and_actions(
        # Observations are different, actions are the same.
        observations=np.array([[0, 1], [1, 1]]),
        actions=np.array([[0], [0]]),
        mask=np.array([[1], [1]]),
        **self._serialization_utils_kwargs
    )
    obs_mask = serialization_utils.observation_mask(
        **self._serialization_utils_kwargs
    )
    act_mask = serialization_utils.action_mask(
        **self._serialization_utils_kwargs
    )

    self.assertFalse(np.array_equal(reprs[0] * obs_mask, reprs[1] * obs_mask))
    np.testing.assert_array_equal(reprs[0] * act_mask, reprs[1] * act_mask)

  def test_masks_actions(self):
    (reprs, _) = serialization_utils.serialize_observations_and_actions(
        # Observations are the same, actions are different.
        observations=np.array([[0, 1], [0, 1]]),
        actions=np.array([[0], [1]]),
        mask=np.array([[1], [1]]),
        **self._serialization_utils_kwargs
    )
    obs_mask = serialization_utils.observation_mask(
        **self._serialization_utils_kwargs
    )
    act_mask = serialization_utils.action_mask(
        **self._serialization_utils_kwargs
    )

    np.testing.assert_array_equal(reprs[0] * obs_mask, reprs[1] * obs_mask)
    self.assertFalse(np.array_equal(reprs[0] * act_mask, reprs[1] * act_mask))

  def test_significance_map(self):
    gin.bind_parameter("BoxSpaceSerializer.precision", 3)
    significance_map = serialization_utils.significance_map(
        observation_serializer=space_serializer.create(
            gym.spaces.Box(low=0, high=1, shape=(2,)), vocab_size=2
        ),
        action_serializer=space_serializer.create(
            gym.spaces.MultiDiscrete(nvec=[2, 2]), vocab_size=2
        ),
        representation_length=20,
    )
    np.testing.assert_array_equal(
        significance_map,
        # obs1, act1, obs2, act2, obs3 cut after 4th symbol.
        [0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 2, 0],
    )

  def test_rewards_to_actions_map(self):
    rewards = np.array([1, 2, 3])
    r2a_map = serialization_utils.rewards_to_actions_map(
        observation_serializer=space_serializer.create(
            gym.spaces.MultiDiscrete(nvec=[2, 2, 2]), vocab_size=2
        ),
        action_serializer=space_serializer.create(
            gym.spaces.MultiDiscrete(nvec=[2, 2]), vocab_size=2
        ),
        n_timesteps=len(rewards),
        representation_length=16,
    )
    broadcast_rewards = np.dot(rewards, r2a_map)
    np.testing.assert_array_equal(
        broadcast_rewards,
        # obs1, act1, obs2, act2, obs3 cut after 1st symbol.
        [0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 0],
    )


if __name__ == "__main__":
  test.main()
