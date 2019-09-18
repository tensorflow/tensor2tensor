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

"""Utilities for serializing trajectories into discrete sequences."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def serialize_observations_and_actions(
    observations,
    actions,
    mask,
    observation_serializer,
    action_serializer,
    representation_length,
):
  """Serializes observations and actions into a discrete sequence.

  Args:
    observations: Array (B, T + 1, ...), of observations, where B is the batch
      size and T is the number of timesteps excluding the last observation.
    actions: Array (B, T, ...) of actions.
    mask: Binary array (B, T) indicating where each sequence ends (1s while
      it continues).
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    representation_length: Number of symbols in the serialized sequence. The
      sequence is padded up to this number.
  Returns:
    Pair (representation, mask), where representation is the serialized sequence
    of shape (B, R) where R = representation_length, and mask is a binary array
    of shape (B, R) indicating where each sequence ends.
  """
  (batch_size, n_timesteps) = actions.shape[:2]
  assert observations.shape[:2] == (batch_size, n_timesteps + 1)
  assert mask.shape == (batch_size, n_timesteps)

  reprs = []
  for t in range(n_timesteps):
    reprs.append(observation_serializer.serialize(observations[:, t, ...]))
    reprs.append(action_serializer.serialize(actions[:, t, ...]))
  reprs.append(observation_serializer.serialize(observations[:, -1, ...]))
  reprs = np.concatenate(reprs, axis=1)
  assert reprs.shape[1] <= representation_length
  reprs = np.pad(
      reprs,
      pad_width=((0, 0), (0, representation_length - reprs.shape[1])),
      mode="constant",
  )

  obs_repr_length = observation_serializer.representation_length
  act_repr_length = action_serializer.representation_length
  step_repr_length = obs_repr_length + act_repr_length
  seq_lengths = np.sum(mask, axis=1).astype(np.int32)
  repr_lengths = seq_lengths * step_repr_length + obs_repr_length
  repr_mask = np.zeros((batch_size, representation_length), dtype=np.int32)
  for (i, repr_length) in enumerate(repr_lengths):
    repr_mask[i, :repr_length] = 1

  return (reprs, repr_mask)


def observation_mask(
    observation_serializer, action_serializer, representation_length
):
  """Calculates an observation mask for a serialized sequence.

  Args:
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    representation_length: Number of symbols in the serialized sequence. The
      mask is padded up to this number.

  Returns:
    Binary mask indicating which symbols in the representation correspond to
    observations.
  """
  mask = np.zeros(representation_length, dtype=np.int32)
  obs_repr_length = observation_serializer.representation_length
  step_repr_length = obs_repr_length + action_serializer.representation_length
  for step_start_index in range(0, representation_length, step_repr_length):
    mask[step_start_index:(step_start_index + obs_repr_length)] = 1
  return mask


def action_mask(
    observation_serializer, action_serializer, representation_length
):
  """Calculates an action mask for a serialized sequence.

  Args:
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    representation_length: Number of symbols in the serialized sequence. The
      mask is padded up to this number.

  Returns:
    Binary mask indicating which symbols in the representation correspond to
    actions.
  """
  return 1 - observation_mask(
      observation_serializer, action_serializer, representation_length
  )


def significance_map(
    observation_serializer, action_serializer, representation_length
):
  """Calculates a significance map for the entire serialized sequence.

  See SpaceSerializer.significance_map.

  Args:
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    representation_length: Number of symbols in the serialized sequence. The
      significance map is padded up to this number.

  Returns:
    Significance map for the entire serialized sequence.
  """
  sig_map = np.zeros(representation_length, dtype=np.int32)
  obs_repr_length = observation_serializer.representation_length
  act_repr_length = action_serializer.representation_length
  step_repr_length = obs_repr_length + act_repr_length
  for step_start_index in range(0, representation_length, step_repr_length):
    act_start_index = step_start_index + obs_repr_length
    step_end_index = step_start_index + step_repr_length
    limit = representation_length - step_start_index
    sig_map[step_start_index:act_start_index] = (
        observation_serializer.significance_map[:limit]
    )
    limit = representation_length - act_start_index
    sig_map[act_start_index:step_end_index] = (
        action_serializer.significance_map[:limit]
    )
  return sig_map


def rewards_to_actions_map(
    observation_serializer,
    action_serializer,
    n_timesteps,
    representation_length,
):
  """Calculates a mapping between the rewards and the serialized sequence.

  Used to broadcast advantages over the log-probabilities of corresponding
  actions.

  Args:
    observation_serializer: SpaceSerializer for observations.
    action_serializer: SpaceSerializer for actions.
    n_timesteps: Number of timesteps (length of the reward sequence).
    representation_length: Number of symbols in the serialized sequence.

  Returns:
    Array (T, R) translating from the reward sequence to actions in the
    representation.
  """
  r2a_map = np.zeros((n_timesteps, representation_length))
  obs_repr_length = observation_serializer.representation_length
  act_repr_length = action_serializer.representation_length
  step_repr_length = obs_repr_length + act_repr_length
  for t in range(n_timesteps):
    act_start_index = t * step_repr_length + obs_repr_length
    r2a_map[t, act_start_index:(act_start_index + act_repr_length)] = 1
  return r2a_map
