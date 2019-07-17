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

"""Trajectory manages a sequence of TimeSteps.

BatchTrajectory manages a batch of trajectories, also keeping account of
completed trajectories.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensor2tensor.envs import time_step


class Trajectory(object):
  """Basically a list of TimeSteps with convenience methods."""

  def __init__(self, time_steps=None):
    # Contains a list of time steps.
    if time_steps is None:
      self._time_steps = []
    else:
      self._time_steps = time_steps

  def __str__(self):
    if not self.time_steps:
      return "Trajectory[]"
    return "Trajectory[{}]".format(", ".join(str(ts) for ts in self.time_steps))

  def add_time_step(self, **create_time_step_kwargs):
    """Creates a time-step and appends it to the list.

    Args:
      **create_time_step_kwargs: Forwarded to
        time_step.TimeStep.create_time_step.
    """
    ts = time_step.TimeStep.create_time_step(**create_time_step_kwargs)
    assert isinstance(ts, time_step.TimeStep)
    self._time_steps.append(ts)

  def change_last_time_step(self, **replace_time_step_kwargs):
    """Replace the last time-steps with the given kwargs."""

    # Pre-conditions: self._time_steps shouldn't be empty.
    assert self._time_steps
    self._time_steps[-1] = self._time_steps[-1].replace(
        **replace_time_step_kwargs)

  def truncate(self, num_to_keep=1):
    """Truncate trajectories, keeping the last `num_to_keep` time-steps."""

    # We return `ts_copy` back to the truncator.
    ts_copy = self._time_steps[:]

    # We keep the last few observations.
    self._time_steps = self._time_steps[-num_to_keep:]

    # NOTE: We will need to set the rewards to 0, to eliminate double counting.
    for i in range(self.num_time_steps):
      self._time_steps[i] = self._time_steps[i].replace(
          raw_reward=0, processed_reward=0)

    return Trajectory(time_steps=ts_copy)

  @property
  def last_time_step(self):
    # Pre-conditions: self._time_steps shouldn't be empty.
    assert self._time_steps
    return self._time_steps[-1]

  @property
  def num_time_steps(self):
    return len(self._time_steps)

  @property
  def is_active(self):
    return bool(self.num_time_steps)

  @property
  def time_steps(self):
    return self._time_steps

  @property
  def done(self):
    return self.is_active and self.last_time_step.done

  # TODO(afrozm): Add discounting and rewards-to-go when it makes sense.
  @property
  def reward(self):
    """Returns a tuple of sum of raw and processed rewards."""
    raw_rewards, processed_rewards = 0, 0
    for ts in self.time_steps:
      # NOTE: raw_reward and processed_reward are None for the first time-step.
      if ts.raw_reward is not None:
        raw_rewards += ts.raw_reward
      if ts.processed_reward is not None:
        processed_rewards += ts.processed_reward
    return raw_rewards, processed_rewards

  @property
  def observations_np(self):
    return np.stack([ts.observation for ts in self.time_steps])

  def last_n_observations_np(self, n=None):
    if n is not None:
      n = -n  # pylint: disable=invalid-unary-operand-type
    return np.stack([ts.observation for ts in self.time_steps[n:]])

  @property
  def actions_np(self):
    # The last action is None, so let's skip it.
    return np.stack([ts.action for ts in self.time_steps[:-1]])

  @property
  def info_np(self):
    if not self.time_steps or not self.time_steps[0].info:
      return None
    info_np_dict = {}
    for info_key in self.time_steps[0].info:
      # Same as actions, the last info is missing, so we skip it.
      info_np_dict[info_key] = np.stack(
          [ts.info[info_key] for ts in self.time_steps[:-1]])
    return info_np_dict

  @property
  def rewards_np(self):
    # The first reward is None, so let's skip it.
    return np.stack([ts.processed_reward for ts in self.time_steps[1:]])

  @property
  def raw_rewards_np(self):
    return np.stack([ts.raw_reward for ts in self.time_steps[1:]])

  @property
  def as_numpy(self):
    # TODO(afrozm): Return a named tuple here, ex: TrajectoryArrays
    return (self.observations_np, self.actions_np, self.rewards_np,
            self.raw_rewards_np, self.info_np)


class BatchTrajectory(object):
  """Basically a batch of active trajectories and a list of completed ones."""

  def __init__(self, batch_size=1):
    self.batch_size = batch_size

    # Stores trajectories that are currently active, i.e. aren't done or reset.
    self._trajectories = [Trajectory() for _ in range(self.batch_size)]

    # Stores trajectories that are completed.
    # NOTE: We don't track the index this came from, as it's not needed, right?
    self._completed_trajectories = []

  def reset_batch_trajectories(self):
    self.__init__(batch_size=self.batch_size)

  def __str__(self):
    string = "BatchTrajectory["
    for i, t in enumerate(self.trajectories):
      string += "Trajectory {} = {}\n".format(i, str(t))
    for i, t in enumerate(self.completed_trajectories):
      string += "Completed Trajectory {} = {}\n".format(i, str(t))
    return string + "]"

  @property
  def trajectories(self):
    return self._trajectories

  @property
  def completed_trajectories(self):
    return self._completed_trajectories

  def clear_completed_trajectories(self, num=None):
    """Clear the first `num` completed trajectories, or all if num is None."""
    if num is None:
      self._completed_trajectories = []
    else:
      self._completed_trajectories = self._completed_trajectories[num:]

  def _complete_trajectory(self, trajectory, index):
    """Completes the given trajectory at the given index."""

    assert isinstance(trajectory, Trajectory)

    # This *should* be the case.
    assert trajectory.last_time_step.action is None

    # Add to completed trajectories.
    self._completed_trajectories.append(trajectory)

    # Make a new one to replace it.
    self._trajectories[index] = Trajectory()

  def truncate_trajectories(self, indices, num_to_keep=1):
    """Truncate trajectories at specified indices.

     This puts the truncated trajectories in the completed list and makes new
     trajectories with the observation from the trajectory that was truncated at
     the same index.

    Args:
        indices: iterable with the indices to truncate.
        num_to_keep: int, number of last time-steps to keep while truncating.
    """
    for index in indices:
      trajectory = self._trajectories[index]
      assert trajectory.is_active, "Trajectory to truncate can't be inactive."

      # Now `trajectory` just consists of the last `num_to_keep` observations
      # and actions. Rewards are zeroed out.
      # The old data is placed in `old_trajectory`.
      old_trajectory = trajectory.truncate(num_to_keep=num_to_keep)

      # We put the old data in _completed_trajectories.
      self._completed_trajectories.append(old_trajectory)

  def reset(self, indices, observations):
    """Resets trajectories at given indices and populates observations.

    Reset can either be called right at the beginning, when there are no
    time-steps, or to reset a currently active trajectory.

    If resetting a currently active trajectory then we save it in
    self._completed_trajectories.

    Args:
      indices: 1-D np.ndarray stating the indices to reset.
      observations: np.ndarray of shape (indices len, obs.shape) of observations
    """

    # Pre-conditions: indices, observations are np arrays.
    #               : indices is one-dimensional.
    #               : their first dimension (batch) is the same.
    assert isinstance(indices, np.ndarray)
    assert len(indices.shape) == 1
    assert isinstance(observations, np.ndarray)
    assert indices.shape[0] == observations.shape[0]

    for index, observation in zip(indices, observations):
      trajectory = self._trajectories[index]

      # Are we starting a new trajectory at the given index?
      if not trajectory.is_active:
        # Then create a new time-step here with the given observation.
        trajectory.add_time_step(observation=observation)
        # That's all we need to do here.
        continue

      # If however we are resetting a currently active trajectory then we need
      # to put that in self._completed_trajectories and make a new trajectory
      # with the current observation.

      # TODO(afrozm): Should we mark these are done? Or is the done=False and
      # this being the last time-step in the trajectory good enough to recognize
      # that this was reset?

      # Mark trajectory as completed and move into completed_trajectories.
      self._complete_trajectory(trajectory, index)

      # Put the observation in the newly created trajectory.
      # TODO(afrozm): Add 0 reward.
      self._trajectories[index].add_time_step(observation=observation)

  def complete_all_trajectories(self):
    """Essentially same as reset, but we don't have observations."""
    for index in range(self.batch_size):
      trajectory = self._trajectories[index]
      assert trajectory.is_active
      self._complete_trajectory(trajectory, index)

  def step(self, observations, raw_rewards, processed_rewards, dones, actions,
           infos=None):
    """Record the information obtained from taking a step in all envs.

    Records (observation, rewards, done) in a new time-step and actions in the
    current time-step.

    If any trajectory gets done, we move that trajectory to
    completed_trajectories.

    Args:
      observations: ndarray of first dimension self.batch_size, which has the
        observations after we've stepped, i.e. s_{t+1} where t is the current
        state.
      raw_rewards: ndarray of first dimension self.batch_size containing raw
        rewards i.e. r_{t+1}.
      processed_rewards: ndarray of first dimension self.batch_size containing
        processed rewards. i.e. r_{t+1}
      dones: ndarray of first dimension self.batch_size, containing true at an
        index if that env is done, i.e. d_{t+1}
      actions: ndarray of first dimension self.batch_size, containing actions
        applied at the current time-step, which leads to the observations
        rewards and done at the next time-step, i.e. a_t
      infos: (optional) a dictionary of keys and values, where all the values
        have the first dimension as self.batch_size.
    """
    # Pre-conditions
    assert isinstance(observations, np.ndarray)
    assert isinstance(raw_rewards, np.ndarray)
    assert isinstance(processed_rewards, np.ndarray)
    assert isinstance(dones, np.ndarray)
    assert isinstance(actions, np.ndarray)
    if infos:
      assert isinstance(infos, dict)

    # We assume that we step in all envs, i.e. not like reset where we can reset
    # some envs and not others.
    assert self.batch_size == observations.shape[0]
    assert self.batch_size == raw_rewards.shape[0]
    assert self.batch_size == processed_rewards.shape[0]
    assert self.batch_size == dones.shape[0]
    assert self.batch_size == actions.shape[0]
    if infos:
      for _, v in infos.items():
        assert self.batch_size == len(v)

    def extract_info_at_index(infos, index):
      if not infos:
        return None
      return {k: v[index] for k, v in infos.items()}

    for index in range(self.batch_size):
      trajectory = self._trajectories[index]

      # NOTE: If the trajectory isn't active, that means it doesn't have any
      # time-steps in it, but we are in step, so the assumption is that it has
      # a prior observation from which we are stepping away from.

      # TODO(afrozm): Let's re-visit this if it becomes too restrictive.
      assert trajectory.is_active

      # To this trajectory's last time-step, set actions.
      trajectory.change_last_time_step(
          action=actions[index],
          info=extract_info_at_index(infos, index))

      # Create a new time-step to add observation, done & rewards (no actions).
      trajectory.add_time_step(
          observation=observations[index],
          done=dones[index],
          raw_reward=raw_rewards[index],
          processed_reward=processed_rewards[index])

      # If the trajectory is completed, i.e. dones[index] == True, then we
      # account for it right-away.
      if dones[index]:
        self._complete_trajectory(trajectory, index)

        # NOTE: The new trajectory at `index` is going to be in-active and
        # `reset` should be called on it.
        assert not self._trajectories[index].is_active

  @staticmethod
  def _trajectory_lengths(trajectories):
    return np.array([t.num_time_steps for t in trajectories])

  @property
  def num_completed_time_steps(self):
    """Returns the number of time-steps in completed trajectories."""

    return sum(BatchTrajectory._trajectory_lengths(self.completed_trajectories))

  @property
  def num_time_steps(self):
    """Returns the number of time-steps in completed and incomplete trajectories."""

    num_time_steps = sum(BatchTrajectory._trajectory_lengths(self.trajectories))
    return num_time_steps + self.num_completed_time_steps

  @property
  def trajectory_lengths(self):
    return BatchTrajectory._trajectory_lengths(self.trajectories)

  @property
  def num_completed_trajectories(self):
    """Returns the number of completed trajectories."""
    return len(self.completed_trajectories)

  # TODO(afrozm): Take in an already padded observation ndarray and just append
  # the last time-step and adding more padding if needed.
  def observations_np(self, boundary=20, len_history_for_policy=20):
    """Pads the observations in all the trajectories and returns them.

    Args:
      boundary: integer, Observations will be padded to (n * boundary) + 1 where
        n is an integer.
      len_history_for_policy: int, For each trajectory return only the last
        `len_history_for_policy` observations. Set to None for all the
        observations.

    Returns:
      padded_observations: (self.batch_size, n * boundary + 1) + OBS
    """
    list_observations_np_ts = [
        t.last_n_observations_np(n=len_history_for_policy)
        for t in self.trajectories
    ]
    # Every element in `list_observations_np_ts` is shaped (t,) + OBS
    OBS = list_observations_np_ts[0].shape[1:]  # pylint: disable=invalid-name

    trajectory_lengths = np.stack(
        [obs.shape[0] for obs in list_observations_np_ts])

    t_max = max(trajectory_lengths)
    # t_max is rounded to the next multiple of `boundary`
    boundary = int(boundary)
    bucket_length = boundary * int(np.ceil(float(t_max) / boundary))

    def padding_config(obs):
      # We're padding the first axis only, since that is the time-step.
      num_to_pad = bucket_length + 1 - obs.shape[0]
      return [(0, num_to_pad)] + [(0, 0)] * len(OBS)

    return np.stack([
        np.pad(obs, padding_config(obs), "constant")
        for obs in list_observations_np_ts
    ]), trajectory_lengths
