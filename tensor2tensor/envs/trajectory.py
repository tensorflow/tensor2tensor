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

  def __init__(self):
    # Contains a list of time steps.
    self._time_steps = []

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

  def _complete_trajectory(self, trajectory, index):
    """Completes the given trajectory at the given index."""

    assert isinstance(trajectory, Trajectory)

    # This *should* be the case.
    assert trajectory.last_time_step.action is None

    # Add to completed trajectories.
    self._completed_trajectories.append(trajectory)

    # Make a new one to replace it.
    self._trajectories[index] = Trajectory()

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

  def step(self, observations, raw_rewards, processed_rewards, dones, actions):
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
    """
    # Pre-conditions
    assert isinstance(observations, np.ndarray)
    assert isinstance(raw_rewards, np.ndarray)
    assert isinstance(processed_rewards, np.ndarray)
    assert isinstance(dones, np.ndarray)
    assert isinstance(actions, np.ndarray)

    # We assume that we step in all envs, i.e. not like reset where we can reset
    # some envs and not others.
    assert self.batch_size == observations.shape[0]
    assert self.batch_size == raw_rewards.shape[0]
    assert self.batch_size == processed_rewards.shape[0]
    assert self.batch_size == dones.shape[0]
    assert self.batch_size == actions.shape[0]

    for index in range(self.batch_size):
      trajectory = self._trajectories[index]

      # NOTE: If the trajectory isn't active, that means it doesn't have any
      # time-steps in it, but we are in step, so the assumption is that it has
      # a prior observation from which we are stepping away from.

      # TODO(afrozm): Let's re-visit this if it becomes too restrictive.
      assert trajectory.is_active

      # To this trajectory's last time-step, set actions.
      trajectory.change_last_time_step(action=actions[index])

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

  @property
  def num_completed_time_steps(self):
    """Returns the number of time-steps in completed trajectories."""

    return sum(t.num_time_steps for t in self.completed_trajectories)

  @property
  def num_time_steps(self):
    """Returns the number of time-steps in completed and incomplete trajectories."""

    num_time_steps = sum(t.num_time_steps for t in self.trajectories)
    return num_time_steps + self.num_completed_time_steps
