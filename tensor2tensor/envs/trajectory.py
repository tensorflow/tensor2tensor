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

"""Trajectory manages a sequence of TimeSteps.

BatchTrajectory manages a batch of trajectories, also keeping account of
completed trajectories.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import re
import sys
import time
from absl import logging
import cloudpickle
import numpy as np
from tensor2tensor.envs import time_step
import tensorflow.compat.v1 as tf

TRAJECTORY_FILE_FORMAT = r"trajectory_epoch_{epoch}_env_id_{env_id}_temperature_{temperature}_r_{r}.pkl"


def get_pickle_module():
  if sys.version_info[0] < 3:
    return cloudpickle
  return pickle


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

  def __init__(self,
               batch_size=1,
               trajectories=None,
               completed_trajectories=None):
    self.batch_size = batch_size

    # Stores trajectories that are currently active, i.e. aren't done or reset.
    self._trajectories = trajectories or [
        Trajectory() for _ in range(self.batch_size)
    ]

    # Stores trajectories that are completed.
    # NOTE: We don't track the index this came from, as it's not needed, right?
    self._completed_trajectories = completed_trajectories or []

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
      # TODO(pkozakowski): This assertion breaks something in SimPLe trajectory
      # collection code - we're probably doing something wrong there. Commenting
      # out the assertion as a temporary measure.
      # assert trajectory.is_active
      if trajectory.is_active:
        self._complete_trajectory(trajectory, index)

  def step(self,
           observations,
           raw_rewards,
           processed_rewards,
           dones,
           actions,
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
          action=actions[index], info=extract_info_at_index(infos, index))

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

  @staticmethod
  def parse_trajectory_file_name(trajectory_file_name):
    """Parse out the trajectory file's groups and return to caller."""
    base_trajectory_file_name = os.path.basename(trajectory_file_name)
    trajectory_file_regexp = TRAJECTORY_FILE_FORMAT.format(
        epoch="(.*)",
        env_id="(.*)",
        temperature="(.*)",
        r="(.*)",
    )
    compiled_regexp = re.compile(trajectory_file_regexp)
    r = compiled_regexp.match(base_trajectory_file_name)
    if not r:
      return None
    g = r.groups()
    if len(g) is not compiled_regexp.groups:
      return None
    # epoch, env_id, temp, random string
    try:
      epoch = int(g[0])
      env_id = int(g[1])
      temperature = float(g[2])
      random_string = g[3]
    except ValueError:
      logging.error("Trajectory file name isn't parseable: %s",
                    base_trajectory_file_name)
      return None
    return epoch, env_id, temperature, random_string

  @staticmethod
  def load_from_directory(trajectory_dir,
                          epoch=None,
                          temperature=None,
                          n_trajectories=None,
                          up_sample=False,
                          sleep_time_secs=0.1,
                          max_tries=100,
                          wait_forever=False):
    """Load trajectories from specified dir and epoch.

    Args:
      trajectory_dir: (string) directory to find trajectories.
      epoch: (int) epoch for which to load trajectories, if None we don't filter
        on an epoch.
      temperature: (float) this is used to filter the trajectory files, if None
        we don't filter on temperature.
      n_trajectories: (int) This is the batch size of the returned
        BatchTrajectory object if one is returned. If set to None, then the
        number of trajectories becomes the batch size. If set to some number,
        then we wait for those many trajectory files to be available.
      up_sample: (bool) If there are fewer than required (n_trajectories) number
        of incomplete trajectories, then we upsample to make up the numbers.
      sleep_time_secs: (float) Sleep time, to wait for min_trajectories. We
        exponentially back-off this up till a maximum of 10 seconds.
      max_tries: (int) The number of tries to get min_trajectories trajectories.
      wait_forever: (bool) If true, overrides max_tries and waits forever.

    Returns:
      A BatchTrajectory object with all the constraints satisfied or None.
    """

    # Modify the format to get a glob with desired epoch and temperature.
    trajectory_file_glob = TRAJECTORY_FILE_FORMAT.format(
        epoch=epoch if epoch is not None else "*",
        env_id="*",
        temperature=temperature if temperature is not None else "*",
        r="*",
    )

    trajectory_files = tf.io.gfile.glob(
        os.path.join(trajectory_dir, trajectory_file_glob))

    if n_trajectories:
      # We need to get `n_trajectories` number of `trajectory_files`.
      # This works out to a maximum ~3hr waiting period.
      while ((max_tries > 0 or wait_forever) and
             len(trajectory_files) < n_trajectories):
        logging.info(
            "Sleeping for %s seconds while waiting for %s trajectories, found "
            "%s right now.", sleep_time_secs, n_trajectories,
            len(trajectory_files))
        time.sleep(sleep_time_secs)
        max_tries -= 1
        sleep_time_secs = min(10.0, sleep_time_secs * 2)
        trajectory_files = tf.io.gfile.glob(
            os.path.join(trajectory_dir, trajectory_file_glob))

      # We can't get the required number of files and we can't up-sample either.
      if (len(trajectory_files) < n_trajectories) and not up_sample:
        return None

      # Sample up or down as the case maybe.
      trajectory_files = list(
          np.random.choice(trajectory_files, n_trajectories))

    # We read and load all the files, revisit if this becomes a problem.
    trajectories_buffer = []
    for trajectory_file in trajectory_files:
      with tf.io.gfile.GFile(trajectory_file, "rb") as f:
        trajectory = get_pickle_module().load(f)
        assert isinstance(trajectory, Trajectory)
        trajectories_buffer.append(trajectory)

    if not trajectories_buffer:
      return None

    # If n_trajectories wasn't set, then set to the number of trajectories we're
    # returning.
    n_trajectories = n_trajectories or len(trajectories_buffer)

    # Construct and return a new BatchTrajectory object.
    return BatchTrajectory(
        batch_size=n_trajectories,
        trajectories=[Trajectory() for _ in range(n_trajectories)],
        completed_trajectories=trajectories_buffer)
