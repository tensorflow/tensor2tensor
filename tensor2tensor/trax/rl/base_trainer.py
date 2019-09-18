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

"""Base class for RL trainers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
from tensor2tensor.trax import utils
from tensorflow.io import gfile


class BaseTrainer(object):
  """Base class for RL trainers."""

  def __init__(
      self, train_env, eval_env, output_dir,
      trajectory_dump_dir=None, trajectory_dump_min_count_per_shard=16,
      async_mode=False,
  ):
    """Base class constructor.

    Args:
      train_env: EnvProblem to use for training. Settable.
      eval_env: EnvProblem to use for evaluation. Settable.
      output_dir: Directory to save checkpoints and metrics to.
      trajectory_dump_dir: Directory to dump trajectories to. Trajectories
        are saved in shards of name <epoch>.pkl under this directory. Settable.
      trajectory_dump_min_count_per_shard: Minimum number of trajectories to
        collect before dumping in a new shard. Sharding is for efficient
        shuffling for model training in SimPLe.
      async_mode: (bool) If True, this means we are in async mode and we read
        trajectories from a location rather than interact with the environment.
    """
    self.train_env = train_env
    self.eval_env = eval_env
    self._output_dir = output_dir
    gfile.makedirs(self._output_dir)
    self.trajectory_dump_dir = trajectory_dump_dir
    self._trajectory_dump_min_count_per_shard = (
        trajectory_dump_min_count_per_shard)
    self._trajectory_buffer = []
    self._async_mode = async_mode

  @property
  def async_mode(self):
    return self._async_mode

  @async_mode.setter
  def async_mode(self, async_mode):
    logging.vlog(1, "Changing async mode from %s to: %s",
                 self._async_mode, async_mode)
    self._async_mode = async_mode

  @property
  def epoch(self):
    raise NotImplementedError

  def train_epoch(self, evaluate=True):
    raise NotImplementedError

  def evaluate(self):
    raise NotImplementedError

  def save(self):
    raise NotImplementedError

  def flush_summaries(self):
    raise NotImplementedError

  def dump_trajectories(self, force=False):
    """Dumps trajectories in a new shard.

    Should be called at most once per epoch.

    Args:
      force: (bool) Whether to complete unfinished trajectories and create
        a new shard even if we have not reached the minimum size.
    """
    pkl_module = utils.get_pickle_module()
    if self.trajectory_dump_dir is None:
      return
    gfile.makedirs(self.trajectory_dump_dir)

    trajectories = self.train_env.trajectories
    if force:
      trajectories.complete_all_trajectories()

    # complete_all_trajectories() also adds trajectories that were just reset.
    # We don't want them since they have just the initial observation and no
    # actions, so we filter them out.
    def has_any_action(trajectory):
      return (
          trajectory.time_steps and trajectory.time_steps[0].action is not None)
    self._trajectory_buffer.extend(
        filter(has_any_action, trajectories.completed_trajectories))

    trajectories.clear_completed_trajectories()
    ready = (
        len(self._trajectory_buffer) >=
        self._trajectory_dump_min_count_per_shard
    )
    if ready or force:
      shard_path = os.path.join(
          self.trajectory_dump_dir, "{}.pkl".format(self.epoch))
      if gfile.exists(shard_path):
        # Since we do an extra dump at the end of the training loop, we
        # sometimes dump 2 times in the same epoch. When this happens, merge the
        # two sets of trajectories.
        with gfile.GFile(shard_path, "rb") as f:
          self._trajectory_buffer = pkl_module.load(f) + self._trajectory_buffer
      with gfile.GFile(shard_path, "wb") as f:
        pkl_module.dump(self._trajectory_buffer, f)
      self._trajectory_buffer = []

  def training_loop(self, n_epochs, evaluate=True):
    logging.info("Starting the RL training loop.")
    for _ in range(self.epoch, n_epochs):
      self.train_epoch(evaluate=evaluate)
      self.dump_trajectories()
    self.save()
    self.dump_trajectories(force=True)
    if evaluate:
      self.evaluate()
    self.flush_summaries()
