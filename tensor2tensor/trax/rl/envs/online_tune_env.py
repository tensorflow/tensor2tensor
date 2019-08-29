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

"""An environment for tuning model hyperparameters during training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import gym
import numpy as np

from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax
from tensorflow.io import gfile


class OnlineTuneEnv(gym.Env):
  """An environment for tuning model hyperparameters during training.

  A rollout is one instance of training a specific model on a specific problem.
  Observations are the values of some evaluation metric. Actions control
  hyperparameter changes during training. Reward is the change of the evaluation
  metric. One environment step corresponds to a fixed number of training steps.

  For now we only support tuning the learning rate.
  """

  # Chosen so that the opposite actions cancel each other out, so random walk
  # has a median of 1.
  DEFAULT_ACTION_MULTIPLIERS = [1.0 / 1.5, 1.0 / 1.25, 1.0, 1.25, 1.5]

  def __init__(self,
               model,
               output_dir,
               trainer_class=trax.Trainer,
               loss_fn=trax.loss,
               optimizer=trax_opt.SM3,
               inputs=trax_inputs.inputs,
               action_multipliers=None,
               observation_metrics=(("eval", "metrics/accuracy"),),
               include_lr_in_observation=False,
               reward_metric=("eval", "metrics/accuracy"),
               train_steps=100,
               eval_steps=10,
               env_steps=100,
               start_lr=0.001,
               max_lr=10.0,
               # Don't save checkpoints by default, as they tend to use a lot of
               # space.
               should_save_checkpoints=False):
    if action_multipliers is None:
      action_multipliers = self.DEFAULT_ACTION_MULTIPLIERS
    self._model = model
    self._trainer = trainer_class(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_schedule=(lambda history: lambda step: self._current_lr),
        inputs=inputs,
        should_save=should_save_checkpoints,
    )
    self._action_multipliers = action_multipliers
    self._observation_metrics = observation_metrics
    self._include_lr_in_observation = include_lr_in_observation
    self._reward_metric = reward_metric
    self._train_steps = train_steps
    self._eval_steps = eval_steps
    self._env_steps = env_steps
    self._start_lr = start_lr
    self._max_lr = max_lr

    self._output_dir = output_dir
    gfile.makedirs(self._output_dir)
    # Action is an index in self._action_multipliers.
    self.action_space = gym.spaces.Discrete(len(self._action_multipliers))
    # Observation is a vector with the values of the metrics specified in
    # observation_metrics plus optionally the learning rate.
    observation_dim = (
        len(self._observation_metrics) + int(self._include_lr_in_observation))
    self.observation_space = gym.spaces.Box(
        low=float("-inf"), high=float("+inf"), shape=(observation_dim,))

  @property
  def _next_trajectory_dir(self):
    """Assigns a new output dir for a trajectory under self._output_dir.

    Directory names are consecutive integers starting from zero. New directory
    index is assigned as the maximum of past indices plus one. Directories that
    are not integers are ignored.

    Returns:
      A path of the new directory.
    """
    trajectory_dirs = gfile.listdir(self._output_dir)

    def int_or_none(s):
      try:
        return int(s)
      except TypeError:
        return None

    past_trajectory_ids = [
        trajectory_id for trajectory_id in map(int_or_none, trajectory_dirs)
        if trajectory_id is not None]
    next_trajectory_id = max([-1] + past_trajectory_ids) + 1

    return os.path.join(self._output_dir, str(next_trajectory_id))

  def _current_metric_value(self, metric):
    metric_sequence = self._trainer.state.history.get(*metric)
    assert metric_sequence
    (_, metric_value) = metric_sequence[-1]
    return metric_value

  @property
  def _current_observation(self):
    observation = list(
        map(self._current_metric_value, self._observation_metrics))
    if self._include_lr_in_observation:
      # Logartihm of the learning rate.
      observation.append(math.log(self._current_lr))
    return np.array(observation)

  @property
  def trainer(self):
    if self._trainer is None:
      raise ValueError("The environment has to be reset first.")
    return self._trainer

  def reset(self):
    self._current_lr = self._start_lr
    self._step = 0
    self._trainer.reset(output_dir=self._next_trajectory_dir)
    self._trainer.evaluate(self._eval_steps)
    return self._current_observation

  def step(self, action):
    """Step the environment.

    One environment step corresponds to self.train_steps training steps.

    Args:
      action: (int) Action to take. An index in self.action_multipliers.

    Returns:
      Tuple (observation, reward, done, info). observation is a singleton vector
        with the current value of the metric. reward is the difference in the
        metric since the last step. done is set after reaching self.env_steps
        environment steps. info is an empty dict.
    """
    self._current_lr = min(
        self._current_lr * self._action_multipliers[action], self._max_lr)
    last_metric_value = self._current_metric_value(self._reward_metric)
    self._trainer.train_epoch(self._train_steps, self._eval_steps)
    self._step += 1
    current_metric_value = self._current_metric_value(self._reward_metric)
    observation = self._current_observation
    reward = current_metric_value - last_metric_value
    done = self._step == self._env_steps
    return (observation, reward, done, {})
