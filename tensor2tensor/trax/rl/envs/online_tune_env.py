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

import functools
import os

import gym

from tensor2tensor.trax import inputs as trax_inputs
from tensor2tensor.trax import layers
from tensor2tensor.trax import models as trax_models
from tensor2tensor.trax import optimizers as trax_opt
from tensor2tensor.trax import trax
from tensor2tensor.trax.rl import online_tune
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
               output_dir,
               model=trax_models.TransformerLM,
               trainer_class=trax.Trainer,
               loss_fn=layers.CrossEntropyLossScalar,
               optimizer=trax_opt.Adafactor,
               inputs=trax_inputs.inputs,
               action_multipliers=None,
               observation_metrics=(
                   ("train", "metrics/accuracy"),
                   ("train", "metrics/loss"),
                   ("eval", "metrics/accuracy"),
                   ("eval", "metrics/loss"),
               ),
               include_controls_in_observation=False,
               reward_metric=("eval", "metrics/accuracy"),
               train_steps=100,
               eval_steps=10,
               env_steps=100,
               # This is a tuple instead of a dict because the controls are
               # ordered in the action space.
               control_configs=(
                   # (name, start, (low, high), flip)
                   ("learning_rate", 1e-3, (1e-9, 10.0), False),
               ),
               nontrainable_param_map=None,
               observation_range=(0.0, 10.0),
               # Don't save checkpoints by default, as they tend to use a lot of
               # space.
               should_save_checkpoints=False,
               has_weights=False,
               mask_id=None):
    if action_multipliers is None:
      action_multipliers = self.DEFAULT_ACTION_MULTIPLIERS
    self._model = model
    # Initialize Trainer in OnlineTuneEnv lazily to prevent long startup in the
    # async setup, where we just use the environments as containers for
    # trajectories.
    self._trainer_fn = functools.partial(
        trainer_class,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_schedule=(lambda history: lambda step: self._current_controls),
        inputs=inputs,
        should_save=should_save_checkpoints,
        nontrainable_param_map=nontrainable_param_map,
        has_weights=has_weights,
        mask_id=mask_id,
    )
    self._trainer = None
    self._action_multipliers = action_multipliers
    self._observation_metrics = observation_metrics
    self._include_controls_in_observation = include_controls_in_observation
    self._reward_metric = reward_metric
    self._train_steps = train_steps
    self._eval_steps = eval_steps
    self._env_steps = env_steps
    self._control_configs = control_configs
    self._observation_range = observation_range

    self._output_dir = output_dir
    gfile.makedirs(self._output_dir)
    # Actions are indices in self._action_multipliers.
    self.action_space = gym.spaces.MultiDiscrete(
        [len(self._action_multipliers)] * len(self._control_configs)
    )
    # Observation is a vector with the values of the metrics specified in
    # observation_metrics plus optionally the current controls.
    observation_dim = (
        len(self._observation_metrics) +
        int(self._include_controls_in_observation) * len(self._control_configs)
    )

    (obs_low, obs_high) = observation_range
    self.observation_space = gym.spaces.Box(
        # Observations are clipped to this range.
        low=obs_low, high=obs_high, shape=(observation_dim,),
    )

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

  @property
  def _current_reward_metric(self):
    metric_values = online_tune.historical_metric_values(
        self._trainer.state.history,
        self._reward_metric,
    )
    assert metric_values.shape[0] > 0, (
        "No values in history for metric {}.".format(self._reward_metric))
    return metric_values[-1]

  @property
  def _current_observation(self):
    observations = online_tune.history_to_observations(
        self._trainer.state.history,
        self._observation_metrics,
        self._observation_range,
        self._control_configs if self._include_controls_in_observation
        else None,
    )
    assert observations.shape[0] > 0, "No values in history for any metric."
    return observations[-1, :]

  @property
  def trainer(self):
    if self._trainer is None:
      raise ValueError("The environment has to be reset first.")
    return self._trainer

  def reset(self):
    if self._trainer is None:
      self._trainer = self._trainer_fn()
    self._current_controls = {
        name: start_value
        for (name, start_value, _, _) in self._control_configs
    }
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
    self._current_controls = {
        # name: value
        control_config[0]: online_tune.update_control(  # pylint: disable=g-complex-comprehension
            control_config,
            control_action,
            self._trainer.state.history,
            self._action_multipliers,
        )
        for (control_action, control_config) in zip(
            action, self._control_configs
        )
    }
    last_reward_metric = self._current_reward_metric
    self._trainer.train_epoch(self._train_steps, self._eval_steps)
    self._step += 1
    current_reward_metric = self._current_reward_metric
    observation = self._current_observation
    reward = current_reward_metric - last_reward_metric
    done = self._step == self._env_steps
    return (observation, reward, done, {})
