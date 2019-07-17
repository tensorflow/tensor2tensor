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

"""EnvProblem for environments simulated by a TRAX model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from tensor2tensor.envs import env_problem
from tensor2tensor.trax import backend
from tensor2tensor.trax import trax
from tensor2tensor.trax.backend import random as jax_random


class SimulatedEnvProblem(env_problem.EnvProblem):
  """EnvProblem for environments simulated by TRAX models.

  Wraps an autoregressive TRAX model of signature
  (observation_history, action) -> (observation, reward) in an EnvProblem.
  The model is assumed to take a fixed number of last observations as input
  and produce a single observation, which is fed back into the model in the
  next environment step.

  Shape requirements (without the batch dimension):
    observation: Consistent with observation_space.
    observation_history: (history_length,) + observation.shape.
    action: Consistent with action_space.
    reward: (1,). The singleton dimension is removed in step().

  The initial observations to start the model are taken from
  initial_observation_stream. This iterator in incremented in every reset().

  A checkpoint saved by the TRAX trainer should be available in output_dir.
  """

  def __init__(self, model, history_length, trajectory_length, batch_size,
               observation_space, action_space, reward_range, discrete_rewards,
               initial_observation_stream, output_dir):
    """Initializes the env.

    Args:
      model: TRAX model.
      history_length: (int) Number of last observations fed into the model.
      trajectory_length: (int) Length of each trajectory unrolled from the
        model.
      batch_size: (int) Number of simulated environments run in parallel.
      observation_space: (gym.Space) Observation space.
      action_space: (gym.Space) Action space.
      reward_range: (tuple) Pair (min_reward, max_reward).
      discrete_rewards: (bool) Whether to discretize the rewards.
      initial_observation_stream: Iterator yielding batches of initial
        observations for the model.
      output_dir: (str) Output dir.
    """
    # TODO(pkozakowski): At some point we will have a "predict" mode which we
    # should use here. When this happens, change the mode.
    self._model_predict = backend.jit(model(mode="eval"))
    self._history_length = history_length
    self._trajectory_length = trajectory_length
    self._observation_space = observation_space
    self._action_space = action_space
    self._reward_range = reward_range
    self._output_dir = output_dir

    self._model_params = None
    self._rng = None
    self._initial_observation_stream = None
    self._history = None
    self._steps = None

    # Call the super's ctor. It will use some of the member fields, so we call
    # it in the end.
    super(SimulatedEnvProblem, self).__init__(
        batch_size=batch_size,
        discrete_rewards=discrete_rewards,
        initial_observation_stream=initial_observation_stream,
    )

    self.seed()

  def initialize_environments(self,
                              initial_observation_stream,
                              batch_size=1,
                              parallelism=1):
    """Initializes the environments.

    Args:
      initial_observation_stream: Iterator yielding batches of initial
        observations for the model.
      batch_size: (int) Number of environments in a batch.
      parallelism: (int) Unused.
    """
    del parallelism

    model_state = trax.restore_state(self._output_dir)
    # model_state.params is a pair (model_params, optimizer_state).
    (self._model_params, _) = model_state.params
    self._initial_observation_stream = initial_observation_stream

    self._history = None
    self._steps = np.zeros(batch_size)

  @property
  def observation_space(self):
    return self._observation_space

  @property
  def action_space(self):
    return self._action_space

  @property
  def reward_range(self):
    return self._reward_range

  def seed(self, seed=None):
    if seed is None:
      seed = random.randint(0, 2**31 - 1)
    self._rng = jax_random.get_prng(seed)
    return super(SimulatedEnvProblem, self).seed(seed=seed)

  def _reset(self, indices):
    """Resets environments at the given indices.

    Args:
      indices: list of indices of underlying envs to call reset on.

    Returns:
      np.ndarray of batched observations from the reset envs.
    """
    history = next(self._initial_observation_stream)
    assert history.shape == ((self._batch_size, self._history_length) +
                             self.observation_space.shape)

    if self._history is None:
      # At the first reset, all indices should be triggered.
      assert set(indices) == set(range(self._batch_size))
      self._history = np.array(history)
    else:
      history = history[indices, ...]
      self._history[indices, ...] = history

    # Reset the step counters.
    self._steps[indices] = 0

    # Return just the last timestep at the given indices.
    return history[:, -1, ...]

  def _step(self, actions):
    """Takes a step in all environments.

    Args:
      actions: (np.ndarray) with first dimension equal to the batch size.

    Returns:
      a tuple of batched raw observations, raw rewards, dones and infos.
    """
    # Predict the next observation.
    (subrng, self._rng) = jax_random.split(self._rng)
    (observation, reward) = self._model_predict((self._history, actions),
                                                params=self._model_params,
                                                rng=subrng)

    # Roll the history one timestep back and append the new observation.
    self._history = np.roll(self._history, shift=-1, axis=1)
    self._history[:, -1, ...] = observation

    # Increment the step counters and determine which envs are done.
    self._steps += 1
    done = self._steps == self._trajectory_length

    # Call copy() to get the data as numpy arrays.
    observation = observation.copy()
    # Reshape the rewards to get rid of the extra dimension.
    reward = np.squeeze(reward.copy(), axis=1)
    return (observation, reward, done, {})
