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

import functools
import random

import numpy as np

from tensor2tensor.envs import env_problem
from tensor2tensor.trax import backend
from tensor2tensor.trax import trax
from tensor2tensor.trax.backend import random as jax_random
from tensor2tensor.trax.rl import space_serializer


class SimulatedEnvProblem(env_problem.EnvProblem):
  """EnvProblem base class for environments simulated by TRAX models.

  The initial observations to start the model are taken from
  initial_observation_stream. This iterator in incremented in every reset().

  A checkpoint saved by the TRAX trainer should be available in output_dir.
  """

  def __init__(self, model, batch_size, observation_space, action_space,
               reward_range, discrete_rewards, history_stream, output_dir):
    """Initializes the env.

    Args:
      model: TRAX model.
      batch_size: (int) Number of simulated environments run in parallel.
      observation_space: (gym.Space) Observation space.
      action_space: (gym.Space) Action space.
      reward_range: (tuple) Pair (min_reward, max_reward).
      discrete_rewards: (bool) Whether to discretize the rewards.
      history_stream: Iterator yielding batches of initial input data for the
        model. The format is implementation-specific.
      output_dir: (str) Output dir.
    """
    # TODO(pkozakowski): At some point we will have a "predict" mode which we
    # should use here. When this happens, change the mode.
    self._model = model
    self._model_predict = backend.jit(self._model(mode="eval"))
    self._observation_space = observation_space
    self._action_space = action_space
    self._reward_range = reward_range
    self._output_dir = output_dir

    self._predict_fn = None
    self._rng = None
    self._model_state = None
    self._history_stream = None

    # Call the super's ctor. It will use some of the member fields, so we call
    # it in the end.
    super(SimulatedEnvProblem, self).__init__(
        batch_size=batch_size,
        discrete_rewards=discrete_rewards,
        history_stream=history_stream,
    )

    self.seed()

  def initialize_environments(self,
                              history_stream,
                              batch_size=1,
                              parallelism=1):
    """Initializes the environments.

    Args:
      history_stream: Iterator yielding batches of initial input data for the
        model. The format is implementation-specific.
      batch_size: (int) Number of environments in a batch.
      parallelism: (int) Unused.
    """
    del parallelism

    model_state = trax.restore_state(self._output_dir)
    model_params = model_state.opt_state.params
    self._model_state = model_state.model_state
    self._predict_fn = functools.partial(
        self._model_predict,
        params=model_params,
    )
    self._history_stream = history_stream

    self._steps = np.zeros(batch_size, dtype=np.int32)

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

  def _reset_model(self, predict_fn, indices, history, rng):
    """Resets the environments at the given indices.

    Should be implemented in subclasses.

    Args:
      predict_fn: Function running prediction with the model.
      indices: List of indices of underlying envs to call reset on.
      history: Initial input data for the model.
      rng: Jax RNG.

    Returns:
      np.ndarray of batched observations from the reset envs.
    """
    raise NotImplementedError

  def _step_model(self, predict_fn, actions, rng):
    """Takes a step in all environments.

    Should be implemented in subclasses.

    Args:
      predict_fn: Function running prediction with the model.
      actions: (np.ndarray) with first dimension equal to the batch size.
      rng: Jax RNG.

    Returns:
      a tuple of batched raw observations, rewards and dones.
    """
    raise NotImplementedError

  def trajectory_to_training_examples(self, trajectory):
    raise NotImplementedError

  @property
  def model_input_shape(self):
    raise NotImplementedError

  @property
  def model_input_dtype(self):
    raise NotImplementedError

  def _reset(self, indices):
    """Resets environments at the given indices.

    Args:
      indices: list of indices of underlying envs to call reset on.

    Returns:
      np.ndarray of batched observations from the reset envs.
    """
    history = next(self._history_stream)
    (subrng, self._rng) = jax_random.split(self._rng)
    return self._reset_model(self._predict_fn, indices, history, subrng)

  def _step(self, actions):
    """Takes a step in all environments.

    Args:
      actions: (np.ndarray) with first dimension equal to the batch size.

    Returns:
      a tuple of batched raw observations, raw rewards, dones and infos.
    """
    # Predict the next observation.
    (subrng, self._rng) = jax_random.split(self._rng)
    (observation, reward, done) = self._step_model(
        self._predict_fn, actions, subrng)
    return (observation, reward, done, {})

  @property
  def model(self):
    return self._model


class RawSimulatedEnvProblem(SimulatedEnvProblem):
  """SimulatedEnvProblem running a model operating on raw tensors.

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
  """

  def __init__(self, history_length, trajectory_length, *args, **kwargs):
    """Initializes the env.

    Args:
      history_length: (int) Number of last observations fed into the model.
      trajectory_length: (int) Length of each trajectory unrolled from the
        model.
      *args: (tuple) Positional arguments passed to the base class.
      **kwargs: (dict) Keyword arguments passed to the base class.
    """
    self._history_length = history_length
    self._trajectory_length = trajectory_length
    self._history = None
    self._steps = None

    super(RawSimulatedEnvProblem, self).__init__(*args, **kwargs)

  def initialize_environments(self, batch_size=1, **kwargs):
    """Initializes the environments."""
    self._history = None
    self._steps = np.zeros(batch_size)
    return super(RawSimulatedEnvProblem, self).initialize_environments(
        batch_size=batch_size, **kwargs)

  def _reset_model(self, predict_fn, indices, history, rng):
    del predict_fn
    del rng
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

  def _step_model(self, predict_fn, actions, rng):
    (observation, reward), self._model_state = predict_fn(
        (self._history, actions), state=self._model_state, rng=rng)

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
    return (observation, reward, done)


def index_range_2d(begin_indices, length):
  # Take all indices along the first dimension. Add another axis that'll
  # broadcast along the second one.
  first_dim = np.arange(len(begin_indices))[:, None]
  # Take a range of indices along the second dimension. Offset it by
  # begin_indices.
  # TODO(pkozakowski): This materializes all indices of elements along the
  # second dimension. Do it more efficiently if needed.
  second_dim = np.arange(length)[None, :] + begin_indices[:, None]
  return (first_dim, second_dim)


def index_slice(indices):
  first_dim = np.arange(len(indices))[:, None]
  second_dim = indices[:, None]
  return (first_dim, second_dim)


class SerializedSequenceSimulatedEnvProblem(SimulatedEnvProblem):
  """SimulatedEnvProblem running a model operating on sequences of symbols.

  Wraps an autoregressive TRAX model of signature past_symbols -> symbol_probs
  in an EnvProblem. The model is assumed to take a sequence of symbols as input
  and produce distributions over all symbols in the sequence. The next symbol
  is sampled and fed back to the model in the next decoding step.

  Shape requirements (without the batch dimension):
    past_symbols: (max_trajectory_length * L,)
    symbol_probs: (max_trajectory_length * L, vocab_size)
  where L is the representation length of one environment step.

  Observations, actions, rewards and done flags are (de)serialized from/to
  sequences of symbols using an EnvSerializer passed to the constructor.
  """

  def __init__(self, model, reward_fn, done_fn, vocab_size,
               max_trajectory_length, observation_space, action_space,
               *args, **kwargs):
    """Initializes the env.

    Args:
      model: TRAX model to use for simulation. It's assumed to take keyword
        arguments vocab_size and mode, where vocab_size is the number of symbols
        in the vocabulary and mode is either "train" or "eval".

      reward_fn: Function (previous_observation, current_observation) -> reward.
      done_fn: Function (previous_observation, current_observation) -> done.
      vocab_size: (int) Number of symbols in the vocabulary.
      max_trajectory_length: (int) Maximum length of a trajectory unrolled from
        the model.
      observation_space: (gym.Space) Observation space.
      action_space: (gym.Space) Action space.
      *args: (tuple) Positional arguments passed to the base class.
      **kwargs: (dict) Keyword arguments passed to the base class.
    """
    self._reward_fn = reward_fn
    self._done_fn = done_fn
    self._vocab_size = vocab_size
    self._max_trajectory_length = max_trajectory_length
    self._history = None
    self._steps = None
    self._observation_space = None
    self._action_space = None
    self._last_observations = None

    self._obs_serializer = space_serializer.create(
        observation_space, self._vocab_size)
    self._action_serializer = space_serializer.create(
        action_space, self._vocab_size)
    self._obs_repr_length = self._obs_serializer.representation_length
    self._action_repr_length = self._action_serializer.representation_length
    self._step_repr_length = self._obs_repr_length + self._action_repr_length

    # We assume that the model takes vocab_size as an argument (e.g.
    # TransformerLM).
    model = functools.partial(model, vocab_size=vocab_size)
    super(SerializedSequenceSimulatedEnvProblem, self).__init__(
        *args,
        model=model,
        observation_space=observation_space,
        action_space=action_space,
        **kwargs
    )

  def initialize_environments(self, batch_size=1, **kwargs):
    """Initializes the environments."""
    self._history = np.zeros((
        batch_size,
        self._max_trajectory_length * self._step_repr_length
    ), dtype=np.int32)
    self._steps = np.zeros(batch_size, dtype=np.int32)
    self._last_observations = np.full(
        (batch_size,) + self._observation_space.shape, np.nan)
    super(SerializedSequenceSimulatedEnvProblem, self).initialize_environments(
        batch_size=batch_size, **kwargs)

  @property
  def _obs_repr_indices(self):
    begin_indices = self._step_repr_length * self._steps
    return index_range_2d(begin_indices, self._obs_repr_length)

  @property
  def _action_repr_indices(self):
    begin_indices = self._step_repr_length * self._steps + self._obs_repr_length
    return index_range_2d(begin_indices, self._action_repr_length)

  def _predict_obs(self, predict_fn, rng):
    def gumbel_sample(log_probs):
      u = np.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
      g = -np.log(-np.log(u))
      return np.argmax(log_probs + g, axis=-1)

    for (i, subrng) in enumerate(jax_random.split(rng, self._obs_repr_length)):
      symbol_index = self._steps * self._step_repr_length + i
      log_probs, self._model_state = predict_fn(self._history,
                                                state=self._model_state,
                                                rng=subrng)
      log_probs = log_probs[:, symbol_index, :]
      self._history[:, symbol_index] = gumbel_sample(log_probs)

    obs_repr = self._history[self._obs_repr_indices]
    return self._obs_serializer.deserialize(obs_repr)

  def _reset_model(self, predict_fn, indices, history, rng):
    # TODO(pkozakowski): Random starts.
    del history

    self._steps[indices] = 0
    observation = self._predict_obs(predict_fn, rng)[indices]
    self._last_observations[indices] = observation
    return observation

  def _step_model(self, predict_fn, actions, rng):
    action_repr = self._action_serializer.serialize(actions)
    self._history[self._action_repr_indices] = action_repr
    self._steps += 1
    observation = self._predict_obs(predict_fn, rng)
    reward = self._reward_fn(self._last_observations, observation)
    done = self._done_fn(self._last_observations, observation)
    self._last_observations = observation
    done = np.logical_or(done, self._steps == self._max_trajectory_length - 1)
    return (observation, reward, done)

  def trajectory_to_training_examples(self, trajectory):
    reprs = []
    weights = []
    for time_step in trajectory.time_steps:
      # Serializers work on batches.
      obs_repr = self._obs_serializer.serialize(
          np.array([time_step.observation]))[0]
      reprs.append(obs_repr)
      # TODO(pkozakowski): Digit weighting.
      weights.append(np.ones_like(obs_repr))
      if time_step.action is not None:
        action_repr = self._action_serializer.serialize(
            np.array([time_step.action]))[0]
        reprs.append(action_repr)
        weights.append(np.zeros_like(action_repr))

    def concat_and_pad(arrays):
      (desired_length,) = self.model_input_shape
      flat_array = np.concatenate(arrays, axis=0)
      (actual_length,) = flat_array.shape
      assert actual_length <= desired_length
      return np.pad(
          flat_array,
          pad_width=((0, desired_length - actual_length),),
          mode="constant",
      )
    (reprs, weights) = map(concat_and_pad, (reprs, weights))
    return [(reprs, reprs, weights)]  # (inputs, targets, weights)

  @property
  def model_input_shape(self):
    return (self._max_trajectory_length * self._step_repr_length,)

  @property
  def model_input_dtype(self):
    return np.int32


def cartpole_done_fn(previous_observation, current_observation):
  del previous_observation
  x_threshold = 2.4
  theta_threshold = 12 * 2 * np.pi / 360
  x = current_observation[:, 0]
  theta = current_observation[:, 2]
  return np.logical_or(np.abs(x) > x_threshold, np.abs(theta) > theta_threshold)


def cartpole_reward_fn(previous_observation, current_observation):
  done = cartpole_done_fn(previous_observation, current_observation)
  return 1.0 - done  # Unit reward for every timestep until the end.


def acrobot_done_fn(previous_observation, current_observation):
  del previous_observation
  theta1 = current_observation[:, 0]
  theta2 = current_observation[:, 1]
  return -np.cos(theta1) - np.cos(theta2 + theta1) > 1.0


def acrobot_reward_fn(previous_observation, current_observation):
  done = acrobot_done_fn(previous_observation, current_observation)
  return -1.0 + done  # -1 reward for every timestep until the end.
