# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Connects dopamine to as the another rl traning framework."""

import gym
from gym import spaces, Wrapper
from gym.wrappers import TimeLimit
import numpy as np
import inspect
import os
import tensorflow as tf

from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv, \
  SimulatedBatchGymEnv


_dopamine_path = None
try:
  # pylint: disable=wrong-import-position
  import dopamine
  import cv2
  from dopamine.agents.dqn import dqn_agent
  from dopamine.atari import run_experiment
  from dopamine.agents.dqn.dqn_agent import OBSERVATION_SHAPE, STACK_SIZE
  from dopamine.replay_memory import circular_replay_buffer
  from dopamine.replay_memory.circular_replay_buffer import \
  OutOfGraphReplayBuffer, ReplayElement

  # pylint: enable=wrong-import-position
  _dopamine_path = os.path.dirname(inspect.getfile(dopamine))
except ImportError:
  # Generally we do not need dopamine in tensor2tensor
  # We will raise exception if the code really tries to use it
  pass


class ResizeObservation(gym.ObservationWrapper):
  def __init__(self, env, size=84):
    """
    Based on WarpFrame from openai baselines atari_wrappers.py
    Dopamine also uses cv2.resize(..., interpolation=cv2.INTER_AREA).
    """
    gym.ObservationWrapper.__init__(self, env)
    self.width = size
    self.height = size
    assert env.observation_space.dtype == np.uint8
    self.observation_space = spaces.Box(
        low=0, high=255,
        shape=(self.height, self.width, env.observation_space.shape[2]),
        dtype=np.uint8)

  def observation(self, frame):
    return cv2.resize(frame, (self.width, self.height),
                      interpolation=cv2.INTER_AREA)


class GameOverOnDone(Wrapper):

  def __init__(self, env):
    Wrapper.__init__(self, env)
    self.game_over = False

  def reset(self, **kwargs):
    self.game_over = False
    return self.env.reset(**kwargs)

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.game_over = done
    return ob, reward, done, info


class _DQNAgent(dqn_agent.DQNAgent):
  """ Modify dopamine DQNAgent to match our needs.

  Allow passing batch_size and replay_capacity to ReplayBuffer, allow not using
  (some of) terminal episode transitions in training.
  """

  def __init__(self, replay_capacity, batch_size,
               generates_trainable_dones, **kwargs):
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    self._generates_trainable_dones = generates_trainable_dones
    super(_DQNAgent, self).__init__(**kwargs)

  def _build_replay_buffer(self, use_staging):
    """Build WrappedReplayBuffer with custom OutOfGraphReplayBuffer"""
    replay_buffer_kwargs = dict(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        replay_capacity=self._replay_capacity,
        batch_size=self._batch_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        extra_storage_types=None,
        observation_dtype=np.uint8,
    )
    replay_memory = _OutOfGraphReplayBuffer(
        artificial_done=not self._generates_trainable_dones,
        **replay_buffer_kwargs)

    # TODO(KC) pass use_staging
    return circular_replay_buffer.WrappedReplayBuffer(
        wrapped_memory=replay_memory, use_staging=False,
        **replay_buffer_kwargs)


class _OutOfGraphReplayBuffer(OutOfGraphReplayBuffer):
  """ Replay not sampling artificial_terminal transition.

  Adds to stored tuples 'artificial_done' field (as last ReplayElement).
  When sampling, ignores tuples for which artificial_done is True.

  When adding new attributes check if there are loaded from disk, when using
  load() method.

  Attributes:
      are_terminal_valid: A boolean indicating if newly added terminal
      transitions should be marked as artificially done. Replay data
      loaded from disk will not be overridden.
  """

  def __init__(self, artificial_done, **kwargs):
    extra_storage_types = kwargs.pop('extra_storage_types', None) or []
    extra_storage_types.append(
        ReplayElement('artificial_done', (), np.uint8))
    super(_OutOfGraphReplayBuffer, self).__init__(
        extra_storage_types=extra_storage_types, **kwargs)
    self._artificial_done = artificial_done

  def is_valid_transition(self, index):
    valid = super(_OutOfGraphReplayBuffer, self).is_valid_transition(index)
    valid &= not self.get_artificial_done_stack(index).any()
    return valid

  def get_artificial_done_stack(self, index):
    return self.get_range(self._store['artificial_done'],
                          index - self._stack_size + 1,
                          index + 1)

  def add(self, observation, action, reward, terminal, *args):
    """Append artificial_done to *args and run parent method."""
    # If this will be a problem for maintenance, we could probably override
    # DQNAgent.add() method instead.
    artificial_done = self._artificial_done and terminal
    args = list(args)
    args.append(artificial_done)
    return super(_OutOfGraphReplayBuffer, self).add(
        observation, action, reward, terminal, *args)

  def load(self, *args, **kwargs):
    # Check that appropriate attributes are not overridden
    are_terminal_valid = self._artificial_done
    super(_OutOfGraphReplayBuffer, self).load(*args, **kwargs)
    assert self._artificial_done == are_terminal_valid


def get_create_agent(agent_kwargs):
  def create_agent(sess, environment, summary_writer=None):
    """Creates a DQN agent.

    Simplified version of `dopamine.atari.train.create_agent`
    """
    return _DQNAgent(sess=sess, num_actions=environment.action_space.n,
                     summary_writer=summary_writer,
                     tf_device='/gpu:*', **agent_kwargs)

  return create_agent


def get_create_env_fun(env_spec, world_model_dir, time_limit):
  simulated = env_spec.simulated_env

  def create_env_fun(game_name, sticky_actions=True):
    del game_name, sticky_actions
    if simulated:
      batch_env = SimulatedBatchGymEnv(env_spec, 1, model_dir=world_model_dir)
    else:
      batch_env = env_spec.env

    env = FlatBatchEnv(batch_env)
    env = TimeLimit(env, max_episode_steps=time_limit)
    env = ResizeObservation(env)  # pylint: disable=redefined-variable-type

    env = GameOverOnDone(env)
    return env

  return create_env_fun


def _parse_hparams(hparams):
  prefixes = ["agent_", "optimizer_", "runner_", "replay_buffer_"]
  ret = []

  for prefix in prefixes:
    ret_dict = {}
    for key in hparams.values():
      if prefix in key:
        par_name = key[len(prefix):]
        ret_dict[par_name] = hparams.get(key)
    ret.append(ret_dict)

  return ret


def _get_optimizer(params):
  assert params['class'] == "RMSProp", "RMSProp is the only one supported"
  params.pop('class')
  return tf.train.RMSPropOptimizer(**params)


def dopamine_trainer(hparams, model_dir):
  """ Simplified version of `dopamine.atari.train.create_runner` """
  assert _dopamine_path is not None, "Dopamine not available. Please install " \
                                     "from " \
                                     "https://github.com/google/dopamine and " \
                                     "add to PYTHONPATH"

  # pylint: disable=unbalanced-tuple-unpacking
  agent_params, optimizer_params, \
  runner_params, replay_buffer_params = _parse_hparams(hparams)
  # pylint: enable=unbalanced-tuple-unpacking
  optimizer = _get_optimizer(optimizer_params)
  agent_params['optimizer'] = optimizer
  agent_params.update(replay_buffer_params)

  create_agent = get_create_agent(agent_params)

  with tf.Graph().as_default():
    runner = run_experiment.Runner(
        game_name="unused_arg", sticky_actions="unused_arg",
        base_dir=model_dir, create_agent_fn=create_agent,
        create_environment_fn=get_create_env_fun(hparams.environment_spec,
                                                 hparams.get('world_model_dir',
                                                             None),
                                                 time_limit=hparams.time_limit),
        evaluation_steps=0,
        **runner_params
    )

    runner.run_experiment()
