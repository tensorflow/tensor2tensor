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

import gin
import gym
import numpy as np
import inspect
import os

import cv2
from gym import spaces, Wrapper
from gym.wrappers import TimeLimit

from dopamine.agents.dqn.dqn_agent import OBSERVATION_SHAPE, STACK_SIZE
from dopamine.replay_memory import circular_replay_buffer

_dopamine_path = None
try:
  import dopamine
  from dopamine.agents.dqn import dqn_agent
  from dopamine.atari import run_experiment
  _dopamine_path = os.path.dirname(inspect.getfile(dopamine))
except:
  # Generally we do not need dopamine in tensor2tensor
  # We will raise exception if the code really tries to use it
  pass

import tensorflow as tf
from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv, SimulatedBatchGymEnv


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
    res = self.env.step(action)
    self.game_over = res[2]
    return res


class _DQNAgent(dqn_agent.DQNAgent):
  """Allow passing batch_size and replay_capacity to ReplayBuffer"""

  def __init__(self, replay_capacity, batch_size, **kwargs):
    self._replay_capacity = replay_capacity
    self._batch_size = batch_size
    super(_DQNAgent, self).__init__(**kwargs)

  def _build_replay_buffer(self, use_staging):
    return circular_replay_buffer.WrappedReplayBuffer(
        observation_shape=OBSERVATION_SHAPE,
        stack_size=STACK_SIZE,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        batch_size=self._batch_size,
        replay_capacity=self._replay_capacity)


def get_create_agent(kwargs):
  def create_agent(sess, environment, summary_writer=None):
    """Creates a DQN agent.
  
    Simplified version of `dopamine.atari.train.create_agent`
    """
    return _DQNAgent(sess=sess, num_actions=environment.action_space.n,
                     summary_writer=summary_writer,
                     tf_device='/gpu:*',
                      **kwargs)
  return create_agent


def get_create_env_fun(env_spec, world_model_dir, time_limit):
  simulated = env_spec.simulated_env
  def create_env_fun(_1, _2):
    if simulated:
      batch_env = SimulatedBatchGymEnv(env_spec, 1, model_dir=world_model_dir)
    else:
      batch_env = env_spec.env

    env = FlatBatchEnv(batch_env)
    env = TimeLimit(env, max_episode_steps=time_limit)
    env = ResizeObservation(env)

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
  assert params['class']=="RMSProp", "RMSProp is the only one supported"
  params.pop('class')
  return tf.train.RMSPropOptimizer(**params)


def dopamine_trainer(hparams, model_dir):
  """ Simplified version of `dopamine.atari.train.create_runner` """
  global _dopamine_path

  assert _dopamine_path is not None, "Dopamine not available. Please install from " \
                                     "https://github.com/google/dopamine and add to PYTHONPATH"

  agent_params, optimizer_params, \
  runner_params, replay_buffer_params = _parse_hparams(hparams)
  optimizer = _get_optimizer(optimizer_params)
  agent_params['optimizer'] = optimizer
  agent_params.update(replay_buffer_params)

  create_agent = get_create_agent(agent_params)

  with tf.Graph().as_default():
    runner = run_experiment.Runner(
        base_dir=model_dir, game_name="fake", create_agent_fn=create_agent,
        create_environment_fn=get_create_env_fun(hparams.environment_spec,
                                                 hparams.get('world_model_dir', None),
                                                 time_limit=hparams.time_limit),
        evaluation_steps=0,
        **runner_params
    )

    runner.run_experiment()
