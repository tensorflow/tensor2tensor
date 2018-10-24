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


class AddGameOverProperty(Wrapper):
  @property
  def game_over(self):
    if hasattr(self.env, 'game_over'):
      game_over = self.env.game_over
    else:
      game_over = False
    return game_over

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

  def step(self, action):
    return self.env.step(action)


def create_agent(sess, environment, summary_writer=None):
  """Creates a DQN agent.

  Simplified version of `dopamine.atari.train.create_agent`
  """
  return dqn_agent.DQNAgent(sess, num_actions=environment.action_space.n,
                            summary_writer=summary_writer,
                            tf_device='/cpu:*')  # TODO: put gpu here!!!!

def get_create_env_fun(env_spec, world_model_dir):
  simulated = env_spec.simulated_env
  def create_env_fun(_1, _2):
    if simulated:
      batch_env = SimulatedBatchGymEnv(env_spec, 1, model_dir=world_model_dir)
    else:
      batch_env = env_spec.env
    env = FlatBatchEnv(batch_env)
    env = ResizeObservation(env)

    if simulated:
      # TODO: check how timesteps limit is choosen in PPO
      # (hardcoded? passed by hparams?)
      env = TimeLimit(env, max_episode_steps=100)

    # This needs to be top level wrapper for dopamine.
    env = AddGameOverProperty(env)
    return env
  return create_env_fun


def dopamine_trainer(hparams, model_dir):
  """ Simplified version of `dopamine.atari.train.create_runner` """
  global _dopamine_path

  assert _dopamine_path is not None, "Dopamine not available. Please install from " \
                                     "https://github.com/google/dopamine and add to PYTHONPATH"
  # TODO: how to make appopriate number of training steps? Can we safely change
  # training_steps Runner argument (note that these are per iteration)?
  # Take into consideration that each iteration == checkpoint. We want
  # checkpoints when training on simulated env, but not in the middle of real
  # env training (to not gather to much data when restarting experiment)
  # TODO: pass and clean up hparams
  # num_iterations are mocked for debug (/development) Run first iteration
  # when training on real env, continue on simulated.
  if hparams.environment_spec.simulated_env:
    num_iterations = 2
  else:
    num_iterations = 1
  # TODO: this must be 0 for real_env (to not generate addtionall data for
  # world-model, but maybe we can use it in simulated env?
  evaluation_steps = 0
  training_steps = 100

  with gin.unlock_config():# This is slight wierdness due to multiple runs DQN
    # TODO(pm): Think how to integrate with dopamine.
    gin_files = [os.path.join(_dopamine_path, 'agents/dqn/configs/dqn.gin')]
    run_experiment.load_gin_configs(gin_files, [])

  with tf.Graph().as_default():
    runner = run_experiment.Runner(
        model_dir, create_agent,
        create_environment_fn=get_create_env_fun(hparams.environment_spec,
                                                 hparams.get('world_model_dir', None)),
        num_iterations=num_iterations,
        training_steps=training_steps,
        evaluation_steps=evaluation_steps,
        max_steps_per_episode=20  # TODO: remove this
    )

    runner.run_experiment()
