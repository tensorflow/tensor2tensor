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

"""Utilities for player.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from copy import deepcopy

import gym
import six
from gym import wrappers, spaces
from gym.core import Env
from gym.envs.atari.atari_env import ACTION_MEANING
from gym.spaces import Box

import numpy as np

import rl_utils
from envs.simulated_batch_gym_env import FlatBatchEnv
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.rl.trainer_model_based import FLAGS, make_simulated_env_fn, \
  random_rollout_subsequences, PIL_Image, PIL_ImageDraw
from tensor2tensor.rl.trainer_model_based import setup_directories

from tensor2tensor.utils import registry, trainer_lib
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


def make_simulated_env(real_env, world_model_dir, hparams, random_starts):
  # Based on train_agent() from rlmb pipeline.
  frame_stack_size = hparams.frame_stack_size
  initial_frame_rollouts = real_env.current_epoch_rollouts(
      split=tf.contrib.learn.ModeKeys.TRAIN,
      minimal_rollout_frames=frame_stack_size,
  )
  def initial_frame_chooser(batch_size):
    """Frame chooser."""

    deterministic_initial_frames =\
        initial_frame_rollouts[0][:frame_stack_size]
    if not random_starts:
      # Deterministic starts: repeat first frames from the first rollout.
      initial_frames = [deterministic_initial_frames] * batch_size
    else:
      # Random starts: choose random initial frames from random rollouts.
      initial_frames = random_rollout_subsequences(
          initial_frame_rollouts, batch_size, frame_stack_size
      )
    return np.stack([
        [frame.observation.decode() for frame in initial_frame_stack]
        for initial_frame_stack in initial_frames
    ])
  env_fn = make_simulated_env_fn(
      real_env, hparams,
      batch_size=1,
      initial_frame_chooser=initial_frame_chooser,
      model_dir=world_model_dir
  )
  env = env_fn(in_graph=False)
  flat_env = FlatBatchEnv(env)
  return flat_env


def last_epoch(data_dir):
  """Infer highest epoch number from file names in data_dir."""
  names = os.listdir(data_dir)
  epochs_str = [re.findall(pattern='.*\.(-?\d+)$', string=name)
                for name in names]
  epochs_str = sum(epochs_str, [])
  return max([int(epoch_str) for epoch_str in epochs_str])


class SimulatedEnv(Env):
  def __init__(self, output_dir, hparams, which_epoch_data='last',
               random_starts=True):
    """"Gym environment interface for simulated environment."""
    hparams = deepcopy(hparams)
    self._output_dir = output_dir

    subdirectories = [
      "data", "tmp", "world_model", ("world_model", "debug_videos"),
      "policy", "eval_metrics"
    ]
    directories = setup_directories(output_dir, subdirectories)
    data_dir = directories["data"]

    if which_epoch_data == 'last':
      which_epoch_data = last_epoch(data_dir)
    assert isinstance(which_epoch_data, int), \
        '{}'.format(type(which_epoch_data))

    self.t2t_env = rl_utils.setup_env(
      hparams, batch_size=hparams.real_batch_size,
      max_num_noops=hparams.max_num_noops
    )

    # Load data.
    self.t2t_env.start_new_epoch(which_epoch_data, data_dir)

    self.env = make_simulated_env(self.t2t_env, directories["world_model"],
                                  hparams, random_starts=random_starts)

  def step(self, *args, **kwargs):
    ob, reward, done, info = self.env.step(*args, **kwargs)
    return ob, reward, done, info

  def reset(self):
    return self.env.reset()

  @property
  def observation_space(self):
    return self.t2t_env.observation_space

  @property
  def action_space(self):
    return self.t2t_env.action_space


class MockEnv(SimulatedEnv):
  def __init__(self, *args, **kwargs):
    self.env = gym.make('PongDeterministic-v4')
    self.t2t_env = gym.make('PongDeterministic-v4')


class MockWrapper(gym.Wrapper):
  def step(self, action):
    return self.env.step(action)

  def reset(self):
    return self.env.reset()


class ExtendToEvenDimentions(gym.ObservationWrapper):
  """ Force even dimentions of both height and width.

  Adds single zero row/column if needed.
  """
  HW_AXES = (0, 1)
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)

    orig_shape = env.observation_space.shape
    extended_shape = list(orig_shape)
    for axis in self.HW_AXES:
      if self.if_odd(orig_shape[axis]):
        extended_shape[axis] += 1

    assert env.observation_space.dtype == np.uint8
    self.observation_space = spaces.Box(
        low=0,
        high=255,
        shape=extended_shape,
        dtype=np.uint8)

  def observation(self, frame):
    if frame.shape == self.observation_space.shape:
      return frame
    else:
      extended_frame = np.zeros(self.observation_space.shape,
                                self.observation_space.dtype)
      assert self.HW_AXES == (0, 1)
      extended_frame[:frame.shape[0], :frame.shape[1]] = frame
      return extended_frame

  def if_odd(self, n):
    return n % 2


class RenderObservations(gym.Wrapper):
  def __init__(self, env):
    super(RenderObservations, self).__init__(env)
    if 'rgb_array' not in self.metadata['render.modes']:
      self.metadata['render.modes'].append('rgb_array')

  def step(self, action):
    ret = self.env.step(action)
    self.last_ob = ret[0]
    return ret

  def reset(self, **kwargs):
    self.last_ob = self.env.reset(**kwargs)
    return self.last_ob

  def render(self, mode='human', **kwargs):
    assert mode == 'rgb_array'
    return self.last_ob


def wrap_with_monitor(env, video_dir):
  env = ExtendToEvenDimentions(env)
  env = RenderObservations(env)
  env = wrappers.Monitor(env, video_dir, force=True,
                         video_callable=lambda idx: True,
                         write_upon_reset=True)
  return env


def create_simulated_env(
        output_dir, grayscale, resize_width_factor, resize_height_factor,
        frame_stack_size, generative_model, generative_model_params,
        random_starts=True, which_epoch_data='last', **other_hparams
):
  # We need these, to initialize T2TGymEnv, but these values (hopefully) have
  # no effect on player.
  a_bit_risky_defaults = {
    'game': 'pong',  # assumes that T2TGymEnv has always reward_range (-1,1)
    'real_batch_size': 1,
    'rl_env_max_episode_steps': -1,
    'max_num_noops': 0
  }

  for key in a_bit_risky_defaults:
    if key not in other_hparams:
      other_hparams[key] = a_bit_risky_defaults[key]


  hparams = tf.contrib.training.HParams(
    grayscale=grayscale,
    resize_width_factor=resize_width_factor,
    resize_height_factor=resize_height_factor,
    frame_stack_size=frame_stack_size,
    generative_model=generative_model,
    generative_model_params=generative_model_params,
    **other_hparams
  )
  return SimulatedEnv(output_dir, hparams, which_epoch_data=which_epoch_data,
                      random_starts=random_starts)


class PPOPolicyInferencer(object):
  """Non-tensorflow api for infering policy.

  Initialize fram
  """
  def __init__(self, hparams, action_space, observation_space, policy_dir):
    assert hparams.base_algo == 'ppo'
    ppo_hparams = trainer_lib.create_hparams(hparams.base_algo_params)

    frame_stack_shape = (1, hparams.frame_stack_size) + observation_space.shape
    self._frame_stack = np.zeros(frame_stack_shape, dtype=np.uint8)

    with tf.Graph().as_default():
      self.obs_t = tf.placeholder(shape=self.frame_stack_shape, dtype=np.uint8)
      self.logits_t, self.value_function_t = get_policy(
        self.obs_t, ppo_hparams, action_space
      )
      model_saver = tf.train.Saver(
        tf.global_variables(ppo_hparams.policy_network + "/.*")
      )
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
      trainer_lib.restore_checkpoint(policy_dir, model_saver,
                                     self.sess)

  @property
  def frame_stack_shape(self):
    return self._frame_stack.shape

  def reset_frame_stack(self, frame_stack=None):
    if frame_stack is None:
      self._frame_stack.fill(0)
    else:
      assert frame_stack.shape == self.frame_stack_shape, \
        '{}, {}'.format(frame_stack.shape, self.frame_stack_shape)
      self._frame_stack = frame_stack.copy()

  def _add_to_stack(self, ob):
    stack = np.roll(self._frame_stack, shift=-1, axis=1)
    stack[0, -1, ...] = ob
    self._frame_stack = stack

  def infer(self, ob):
    """Add new observation to frame stack and infer.

    Args:
      ob: array of shape (height, width, channels)
    """
    self._add_to_stack(ob)
    logits, vf = self.infer_from_frame_stack(self._frame_stack)
    return logits, vf

  def infer_from_frame_stack(self, ob_stack):
    """ Infer from stack of observations

    Args:
      ob_stack: array of shape (1, frame_stack_size, height, width, channels)
    """
    logits, vf = self.sess.run([self.logits_t, self.value_function_t],
                               feed_dict={self.obs_t: ob_stack})
    return logits, vf
