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

"""Play with a world model.

Run this script with the same parameters as trainer_model_based.py. Note that
values of most of them have no effect on player, so running just

python -m tensor2tensor/rl/player.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base

might work for you.

Controls:
  WSAD and SPACE to control the agent.
  R key to reset env.
  C key to toggle WAIT mode.
  N to perform NOOP action under WAIT mode.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from copy import deepcopy

import gym
import six
from gym import wrappers
from gym.core import Env
from gym.envs.atari.atari_env import ACTION_MEANING
from gym.spaces import Box

import numpy as np

import rl_utils
from envs.simulated_batch_gym_env import FlatBatchEnv
from tensor2tensor.rl.trainer_model_based import FLAGS, make_simulated_env_fn, \
  random_rollout_subsequences, PIL_Image, PIL_ImageDraw
from tensor2tensor.rl.trainer_model_based import setup_directories

from tensor2tensor.utils import registry
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("video_dir", "/tmp/gym-results",
                    "Where to save played trajectories.")
flags.DEFINE_string("zoom", '4',
                    "Resize factor of displayed game.")
flags.DEFINE_string("fps", '20',
                    "Frames per second.")
flags.DEFINE_string("epoch", 'last',
                    "Data from which epoch to use.")
flags.DEFINE_string("env", 'simulated',
                    "Either to use 'simulated' or 'real' env.")


def make_simulated_env(real_env, world_model_dir, hparams, random_starts):
  # Based on train_agent() from rlmb pipeline.
  frame_stack_size = hparams.frame_stack_size
  initial_frame_rollouts = real_env.current_epoch_rollouts(
      split=tf.contrib.learn.ModeKeys.TRAIN,
      minimal_rollout_frames=frame_stack_size,
  )
  # TODO: use the same version as train_agent? But skip
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


class PlayerEnvWrapper(gym.Wrapper):

  RESET_ACTION = 101
  TOGGLE_WAIT_ACTION = 102
  WAIT_MODE_NOOP_ACTION = 103

  HEADER_HEIGHT = 12

  def __init__(self, env):
    super(PlayerEnvWrapper, self).__init__(env)

    # Set observation space
    orig = self.env.observation_space
    shape = tuple([orig.shape[0] + self.HEADER_HEIGHT] + list(orig.shape[1:]))
    self.observation_space = gym.spaces.Box(low=orig.low.min(),
                                            high=orig.high.max(),
                                            shape=shape, dtype=orig.dtype)

    # gym play() looks for get_keys_to_action() only on top and bottom level
    # of env and wrappers stack.
    self.unwrapped.get_keys_to_action = self.get_keys_to_action

    self._wait = True
    self.action_meaning = {i: ACTION_MEANING[i]
                           for i in range(self.action_space.n)}
    self.name_to_action_num = {v: k for k, v in
                               six.iteritems(self.action_meaning)}

  def get_action_meanings(self):
    return [self.action_meaning[i] for i in range(self.action_space.n)]

  def get_keys_to_action(self):
    # Based on gym atari.py AtariEnv.get_keys_to_action()
    KEYWORD_TO_KEY = {
      'UP': ord('w'),
      'DOWN': ord('s'),
      'LEFT': ord('a'),
      'RIGHT': ord('d'),
      'FIRE': ord(' '),
    }

    keys_to_action = {}

    for action_id, action_meaning in enumerate(self.get_action_meanings()):
      keys = []
      for keyword, key in KEYWORD_TO_KEY.items():
        if keyword in action_meaning:
          keys.append(key)
      keys = tuple(sorted(keys))

      assert keys not in keys_to_action
      keys_to_action[keys] = action_id

    # Add utility actions
    keys_to_action[(ord("r"),)] = self.RESET_ACTION
    keys_to_action[(ord("c"),)] = self.TOGGLE_WAIT_ACTION
    keys_to_action[(ord("n"),)] = self.WAIT_MODE_NOOP_ACTION

    return keys_to_action

  def step(self, action):
    # Special codes
    if action == self.TOGGLE_WAIT_ACTION:
      self._wait = not self._wait
      ob, reward, done, info = self._last_step
      ob = self.augment_observation(ob, reward, self.total_reward)
      return ob, reward, done, info

    if action == self.RESET_ACTION:
      ob = self.empty_observation()
      return ob, 0, True, {}

    if self._wait and action == self.name_to_action_num['NOOP']:
      ob, reward, done, info = self._last_step
      ob = self.augment_observation(ob, reward, self.total_reward)
      return ob, reward, done, info

    if action == self.WAIT_MODE_NOOP_ACTION:
      action = self.name_to_action_num['NOOP']


    ob, reward, done, info = self.env.step(action)
    self._last_step = ob, reward, done, info

    self.total_reward += reward

    ob = self.augment_observation(ob, reward, self.total_reward)
    return ob, reward, done, info

  def reset(self):
    ob = self.env.reset()
    self._last_step = ob, 0, False, {}
    self.total_reward = 0
    return self.augment_observation(ob, 0, self.total_reward)

  def empty_observation(self):
    return np.zeros(self.observation_space.shape)

  def augment_observation(self, ob, reward, total_reward):
    img = PIL_Image().new("RGB",
                          (ob.shape[1], PlayerEnvWrapper.HEADER_HEIGHT,))
    draw = PIL_ImageDraw().Draw(img)
    draw.text((1, 0), "c:{:3}, r:{:3}".format(int(total_reward), int(reward)),
              fill=(255, 0, 0))
    header = np.asarray(img)
    del img
    header.setflags(write=1)
    if self._wait:
      pixel_fill = (0, 255, 0)
    else:
      pixel_fill = (255, 0, 0)
    header[0, :, :] = pixel_fill
    return np.concatenate([header, ob], axis=0)


class MockEnv(SimulatedEnv):
  def __init__(self, *args, **kwargs):
    self.env = gym.make('PongDeterministic-v4')
    self.t2t_env = gym.make('PongDeterministic-v4')

class MockWrapper(gym.Wrapper):
  def step(self, action):
    return self.env.step(action)

  def reset(self):
    return self.env.reset()


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


def main(_):
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  # TODO(konradczechowski) remove this?
  if 'wm_policy_param_sharing' not in hparams.values().keys():
    hparams.add_hparam('wm_policy_param_sharing', False)
  output_dir = FLAGS.output_dir
  video_dir = FLAGS.video_dir
  fps = int(FLAGS.fps)
  zoom = int(FLAGS.zoom)
  epoch = FLAGS.epoch if FLAGS.epoch == 'last' else int(FLAGS.epoch)

  # Two options to initialize env:
  # 1 - with hparams from rlmb run
  if FLAGS.env == "simulated":
    env = SimulatedEnv(output_dir, hparams)
  elif FLAGS.env == "real":
    env = MockEnv()
  else:
    raise ValueError("Invalid 'env' flag {}".format(FLAGS.env))

  # 2 - explicitly with minimal parameters required.
  # env = create_simulated_env(
  #     output_dir=output_dir, grayscale=hparams.grayscale,
  #     resize_width_factor=hparams.resize_width_factor,
  #     resize_height_factor=hparams.resize_height_factor,
  #     frame_stack_size=hparams.frame_stack_size,
  #     epochs=hparams.epochs,
  #     generative_model=hparams.generative_model,
  #     generative_model_params=hparams.generative_model_params,
  #     intrinsic_reward_scale=0.,
  # )

  env = PlayerEnvWrapper(env)

  env = wrappers.Monitor(env, video_dir, force=True,
                         write_upon_reset=True)

  # env.reset()
  # for i in range(50):
  #   env.step(i % 3)
  # k2a = PlayerEnvWrapper.get_keys_to_action(env)
  from gym.utils import play
  play.play(env, zoom=zoom, fps=fps)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
