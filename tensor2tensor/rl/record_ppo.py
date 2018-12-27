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

"""

Run this script with the same parameters as trainer_model_based.py /
trainer_model_free.py. Note that values of most of them have no effect,
so running just

python -m tensor2tensor/rl/record_ppo.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base \
    --loop_hparams=game=right_game

might work for you.
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
from gym.wrappers import TimeLimit

import rl_utils
from envs.simulated_batch_gym_env import FlatBatchEnv
from player_utils import SimulatedEnv, wrap_with_monitor, PPOPolicyInferencer
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.rl.trainer_model_based import FLAGS
from tensor2tensor.rl.trainer_model_based import setup_directories

from tensor2tensor.utils import registry, trainer_lib
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("video_dir", "/tmp/record_ppo_out",
                    "Where to save recorded trajectories.")
flags.DEFINE_string("epoch", 'last',
                    "Data from which epoch to use.")
flags.DEFINE_string("env", 'simulated',
                    "Either to use 'simulated' or 'real' env.")
flags.DEFINE_string("simulated_episode_len", '100',
                    "Timesteps limit for simulated env")
flags.DEFINE_string("num_episodes", '20',
                    "How many episodes record.")


def main(_):
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  # TODO(konradczechowski) remove this?
  if 'wm_policy_param_sharing' not in hparams.values().keys():
    hparams.add_hparam('wm_policy_param_sharing', False)
  output_dir = FLAGS.output_dir
  video_dir = FLAGS.video_dir
  epoch = FLAGS.epoch if FLAGS.epoch == 'last' else int(FLAGS.epoch)
  simulated_episode_len = int(FLAGS.simulated_episode_len)
  num_episodes = int(FLAGS.num_episodes)

  if FLAGS.env == "simulated":
    env = SimulatedEnv(output_dir, hparams, which_epoch_data=epoch)
    env = TimeLimit(env, max_episode_steps=simulated_episode_len)
  elif FLAGS.env == "real":
    # TODO(konradczechowski): Implement this
    raise NotImplementedError
    # env = MockEnv()
  else:
    raise ValueError("Invalid 'env' flag {}".format(FLAGS.env))

  env = wrap_with_monitor(env, video_dir=video_dir)

  subdirectories = [
    "data", "tmp", "world_model", ("world_model", "debug_videos"),
    "policy", "eval_metrics"
  ]
  directories = setup_directories(output_dir, subdirectories)

  policy_dir = directories['policy']
  action_space = env.action_space
  observation_space = env.observation_space
  pi = PPOPolicyInferencer(hparams, action_space, observation_space,
                           policy_dir)

  pi.reset_frame_stack()
  ob = env.reset()
  for _ in range(num_episodes):
    done = False
    while not done:
      logits, vf = pi.infer(ob)
      probs = np.exp(logits) / np.sum(np.exp(logits))
      action = np.random.choice(probs.size, p=probs[0])
      ob, rew, done, _ = env.step(action)
    ob = env.reset()
    pi.reset_frame_stack()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
