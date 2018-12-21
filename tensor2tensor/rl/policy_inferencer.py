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

Run this script with the same parameters as trainer_model_based.py. Note that
values of most of them have no effect on player, so running just

python -m tensor2tensor/rl/player.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base

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

import rl_utils
from envs.simulated_batch_gym_env import FlatBatchEnv
from player import SimulatedEnv, MockEnv
from tensor2tensor.models.research.rl import get_policy
from tensor2tensor.rl.trainer_model_based import FLAGS, make_simulated_env_fn, \
  random_rollout_subsequences, PIL_Image, PIL_ImageDraw
from tensor2tensor.rl.trainer_model_based import setup_directories

from tensor2tensor.utils import registry, trainer_lib
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


# flags.DEFINE_string("video_dir", "/tmp/gym-results",
#                     "Where to save played trajectories.")
# flags.DEFINE_string("epoch", 'last',
#                     "Data from which epoch to use.")
# flags.DEFINE_string("env", 'simulated',
#                     "Either to use 'simulated' or 'real' env.")


class PPOPolicyInferencer(object):
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

  def reset_frame_stack(self):
    self._frame_stack.fill(0)

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
    logits, vf = self.sess.run([self.logits_t, self.value_function_t],
                               feed_dict={self.obs_t: self._frame_stack})
    return logits, vf

  def infer_from_frame_stack(self, ob_stack):
    """ Infer from stack of observations

    Args:
      ob_stack: array of shape (frame_stack_size, height, width, channels)
    """
    logits, vf = self.sess.run([self.logits_t, self.value_function_t],
                               feed_dict={self.obs_t: ob_stack})
    return logits, vf


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

  if FLAGS.env == "simulated":
    env = SimulatedEnv(output_dir, hparams)
  elif FLAGS.env == "real":
    env = MockEnv()
  else:
    raise ValueError("Invalid 'env' flag {}".format(FLAGS.env))

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
  ob = np.zeros(shape=pi.frame_stack_shape[2:], dtype=np.uint8)
  print(pi.infer(ob))


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
