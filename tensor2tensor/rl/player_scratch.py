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

"""Play with a world model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

from gym.core import Env
from gym.spaces import Box
from gym.spaces import Discrete
# from gym.utils import play

import numpy as np

import rl_utils
from envs.simulated_batch_gym_env import FlatBatchEnv
from tensor2tensor.rl.trainer_model_based import FLAGS, make_simulated_env_fn, \
  random_rollout_subsequences
from tensor2tensor.rl.trainer_model_based import setup_directories

from tensor2tensor.utils import registry, misc_utils
import tensorflow as tf


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


class SimulatedEnv(Env):
  def __init__(self, output_dir, hparams, last_epoch=None, random_starts=True):
    """" Gym environment interface for simulated environment.

    Args:
      last_epoch: number of last epoch stored on disk. Frames from this and
      earlier epoch would be used at restarts of simulated env. Note that world
      model would be loaded from latest checkpoint regardless of this argument.
      Assumes that there are all integer epochs from -1 to last_epoch stored on
      disk. Default uses hparams.epoch -1 (assumes finished training)
    """
    hparams = deepcopy(hparams)
    self._output_dir = output_dir

    if last_epoch is None:
      # TODO(konradczechowski): read filenames from disk?
      last_epoch = hparams.epochs - 1

    subdirectories = [
      "data", "tmp", "world_model", ("world_model", "debug_videos"),
      "policy", "eval_metrics"
    ]
    directories = setup_directories(output_dir, subdirectories)


    data_dir = directories["data"]

    t2t_env = rl_utils.setup_env(
      hparams, batch_size=hparams.real_batch_size,
      max_num_noops=hparams.max_num_noops
    )
    # Load data from epochs
    for epoch in range(-1, last_epoch + 1):
      t2t_env.start_new_epoch(epoch, data_dir)
      # train_agent_real_env(env, learner, hparams, epoch)
      t2t_env.generate_data(data_dir)

    self.env = make_simulated_env(t2t_env, directories["world_model"], hparams,
                                  random_starts=random_starts)

  def step(self, *args, **kwargs):
    return self.env.step(*args, **kwargs)

  def reset(self):
    return self.env.reset()


def create_simulated_env(
        output_dir, grayscale, resize_width_factor, resize_height_factor,
        frame_stack_size, epochs, generative_model, generative_model_params,
        random_starts=True, last_epoch=None, **other_hparams
):
  # We need these, to initialize T2TGymEnv, but these values (hopefully) are
  # not needed.
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
    epochs=epochs,
    grayscale=grayscale,
    resize_width_factor=resize_width_factor,
    resize_height_factor=resize_height_factor,
    frame_stack_size=frame_stack_size,
    generative_model=generative_model,
    generative_model_params=generative_model_params,
    **other_hparams
  )
  return SimulatedEnv(output_dir, hparams, last_epoch=last_epoch,
                      random_starts=random_starts)


def main(_):
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  output_dir = FLAGS.output_dir

  # Two options to initialize env:
  # 1 - with hparams from rlmb run
  env1 = SimulatedEnv(output_dir, hparams)

  # 2 - explicitly with minimal parameters required.
  env2 = create_simulated_env(
      output_dir=output_dir, grayscale=hparams.grayscale,
      resize_width_factor=hparams.resize_width_factor,
      resize_height_factor=hparams.resize_height_factor,
      frame_stack_size=hparams.frame_stack_size,
      epochs=hparams.epochs,
      generative_model=hparams.generative_model,
      generative_model_params=hparams.generative_model_params,
      intrinsic_reward_scale=0.,
  )

  for env in [env1, env2]:
    for _ in range(5):
      ob = env.reset()
      for i in range(50):
        ob, rew, _, _ = env.step(i % 3)

  a = 1


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
