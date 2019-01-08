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

Controls:
  WSAD and SPACE to control the agent.
  R key to reset env.
  C key to toggle WAIT mode.
  N to perform NOOP action under WAIT mode.

Run this script with the same parameters as trainer_model_based.py. Note that
values of most of them have no effect on player, so running just

python -m tensor2tensor/rl/player.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base

might work for you.

More advanced example:

python -m tensor2tensor/rl/record_ppo.py \
    --output_dir=path/to/your/experiment \
    --loop_hparams_set=rlmb_base \
    --loop_hparams=game=<right game in case of problems> \
    --video_dir=my/video/dir \
    --zoom=6 \
    --fps=50 \
    --env=real \
    --epoch=-1

Check flags definitions under imports for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

import gym
from gym.envs.atari.atari_env import ACTION_MEANING
from gym.utils import play
import numpy as np

from tensor2tensor.rl.envs.simulated_batch_env import PIL_Image, PIL_ImageDraw
from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv
from tensor2tensor.rl.player_utils import wrap_with_monitor, \
  load_data_and_make_simulated_env, infer_paths
# Import flags from t2t_trainer and trainer_model_based
from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
import tensor2tensor.rl.trainer_model_based_params # pylint: disable=unused-import

from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.utils import registry
import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("video_dir", "/tmp/gym-results",
                    "Where to save played trajectories.")
flags.DEFINE_float("zoom", 4.,
                   "Resize factor of displayed game.")
flags.DEFINE_float("fps", 20.,
                   "Frames per second.")
flags.DEFINE_string("epoch", "last",
                    "Data from which epoch to use.")
flags.DEFINE_boolean("simulated_env", True,
                     "Either to use 'simulated' or 'real' env.")
flags.DEFINE_boolean("dry_run", False,
                     "Dry run - without pygame interaction and display, just "
                     "some random actions on environment")
flags.DEFINE_string("model_ckpt", "",
                    "World model checkpoint path.")
flags.DEFINE_string("wm_dir", "",
                    "Directory with world model checkpoints. Inferred from "
                    "output_dir if empty.")
flags.DEFINE_string("policy_dir", "",
                    "Directory with policy. Inferred from output_dir if empty.")
flags.DEFINE_string("episodes_data_dir", "",
                    "Path to data for simulated environment initialization. "
                    "Inferred from output_dir if empty.")


class PlayerEnvWrapper(gym.Wrapper):
  """ Environment Wrapper for gym.utils.play.

  This probably will be highly refactored.
  """

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
    keyword_to_key = {
        "UP": ord("w"),
        "DOWN": ord("s"),
        "LEFT": ord("a"),
        "RIGHT": ord("d"),
        "FIRE": ord(" "),
    }

    keys_to_action = {}

    for action_id, action_meaning in enumerate(self.get_action_meanings()):
      keys = []
      for keyword, key in keyword_to_key.items():
        if keyword in action_meaning:
          keys.append(key)
      keys_tuple = tuple(sorted(keys))
      del keys
      assert keys_tuple not in keys_to_action
      keys_to_action[keys_tuple] = action_id

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

    if self._wait and action == self.name_to_action_num["NOOP"]:
      ob, reward, done, info = self._last_step
      ob = self.augment_observation(ob, reward, self.total_reward)
      return ob, reward, done, info

    if action == self.WAIT_MODE_NOOP_ACTION:
      action = self.name_to_action_num["NOOP"]

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


def main(_):
  # gym.logger.set_level(gym.logger.DEBUG)
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  # Not important for experiments past 2018
  if "wm_policy_param_sharing" not in hparams.values().keys():
    hparams.add_hparam("wm_policy_param_sharing", False)
  directories = infer_paths(output_dir=FLAGS.output_dir,
                            world_model=FLAGS.wm_dir,
                            policy=FLAGS.policy_dir,
                            data=FLAGS.episodes_data_dir)
  epoch = FLAGS.epoch if FLAGS.epoch == "last" else int(FLAGS.epoch)

  if FLAGS.simulated_env:
    env = load_data_and_make_simulated_env(directories["data"],
                                           directories["world_model"],
                                           hparams, which_epoch_data=epoch)
  else:
    env = T2TGymEnv.setup_and_load_epoch(
        hparams, data_dir=directories["data"],
        which_epoch_data=epoch)
    env = FlatBatchEnv(env)

  env = PlayerEnvWrapper(env)  # pylint: disable=redefined-variable-type

  env = wrap_with_monitor(env, FLAGS.video_dir)

  if FLAGS.dry_run:
    for _ in range(5):
      env.reset()
      for i in range(50):
        env.step(i % 3)
      env.step(PlayerEnvWrapper.RESET_ACTION)  # reset
    return

  play.play(env, zoom=FLAGS.zoom, fps=FLAGS.fps)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
