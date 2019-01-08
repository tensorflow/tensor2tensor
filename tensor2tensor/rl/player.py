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

r"""Play with a world model.

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

import gym
from gym.envs.atari.atari_env import ACTION_MEANING
try:
  from gym.utils import play
except:
  pass
import numpy as np
import six

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.rl import player_utils
from tensor2tensor.rl.envs.simulated_batch_env import PIL_Image
from tensor2tensor.rl.envs.simulated_batch_env import PIL_ImageDraw
from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv
# Import flags from t2t_trainer and trainer_model_based
import tensor2tensor.rl.trainer_model_based_params  # pylint: disable=unused-import
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


class PlayerEnv(gym.Env):
  """Environment Wrapper for gym.utils.play."""

  RETURN_DONE_ACTION = 101
  TOGGLE_WAIT_ACTION = 102
  WAIT_MODE_NOOP_ACTION = 103

  HEADER_HEIGHT = 12

  def __init__(self):
    self._wait = True

  def init_action_space_from_env(self, env):
    self.action_space = env.action_space

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
    keys_to_action[(ord("r"),)] = self.RETURN_DONE_ACTION
    keys_to_action[(ord("c"),)] = self.TOGGLE_WAIT_ACTION
    keys_to_action[(ord("n"),)] = self.WAIT_MODE_NOOP_ACTION

    return keys_to_action

  def player_actions(self):
    return {
        self.RETURN_DONE_ACTION: self.player_return_done_action,
        self.TOGGLE_WAIT_ACTION: self.player_toggle_wait_action,
    }

  def player_toggle_wait_action(self):
    self._wait = not self._wait
    return self._last_step_returns

  def player_return_done_action(self):
    raise NotImplementedError

  def step(self, action):
    # Special codes
    if action in self.player_actions():
      envs_step_returns = self.player_actions()[action]()
    elif self._wait and action == self.name_to_action_num["NOOP"]:
      # Ignore no-op, do not pass to environment.
      envs_step_returns = self._last_step_returns
    else:
      # Run action on environment(s).
      if action == self.WAIT_MODE_NOOP_ACTION:
        action = self.name_to_action_num["NOOP"]
      # normal action to pass to env
      envs_step_returns = self.pass_action_to_envs(action)
      self.update_statistics(envs_step_returns)

    self._last_step_returns = envs_step_returns
    ob, reward, done, info = self.construct_step_return(envs_step_returns)
    return ob, reward, done, info

  def augment_observation(self, ob, reward, total_reward):
    img = PIL_Image().new("RGB",
                          (ob.shape[1], self.HEADER_HEIGHT,))
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

  def pass_action_to_envs(self, action):
    raise NotImplementedError

  def update_statistics(self, envs_step_returns):
    raise NotImplementedError

  def construct_step_return(self, envs_step_returns):
    raise NotImplementedError


class SimAndRealEnvPlayer(PlayerEnv):
  def __init__(self, real_env, sim_env):
    super().__init__()
    self.real_env = real_env
    self.sim_env = sim_env
    # TODO: Set observation space
    # orig = self.env.observation_space
    # shape = tuple([orig.shape[0] + self.HEADER_HEIGHT] + list(orig.shape[1:]))
    # self.observation_space = gym.spaces.Box(low=orig.low.min(),
    #                                         high=orig.high.max(),
    #                                         shape=shape, dtype=orig.dtype)
    self.init_action_space_from_env(sim_env)



class SingleEnvPlayer(PlayerEnv):

  def __init__(self, env):
    super().__init__()
    self.env = env
    # Set observation space
    orig = self.env.observation_space
    shape = tuple([orig.shape[0] + self.HEADER_HEIGHT] + list(orig.shape[1:]))
    self.observation_space = gym.spaces.Box(low=orig.low.min(),
                                            high=orig.high.max(),
                                            shape=shape, dtype=orig.dtype)
    self.init_action_space_from_env(env)

  def construct_step_return(self, envs_step_returns):
    ob, reward, done, info = envs_step_returns['env']
    ob = self.augment_observation(ob, reward, self.total_reward)
    return ob, reward, done, info

  def pack_step_return(self, ob, reward, done, info):
    return dict(env=(ob, reward, done, info))

  def pass_action_to_envs(self, action):
    return self.pack_step_return(*self.env.step(action))

  def reset(self):
    ob = self.env.reset()
    self._last_step_returns = self.pack_step_return(ob, 0, False, {})
    self.total_reward = 0
    return self.augment_observation(ob, 0, self.total_reward)

  def update_statistics(self, envs_step_returns):
    reward = envs_step_returns['env'][1]
    self.total_reward += reward

  def empty_observation(self):
    return np.zeros(self.env.observation_space.shape)

  def player_return_done_action(self):
    ob = self.empty_observation()
    return self.pack_step_return(ob, 0, True, {})


def main(_):
  # gym.logger.set_level(gym.logger.DEBUG)
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  # Not important for experiments past 2018
  if "wm_policy_param_sharing" not in hparams.values().keys():
    hparams.add_hparam("wm_policy_param_sharing", False)
  directories = player_utils.infer_paths(
      output_dir=FLAGS.output_dir,
      world_model=FLAGS.wm_dir,
      policy=FLAGS.policy_dir,
      data=FLAGS.episodes_data_dir)
  epoch = FLAGS.epoch if FLAGS.epoch == "last" else int(FLAGS.epoch)

  if FLAGS.simulated_env:
    env = player_utils.load_data_and_make_simulated_env(
        directories["data"], directories["world_model"],
        hparams, which_epoch_data=epoch)
  else:
    env = T2TGymEnv.setup_and_load_epoch(
        hparams, data_dir=directories["data"],
        which_epoch_data=None)
    env = FlatBatchEnv(env)

  env = SingleEnvPlayer(env)  # pylint: disable=redefined-variable-type

  # TODO:TMP
  # for i in range(100):
  #   ob = env.reset()
  #   print("mean ob", np.mean(ob))
  #   player_utils._FRAMES_FOR_CHOOSER.fill(i)
  # ######

  env = player_utils.wrap_with_monitor(env, FLAGS.video_dir)

  if FLAGS.dry_run:
    for _ in range(5):
      env.reset()
      for i in range(50):
        env.step(i % 3)
      env.step(PlayerEnv.RETURN_DONE_ACTION)  # reset
    return

  play.play(env, zoom=FLAGS.zoom, fps=FLAGS.fps)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
