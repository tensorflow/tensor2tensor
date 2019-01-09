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
  X to reset simulated env only, when running sim-real comparison.

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
    --sim_and_real=False \
    --simulated_env=False \
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

from rl_utils import absolute_hinge_difference

# TODO(konradczechowski): try-except is only for development, remove this.
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
flags.DEFINE_boolean("sim_and_real", True,
                     "Compare simulated and real environment.")
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
  """Base (abstract) environment with custom actions for gym.utils.play.

  Notation:
    envs_step_tuples: dict(env_name=(observation, reward, done, info), ...)
      Dictionary of tuples similar to those returned by gym.Env.step().
  """

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
      envs_step_tuples = self.player_actions()[action]()
    elif self._wait and action == self.name_to_action_num["NOOP"]:
      # Ignore no-op, do not pass to environment.
      envs_step_tuples = self._last_step_returns
    else:
      # Run action on environment(s).
      if action == self.WAIT_MODE_NOOP_ACTION:
        action = self.name_to_action_num["NOOP"]
      # normal action to pass to env
      envs_step_tuples = self.step_envs(action)
      self.update_statistics(envs_step_tuples)

    self._last_step_returns = envs_step_tuples
    ob, reward, done, info = self.player_step_tuple(envs_step_tuples)
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

  def step_envs(self, action):
    raise NotImplementedError

  def update_statistics(self, envs_step_tuples):
    raise NotImplementedError

  def player_step_tuple(self, envs_step_tuples):
    raise NotImplementedError


class SimAndRealEnvPlayer(PlayerEnv):
  # TODO(konradczechowski): document

  RESTART_SIMULATED_ENV_ACTION = 110

  def __init__(self, real_env, sim_env):
    super(SimAndRealEnvPlayer, self).__init__()
    assert real_env.observation_space.shape == sim_env.observation_space.shape
    self.real_env = real_env
    self.sim_env = sim_env
    orig = self.real_env.observation_space
    # Observation consists three side-to-side images - simulated environment
    # observation, real environment observation and difference between these
    # two.
    shape = (orig.shape[0] + self.HEADER_HEIGHT, orig.shape[1] * 3,
             orig.shape[2])

    self.observation_space = gym.spaces.Box(low=orig.low.min(),
                                            high=orig.high.max(),
                                            shape=shape, dtype=orig.dtype)
    self.init_action_space_from_env(sim_env)

  def player_actions(self):
    actions = super(SimAndRealEnvPlayer, self).player_actions()
    actions.update({
        self.RESTART_SIMULATED_ENV_ACTION:
            self.player_restart_simulated_env_action,
    })
    return actions

  def get_keys_to_action(self):
    keys_to_action = super(SimAndRealEnvPlayer, self).get_keys_to_action()
    keys_to_action[(ord("x"),)] = self.RESTART_SIMULATED_ENV_ACTION
    return keys_to_action

  def player_step_tuple(self, envs_step_tuples):
    ob_real, reward_real = envs_step_tuples['real_env'][:2]
    ob_sim, reward_sim = envs_step_tuples['sim_env'][:2]
    ob_err = absolute_hinge_difference(ob_sim, ob_real)

    ob_real_aug = self.augment_observation(ob_real, reward_real,
                                           self.total_real_reward)
    ob_sim_aug = self.augment_observation(ob_sim, reward_sim,
                                          self.total_sim_reward)
    ob_err_aug = self.augment_observation(
      ob_err, reward_sim - reward_real,
      self.total_sim_reward - self.total_real_reward
    )
    ob = np.concatenate([ob_sim_aug, ob_real_aug, ob_err_aug], axis=1)
    reward = reward_real
    done = envs_step_tuples['real_env'][2]
    info = envs_step_tuples['real_env'][3]
    return ob, reward, done, info

  def reset(self):
    ob_real = self.real_env.reset()
    self.sim_env.add_to_initial_stack(ob_real)
    # TODO(konradczechowski): remove when not longer needed.
    # for i in range(12):
    #   ob_real, _, _, _ = self.real_env.step(np.random.choice([2,3, 4, 5]))
    #   self.sim_env.add_to_initial_stack(ob_real)
    for i in range(3):
      ob_real, _, _, _ = self.real_env.step(self.name_to_action_num['NOOP'])
      self.sim_env.add_to_initial_stack(ob_real)
    ob_sim = self.sim_env.reset()
    assert np.all(ob_real == ob_sim)
    self._last_step_returns = self.pack_step_return((ob_real, 0, False, {}),
                                                    (ob_sim, 0, False, {}))
    self.set_zero_total_rewards()
    ob, _, _, _ = self.player_step_tuple(self._last_step_returns)
    return ob

  def pack_step_return(self, real_env_step_tuple, sim_env_step_tuple):
    return dict(real_env=real_env_step_tuple,
                sim_env=sim_env_step_tuple)

  def set_zero_total_rewards(self):
    self.total_real_reward = 0
    self.total_sim_reward = 0

  def step_envs(self, action):
    """Perform step, update initial_frame_stack for simulated environment."""
    real_env_step_tuple = self.real_env.step(action)
    sim_env_step_tuple = self.sim_env.step(action)
    self.sim_env.add_to_initial_stack(real_env_step_tuple[0])
    return self.pack_step_return(real_env_step_tuple, sim_env_step_tuple)

  def update_statistics(self, envs_step_tuples):
    self.total_real_reward += envs_step_tuples['real_env'][1]
    self.total_sim_reward += envs_step_tuples['sim_env'][1]

  def player_return_done_action(self):
    ob = np.zeros(self.real_env.observation_space.shape, dtype=np.uint8)
    return self.pack_step_return((ob, 0, True, {}),
                                 (ob, 0, True, {}))

  def player_restart_simulated_env_action(self):
    ob = self.sim_env.reset()

    # TODO(konradczechowski): remove when this will be not needed
    # new_ob, _, _, _ = self.sim_env.step(2)
    # print("\n\n\n\ndiff {}\n\n\n\n".format((ob - new_ob).sum()))
    # ##########

    assert np.all(self._last_step_returns['real_env'][0] == ob)
    self.set_zero_total_rewards()
    return self.pack_step_return(self._last_step_returns['real_env'],
                                 (ob, 0, False, {}))


class SingleEnvPlayer(PlayerEnv):
  # TODO(konradczechowski): document

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

  def player_step_tuple(self, envs_step_tuples):
    ob, reward, done, info = envs_step_tuples['env']
    ob = self.augment_observation(ob, reward, self.total_reward)
    return ob, reward, done, info

  def pack_step_return(self, env_step_tuple):
    return dict(env=env_step_tuple)

  def reset(self):
    ob = self.env.reset()
    self._last_step_returns = self.pack_step_return((ob, 0, False, {}))
    self.total_reward = 0
    return self.augment_observation(ob, 0, self.total_reward)

  def step_envs(self, action):
    return self.pack_step_return(self.env.step(action))

  def update_statistics(self, envs_step_tuples):
    reward = envs_step_tuples['env'][1]
    self.total_reward += reward

  def player_return_done_action(self):
    ob = np.zeros(self.env.observation_space.shape, dtype=np.uint8)
    return self.pack_step_return((ob, 0, True, {}))


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


  def make_real_env():
    env = T2TGymEnv.setup_and_load_epoch(
            hparams, data_dir=directories["data"],
            which_epoch_data=None)
    env = FlatBatchEnv(env)
    return env

  def make_simulated_env(setable_initial_frames):
    env = player_utils.load_data_and_make_simulated_env(
        directories["data"], directories["world_model"],
        hparams, which_epoch_data=epoch,
        setable_initial_frames=setable_initial_frames)
    return env

  if FLAGS.sim_and_real:
    sim_env = make_simulated_env(setable_initial_frames=True)
    real_env = make_real_env()
    env = SimAndRealEnvPlayer(real_env, sim_env)
  else:
    if FLAGS.simulated_env:
      env = make_simulated_env(setable_initial_frames=False)
    else:
      env = make_real_env()
    env = SingleEnvPlayer(env)  # pylint: disable=redefined-variable-type

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
