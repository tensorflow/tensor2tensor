# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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
    --loop_hparams=generative_model="next_frame" \
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
from gym.utils import play
import numpy as np

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.rl import player_utils
from tensor2tensor.rl.envs.simulated_batch_env import PIL_Image
from tensor2tensor.rl.envs.simulated_batch_env import PIL_ImageDraw
from tensor2tensor.rl.envs.simulated_batch_gym_env import FlatBatchEnv
from tensor2tensor.rl.rl_utils import absolute_hinge_difference
from tensor2tensor.rl.rl_utils import full_game_name
# Import flags from t2t_trainer and trainer_model_based
import tensor2tensor.rl.trainer_model_based_params  # pylint: disable=unused-import
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


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
flags.DEFINE_boolean("game_from_filenames", True,
                     "If infer game name from data_dir filenames or from "
                     "hparams.")


class PlayerEnv(gym.Env):
  """Base (abstract) environment for interactive human play with gym.utils.play.

  Additionally to normal actions passed to underlying environment(s) it
  allows to pass special actions by `step` method.

  Special actions:
    RETURN_DONE_ACTION: Returns done from `step` to force gym.utils.play to
      call reset.
    TOGGLE_WAIT_ACTION: Change between real-time-play and wait-for-pressed-key
      modes.
    WAIT_MODE_NOOP_ACTION: perform noop action (when wait-for-pressed-key mode
    is on)

  For keyboard keys related to actions above see `get_keys_to_action` method.

  Naming conventions:
    envs_step_tuples: Dictionary of tuples similar to these returned by
      gym.Env.step().
      {
        "env_name": (observation, reward, done, info),
        ...
      }
      Keys depend on subclass.
  """

  # Integers (as taken by step() method) related to special actions.
  RETURN_DONE_ACTION = 101
  TOGGLE_WAIT_ACTION = 102
  WAIT_MODE_NOOP_ACTION = 103

  HEADER_HEIGHT = 27

  def __init__(self, action_meanings):
    """Constructor for PlayerEnv.

    Args:
      action_meanings: list of strings indicating action names. Can be obtain by
        >>> env = gym.make("PongNoFrameskip-v4")  # insert your game name
        >>> env.unwrapped.get_action_meanings()
        See gym AtariEnv get_action_meanings() for more details.
    """
    self.action_meanings = action_meanings
    self._wait = True
    # If action_space will be needed, one could use e.g. gym.spaces.Dict.
    self.action_space = None
    self._last_step_tuples = None
    self.action_meanings = action_meanings
    self.name_to_action_num = {name: num for num, name in
                               enumerate(self.action_meanings)}

  def get_keys_to_action(self):
    """Get mapping from keyboard keys to actions.

    Required by gym.utils.play in environment or top level wrapper.

    Returns:
      {
        Unicode code point for keyboard key: action (formatted for step()),
        ...
      }
    """
    # Based on gym AtariEnv.get_keys_to_action()
    keyword_to_key = {
        "UP": ord("w"),
        "DOWN": ord("s"),
        "LEFT": ord("a"),
        "RIGHT": ord("d"),
        "FIRE": ord(" "),
    }

    keys_to_action = {}

    for action_id, action_meaning in enumerate(self.action_meanings):
      keys_tuple = tuple(sorted([
          key for keyword, key in keyword_to_key.items()
          if keyword in action_meaning]))
      assert keys_tuple not in keys_to_action
      keys_to_action[keys_tuple] = action_id

    # Special actions:
    keys_to_action[(ord("r"),)] = self.RETURN_DONE_ACTION
    keys_to_action[(ord("c"),)] = self.TOGGLE_WAIT_ACTION
    keys_to_action[(ord("n"),)] = self.WAIT_MODE_NOOP_ACTION

    return keys_to_action

  def _player_actions(self):
    return {
        self.RETURN_DONE_ACTION: self._player_return_done_action,
        self.TOGGLE_WAIT_ACTION: self._player_toggle_wait_action,
    }

  def _player_toggle_wait_action(self):
    self._wait = not self._wait
    return self._last_step_tuples

  def step(self, action):
    """Pass action to underlying environment(s) or perform special action."""
    # Special codes
    if action in self._player_actions():
      envs_step_tuples = self._player_actions()[action]()
    elif self._wait and action == self.name_to_action_num["NOOP"]:
      # Ignore no-op, do not pass to environment.
      envs_step_tuples = self._last_step_tuples
    else:
      # Run action on environment(s).
      if action == self.WAIT_MODE_NOOP_ACTION:
        action = self.name_to_action_num["NOOP"]
      # Perform action on underlying environment(s).
      envs_step_tuples = self._step_envs(action)
      self._update_statistics(envs_step_tuples)

    self._last_step_tuples = envs_step_tuples
    ob, reward, done, info = self._player_step_tuple(envs_step_tuples)
    return ob, reward, done, info

  def _augment_observation(self, ob, reward, cumulative_reward):
    """"Expand observation array with additional information header (top rows).

    Args:
      ob: observation
      reward: reward to be included in header.
      cumulative_reward: total cumulated reward to be included in header.

    Returns:
      Expanded observation array.
    """
    img = PIL_Image().new("RGB",
                          (ob.shape[1], self.HEADER_HEIGHT,))
    draw = PIL_ImageDraw().Draw(img)
    draw.text(
        (1, 0), "c:{:3}, r:{:3}".format(int(cumulative_reward), int(reward)),
        fill=(255, 0, 0)
    )
    draw.text(
        (1, 15), "fc:{:3}".format(int(self._frame_counter)),
        fill=(255, 0, 0)
    )
    header = np.asarray(img)
    del img
    header.setflags(write=1)
    # Top row color indicates if WAIT MODE is on.
    if self._wait:
      pixel_fill = (0, 255, 0)
    else:
      pixel_fill = (255, 0, 0)
    header[0, :, :] = pixel_fill
    return np.concatenate([header, ob], axis=0)

  def reset(self):
    raise NotImplementedError

  def _step_envs(self, action):
    """Perform action on underlying environment(s)."""
    raise NotImplementedError

  def _update_statistics(self, envs_step_tuples):
    """Update underlying environment(s) total cumulative rewards."""
    raise NotImplementedError

  def _player_return_done_action(self):
    """Function.

    Returns:
       envs_step_tuples: such that `player_step_tuple(envs_step_tuples)`
        will return done.
    """
    raise NotImplementedError

  def _player_step_tuple(self, envs_step_tuples):
    """Infer return tuple for step() given underlying environment tuple(s)."""
    raise NotImplementedError


class SimAndRealEnvPlayer(PlayerEnv):
  """Run simulated and real env side-by-side for comparison.

  Displays three windows - one for real environment, second for simulated
  and third for their differences.

  Normal actions are passed to both environments.

  Special Actions:
    RESTART_SIMULATED_ENV_ACTION: restart simulated environment only, using
      current frames from real environment.
    See `PlayerEnv` for rest of special actions.

  Naming conventions:
    envs_step_tuples: dictionary with two keys.
    {
      "real_env": (observation, reward, done, info),
      "sim_env": (observation, reward, done, info)
    }
  """

  RESTART_SIMULATED_ENV_ACTION = 110

  def __init__(self, real_env, sim_env, action_meanings):
    """Init.

    Args:
      real_env: real environment such as `FlatBatchEnv<T2TGymEnv>`.
      sim_env: simulation of `real_env` to be compared with. E.g.
        `SimulatedGymEnv` must allow to update initial frames for next reset
        with `add_to_initial_stack` method.
      action_meanings: list of strings indicating action names. Can be obtain by
        >>> env = gym.make("PongNoFrameskip-v4")  # insert your game name
        >>> env.unwrapped.get_action_meanings()
        See gym AtariEnv get_action_meanings() for more details.
    """
    super(SimAndRealEnvPlayer, self).__init__(action_meanings)
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

  def _player_actions(self):
    actions = super(SimAndRealEnvPlayer, self)._player_actions()
    actions.update({
        self.RESTART_SIMULATED_ENV_ACTION:
            self.player_restart_simulated_env_action,
    })
    return actions

  def get_keys_to_action(self):
    keys_to_action = super(SimAndRealEnvPlayer, self).get_keys_to_action()
    keys_to_action[(ord("x"),)] = self.RESTART_SIMULATED_ENV_ACTION
    return keys_to_action

  def _player_step_tuple(self, envs_step_tuples):
    """Construct observation, return usual step tuple.

    Args:
      envs_step_tuples: tuples.

    Returns:
      Step tuple: ob, reward, done, info
        ob: concatenated images [simulated observation, real observation,
          difference], with additional informations in header.
        reward: real environment reward
        done: True iff. envs_step_tuples['real_env'][2] is True
        info: real environment info
    """
    ob_real, reward_real, _, _ = envs_step_tuples["real_env"]
    ob_sim, reward_sim, _, _ = envs_step_tuples["sim_env"]
    ob_err = absolute_hinge_difference(ob_sim, ob_real)

    ob_real_aug = self._augment_observation(ob_real, reward_real,
                                            self.cumulative_real_reward)
    ob_sim_aug = self._augment_observation(ob_sim, reward_sim,
                                           self.cumulative_sim_reward)
    ob_err_aug = self._augment_observation(
        ob_err, reward_sim - reward_real,
        self.cumulative_sim_reward - self.cumulative_real_reward
    )
    ob = np.concatenate([ob_sim_aug, ob_real_aug, ob_err_aug], axis=1)
    _, reward, done, info = envs_step_tuples["real_env"]
    return ob, reward, done, info

  def reset(self):
    """Reset simulated and real environments."""
    self._frame_counter = 0
    ob_real = self.real_env.reset()
    # Initialize simulated environment with frames from real one.
    self.sim_env.add_to_initial_stack(ob_real)
    for _ in range(3):
      ob_real, _, _, _ = self.real_env.step(self.name_to_action_num["NOOP"])
      self.sim_env.add_to_initial_stack(ob_real)
    ob_sim = self.sim_env.reset()
    assert np.all(ob_real == ob_sim)
    self._last_step_tuples = self._pack_step_tuples((ob_real, 0, False, {}),
                                                    (ob_sim, 0, False, {}))
    self.set_zero_cumulative_rewards()
    ob, _, _, _ = self._player_step_tuple(self._last_step_tuples)
    return ob

  def _pack_step_tuples(self, real_env_step_tuple, sim_env_step_tuple):
    return dict(real_env=real_env_step_tuple,
                sim_env=sim_env_step_tuple)

  def set_zero_cumulative_rewards(self):
    self.cumulative_real_reward = 0
    self.cumulative_sim_reward = 0

  def _step_envs(self, action):
    """Perform step(action) on environments and update initial_frame_stack."""
    self._frame_counter += 1
    real_env_step_tuple = self.real_env.step(action)
    sim_env_step_tuple = self.sim_env.step(action)
    self.sim_env.add_to_initial_stack(real_env_step_tuple[0])
    return self._pack_step_tuples(real_env_step_tuple, sim_env_step_tuple)

  def _update_statistics(self, envs_step_tuples):
    self.cumulative_real_reward += envs_step_tuples["real_env"][1]
    self.cumulative_sim_reward += envs_step_tuples["sim_env"][1]

  def _player_return_done_action(self):
    ob = np.zeros(self.real_env.observation_space.shape, dtype=np.uint8)
    return self._pack_step_tuples((ob, 0, True, {}),
                                  (ob, 0, True, {}))

  def player_restart_simulated_env_action(self):
    self._frame_counter = 0
    ob = self.sim_env.reset()
    assert np.all(self._last_step_tuples["real_env"][0] == ob)
    self.set_zero_cumulative_rewards()
    return self._pack_step_tuples(
        self._last_step_tuples["real_env"], (ob, 0, False, {}))


class SingleEnvPlayer(PlayerEnv):
  """"Play on single (simulated or real) environment.

  See `PlayerEnv` for more details.

  Naming conventions:
    envs_step_tuples: dictionary with single key.
      {
        "env": (observation, reward, done, info),
      }
      Plural form used for consistency with `PlayerEnv`.
  """

  def __init__(self, env, action_meanings):
    super(SingleEnvPlayer, self).__init__(action_meanings)
    self.env = env
    # Set observation space
    orig = self.env.observation_space
    shape = tuple([orig.shape[0] + self.HEADER_HEIGHT] + list(orig.shape[1:]))
    self.observation_space = gym.spaces.Box(low=orig.low.min(),
                                            high=orig.high.max(),
                                            shape=shape, dtype=orig.dtype)

  def _player_step_tuple(self, envs_step_tuples):
    """Augment observation, return usual step tuple."""
    ob, reward, done, info = envs_step_tuples["env"]
    ob = self._augment_observation(ob, reward, self.cumulative_reward)
    return ob, reward, done, info

  def _pack_step_tuples(self, env_step_tuple):
    return dict(env=env_step_tuple)

  def reset(self):
    self._frame_counter = 0
    ob = self.env.reset()
    self._last_step_tuples = self._pack_step_tuples((ob, 0, False, {}))
    self.cumulative_reward = 0
    return self._augment_observation(ob, 0, self.cumulative_reward)

  def _step_envs(self, action):
    self._frame_counter += 1
    return self._pack_step_tuples(self.env.step(action))

  def _update_statistics(self, envs_step_tuples):
    _, reward, _, _ = envs_step_tuples["env"]
    self.cumulative_reward += reward

  def _player_return_done_action(self):
    ob = np.zeros(self.env.observation_space.shape, dtype=np.uint8)
    return self._pack_step_tuples((ob, 0, True, {}))


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
  if FLAGS.game_from_filenames:
    hparams.set_hparam(
        "game", player_utils.infer_game_name_from_filenames(directories["data"])
    )
  action_meanings = gym.make(full_game_name(hparams.game)).\
      unwrapped.get_action_meanings()
  epoch = FLAGS.epoch if FLAGS.epoch == "last" else int(FLAGS.epoch)

  def make_real_env():
    env = player_utils.setup_and_load_epoch(
        hparams, data_dir=directories["data"],
        which_epoch_data=None)
    env = FlatBatchEnv(env)  # pylint: disable=redefined-variable-type
    return env

  def make_simulated_env(setable_initial_frames, which_epoch_data):
    env = player_utils.load_data_and_make_simulated_env(
        directories["data"], directories["world_model"],
        hparams, which_epoch_data=which_epoch_data,
        setable_initial_frames=setable_initial_frames)
    return env

  if FLAGS.sim_and_real:
    sim_env = make_simulated_env(
        which_epoch_data=None, setable_initial_frames=True)
    real_env = make_real_env()
    env = SimAndRealEnvPlayer(real_env, sim_env, action_meanings)
  else:
    if FLAGS.simulated_env:
      env = make_simulated_env(  # pylint: disable=redefined-variable-type
          which_epoch_data=epoch, setable_initial_frames=False)
    else:
      env = make_real_env()
    env = SingleEnvPlayer(env, action_meanings)  # pylint: disable=redefined-variable-type

  env = player_utils.wrap_with_monitor(env, FLAGS.video_dir)

  if FLAGS.dry_run:
    env.unwrapped.get_keys_to_action()
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
