
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import copy
import os
from gym.spaces import Box
import numpy as np

import six
from gym.spaces import Discrete
from gym.utils.play import PlayPlot

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import gym_problems_specs
from tensor2tensor.layers import discretization
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl.envs.batch_env_factory import batch_env_factory
from tensor2tensor.rl.envs.utils import get_policy
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.rl.trainer_model_based import FLAGS, setup_directories, temporary_flags
from gym.utils import play
import tensorflow as tf


HP_SCOPES = ["loop", "model", "ppo"]

_font = None
FONT_SIZE = 20

def _get_font():
  global _font
  if _font is None:
    #weirdness due to various working dirs
    FONT_PATHS = ["tensor-2-tensor-with-mrunner/tensor-2-tensor-with-mrunner/deepsense_experiments/Xerox Serif Narrow.ttf",
                  "tensor-2-tensor-with-mrunner/deepsense_experiments/Xerox Serif Narrow.ttf",
                  "deepsense_experiments/Xerox Serif Narrow.ttf"]

    for path in FONT_PATHS:
      try:
        _font = ImageFont.truetype(path, FONT_SIZE)
        return _font
      except:
        pass


def _assert_image(img):
  if isinstance(img, np.ndarray):
    img = Image.fromarray(np.ndarray.astype(img, np.uint8))
  return img


def write_on_image(img, text="", positon=(0,0), color=(255,255,255)):
  img = _assert_image(img)
  if text=="":
    return img
  draw = ImageDraw.Draw(img)
  font = _get_font()
  draw.text(positon, text, color, font=font)

  return img

def concatenate_images(*imgs, axis=1):
  imgs = [_assert_image(img) for img in imgs]
  imgs_np = [np.array(img) for img in imgs]
  concatenated_im_np = np.concatenate(imgs_np, axis=axis)

  return _assert_image(concatenated_im_np)


def train_agent(problem_name, agent_model_dir,
                event_dir, world_model_dir, epoch_data_dir, hparams, epoch=0,
                is_final_epoch=False):
  """Train the PPO agent in the simulated environment."""
  gym_problem = registry.problem(problem_name)
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_params_names = ["epochs_num", "epoch_length",
                      "learning_rate", "num_agents",
                      "optimization_epochs"]

  for param_name in ppo_params_names:
    ppo_param_name = "ppo_" + param_name
    if ppo_param_name in hparams:
      ppo_hparams.set_hparam(param_name, hparams.get(ppo_param_name))

  ppo_hparams.save_models_every_epochs = 10
  ppo_hparams.world_model_dir = world_model_dir
  ppo_hparams.add_hparam("force_beginning_resets", True)

  # Adding model hparams for model specific adjustments
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  ppo_hparams.add_hparam("model_hparams", model_hparams)

  environment_spec = copy.copy(gym_problem.environment_spec)
  environment_spec.simulation_random_starts = hparams.simulation_random_starts
  environment_spec.simulation_flip_first_random_for_beginning = False
  environment_spec.intrinsic_reward_scale = hparams.intrinsic_reward_scale

  ppo_hparams.add_hparam("environment_spec", environment_spec)
  ppo_hparams.num_agents = 1

  with temporary_flags({
      "problem": problem_name,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "output_dir": world_model_dir,
      "data_dir": epoch_data_dir,
  }):


    sess = tf.Session()
    env = DebugBatchEnv(ppo_hparams, sess)
    sess.run(tf.global_variables_initializer())
    # env.initialize()

    key_mapping = {(): 100, (ord('q'),):1, (ord('a'),):2,
                   (ord('r'),):101,
                   (ord('p'),):102}

    play.play(env, zoom=1, fps=10, keys_to_action=key_mapping)

from gym.core import Env


class DebugBatchEnv(Env):

  def __init__(self, hparams, sess = None):
    if sess == None:
      self.sess = tf.Session()
    else:
      self.sess = sess

    self.action_space = Discrete(6)
    self.observation_space = Box(low=0, high=255, shape=(210, 320, 3), dtype=np.uint8)

    # batch_env = batch_env_factory(hparams)

    self.action = tf.placeholder(shape=(1,), dtype=tf.int32)

    # self.reward, self.done = batch_env.simulate(self.action)
    # self.observation = batch_env.observ
    # self.reset_op = batch_env.reset(tf.constant([0], dtype=tf.int32))

    environment_wrappers = hparams.environment_spec.wrappers
    wrappers = copy.copy(environment_wrappers) if environment_wrappers else []

    # to_initialize = [batch_env]
    # for w in wrappers:
    #   batch_env = w[0](batch_env, **w[1])
    #   to_initialize.append(batch_env)
    #
    # def initialization_lambda():
    #   for batch_env in to_initialize:
    #     batch_env.initialize(sess)

    # self.initialize = initialization_lambda

    # obs_copy = batch_env.observ + 0

    # actor_critic = get_policy(tf.expand_dims(obs_copy, 0), hparams)
    # self.policy_probs = actor_critic.policy.probs[0, 0, :]
    # self.value = actor_critic.value[0, :]
    self._tmp = 1
    self.res = None

  def render(self, mode='human'):
    raise NotImplemented()

  def reset(self):
    # observ = self.sess.run(self.reset_op)
    self._tmp = 0
    _observ = np.ones(shape=(210, 160, 3), dtype=np.uint8) * 10 * self._tmp
    _observ[0, 0, 0] = 0
    _observ[0, 0, 1] = 255
    self.res = (_observ, 0, False, [0.1, 0.5, 0.5], 1.1)
    observ = self._augment_observation()
    return observ


  def _step_fake(self, action):

    observ = np.ones(shape=(210, 160+250, 3), dtype=np.uint8)*10*self._tmp
    observ[0, 0, 0] = 0
    observ[0, 0, 1] = 255

    self._tmp += 1
    if self._tmp>20:
      self._tmp = 0

    rew = 1
    done = False
    probs = np.ones(shape=(6,), dtype=np.float32)/6
    vf = 0.0

    return observ, rew, done, probs, vf

  def _env_step_fake(self, action):
    observ, rew, done, probs, vf = self.sess.\
      run([self.observation, self.reward, self.done, self.policy_probs, self.value],
          feed_dict={self.action: [action]})

    return observ[0, ...], rew[0, ...], done[0, ...], probs, vf

  def _augment_observation(self):
    _observ, rew, done, probs, vf = self.res
    info_pane = np.zeros(shape=(210, 250, 3), dtype=np.uint8)
    probs_str = ""
    for p in probs:
      probs_str += "%.2f" % p +", "

    action = np.argmax(probs)

    info_str = "Policy:{}\nAction:{}\nValue function:{}\nReward:{}".format(probs_str, action,
                                                                           vf, rew)
    info_pane = write_on_image(info_pane, info_str)

    augmented_observ = concatenate_images(_observ, info_pane)
    augmented_observ = np.array(augmented_observ)
    return augmented_observ


  def step(self, action):
    #Special codes
    if action==100:
      #Skip action
      _, rew, done, _, _ = self.res
      observ = self._augment_observation()
      return observ, rew, done, {}

    if action == 101:
      #reset
      observ, rew, _ = self.res
      return observ, rew, True, {}

    if action == 102:
      #play
      raise NotImplemented()

    #standard codes
    _observ, rew, done, probs, vf = self._step_fake(action)
    self.res = (_observ, rew, done, probs, vf)

    observ = self._augment_observation()
    return observ, rew, done, {"probs": probs, "vf": vf}



def training_loop(hparams, output_dir, report_fn=None, report_metric=None):
  """Run the main training loop."""

  # Directories
  subdirectories = ["data", "tmp", "world_model", "ppo"]
  using_autoencoder = hparams.autoencoder_train_steps > 0
  if using_autoencoder:
    subdirectories.append("autoencoder")
  directories = setup_directories(output_dir, subdirectories)

  if hparams.game in gym_problems_specs.ATARI_GAMES:
    game_with_mode = hparams.game + "_deterministic-v4"
  else:
    game_with_mode = hparams.game
  # Problems
  if using_autoencoder:
    simulated_problem_name = (
        "gym_simulated_discrete_problem_with_agent_on_%s_autoencoded"
        % game_with_mode)
  else:
    simulated_problem_name = ("gym_simulated_discrete_problem_with_agent_on_%s"
                              % game_with_mode)

  epoch_data_dirs = []
  data_dir = os.path.join(directories["data"], "random")
  epoch_data_dirs.append(data_dir)

  for epoch in range(hparams.epochs):
    is_final_epoch = (epoch + 1) == hparams.epochs

    # Combine all previously collected environment data
    epoch_data_dir = os.path.join(directories["data"], str(epoch))


    ppo_event_dir = os.path.join(directories["world_model"],
                                 "ppo_summaries", str(epoch))
    ppo_model_dir = directories["ppo"]
    if not hparams.ppo_continue_training:
      ppo_model_dir = ppo_event_dir
    train_agent(simulated_problem_name, ppo_model_dir,
                ppo_event_dir, directories["world_model"], epoch_data_dir,
                hparams, epoch=epoch, is_final_epoch=is_final_epoch)


  raise NotImplementedError()
  return 1



def create_loop_hparams():
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  return hparams


def main(_):
  hp = create_loop_hparams()
  output_dir = FLAGS.output_dir
  training_loop(hp, output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
