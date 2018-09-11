#--output_dir=/Users/piotr.milos/Downloads/18 --alsologtostderr --loop_hparams_set=rl_modelrl_base_quick
#--output_dir=/Users/piotr.milos/t2t/rl_v1 --alsologtostderr --loop_hparams_set=rl_modelrl_tiny

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

from gym.spaces import Discrete
from tensor2tensor.data_generators import gym_problems_specs
from tensor2tensor.rl.envs.batch_env_factory import batch_env_factory
from tensor2tensor.rl.envs.utils import get_policy
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.rl.trainer_model_based import FLAGS, setup_directories, temporary_flags
from gym.utils import play
import tensorflow as tf
from gym.core import Env


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


def write_on_image(img, text="", position=(0, 0), color=(255, 255, 255)):
  img = _assert_image(img)
  if text=="":
    return img
  draw = ImageDraw.Draw(img)
  font = _get_font()
  draw.text(position, text, color, font=font)

  return img

def concatenate_images(*imgs, axis=1):
  imgs = [_assert_image(img) for img in imgs]
  imgs_np = [np.array(img) for img in imgs]
  concatenated_im_np = np.concatenate(imgs_np, axis=axis)

  return _assert_image(concatenated_im_np)


class DebugBatchEnv(Env):

  INFO_PANE_WIDTH = 250

  def __init__(self, hparams, sess=None):
    self.action_space = Discrete(6)
    self.observation_space = Box(low=0, high=255, shape=(210, 160+DebugBatchEnv.INFO_PANE_WIDTH, 3), dtype=np.uint8)
    self._tmp = 1
    self.res = None
    self.sess = sess if sess is not None else tf.Session()
    self._prepare_networks(hparams, self.sess)

  def _prepare_networks(self, hparams, sess):
    self.action = tf.placeholder(shape=(1,), dtype=tf.int32)
    batch_env = batch_env_factory(hparams)
    self.reward, self.done = batch_env.simulate(self.action)
    self.observation = batch_env.observ
    self.reset_op = batch_env.reset(tf.constant([0], dtype=tf.int32))

    environment_wrappers = hparams.environment_spec.wrappers
    wrappers = copy.copy(environment_wrappers) if environment_wrappers else []

    to_initialize = [batch_env]
    for w in wrappers:
      batch_env = w[0](batch_env, **w[1])
      to_initialize.append(batch_env)

    def initialization_lambda():
      for batch_env in to_initialize:
        batch_env.initialize(sess)

    self.initialize = initialization_lambda

    obs_copy = batch_env.observ + 0

    actor_critic = get_policy(tf.expand_dims(obs_copy, 0), hparams)
    self.policy_probs = actor_critic.policy.probs[0, 0, :]
    self.value = actor_critic.value[0, :]


  def render(self, mode='human'):
    raise NotImplemented()

  def _fake_reset(self):
    self._tmp = 0
    _observ = np.ones(shape=(210, 160, 3), dtype=np.uint8) * 10 * self._tmp
    _observ[0, 0, 0] = 0
    _observ[0, 0, 1] = 255
    self.res = (_observ, 0, False, [0.1, 0.5, 0.5], 1.1)

  def _reset_env(self):
    _observ = self.sess.run(self.reset_op)[0, ...]
    _observ[0, 0, 0] = 0
    _observ[0, 0, 1] = 255
    #TODO:(put correct numbers)
    self.res = (_observ, 0, False, [0.1, 0.5, 0.5], 1.1)

  def reset(self):
    self._reset_env()
    observ = self._augment_observation()
    return observ

  def _step_fake(self, action):
    observ = np.ones(shape=(210, 160, 3), dtype=np.uint8)*10*self._tmp
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

  def _step_env(self, action):
    observ, rew, done, probs, vf = self.sess.\
      run([self.observation, self.reward, self.done, self.policy_probs, self.value],
          feed_dict={self.action: [action]})

    return observ[0, ...], rew[0, ...], done[0, ...], probs, vf

  def _augment_observation(self):
    _observ, rew, done, probs, vf = self.res
    info_pane = np.zeros(shape=(210, DebugBatchEnv.INFO_PANE_WIDTH, 3), dtype=np.uint8)
    probs_str = ""
    for p in probs:
      probs_str += "%.2f" % p +", "

    probs_str = probs_str[:-2]

    action = np.argmax(probs)
    info_str = " Policy:{}\n Action:{}\n Value function:{}\n Reward:{}".format(probs_str, action,
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
      self.reset()
      _, rew, done, _, _ = self.res
      observ = self._augment_observation()
      return observ, rew, done, {}

    if action == 102:
      #play
      raise NotImplemented()

    #standard codes
    _observ, rew, done, probs, vf = self._step_env(action)
    self.res = (_observ, rew, done, probs, vf)

    observ = self._augment_observation()
    return observ, rew, done, {"probs": probs, "vf": vf}


def main(_):
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  output_dir = FLAGS.output_dir

  subdirectories = ["data", "tmp", "world_model", "ppo"]
  using_autoencoder = hparams.autoencoder_train_steps > 0
  if using_autoencoder:
    subdirectories.append("autoencoder")
  directories = setup_directories(output_dir, subdirectories)

  if hparams.game in gym_problems_specs.ATARI_GAMES:
    game_with_mode = hparams.game + "_deterministic-v4"
  else:
    game_with_mode = hparams.game

  if using_autoencoder:
    simulated_problem_name = (
        "gym_simulated_discrete_problem_with_agent_on_%s_autoencoded"
        % game_with_mode)
  else:
    simulated_problem_name = ("gym_simulated_discrete_problem_with_agent_on_%s"
                              % game_with_mode)
  epoch = hparams.epochs-1
  epoch_data_dir = os.path.join(directories["data"], str(epoch))
  ppo_model_dir = directories["ppo"]

  world_model_dir = directories["world_model"]

  gym_problem = registry.problem(simulated_problem_name)

  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  environment_spec = copy.copy(gym_problem.environment_spec)
  environment_spec.simulation_random_starts = hparams.simulation_random_starts

  batch_env_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  batch_env_hparams.add_hparam("model_hparams", model_hparams)
  batch_env_hparams.add_hparam("environment_spec", environment_spec)
  batch_env_hparams.num_agents = 1

  with temporary_flags({
      "problem": simulated_problem_name,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "output_dir": world_model_dir,
      "data_dir": epoch_data_dir,
  }):
    sess = tf.Session()
    env = DebugBatchEnv(batch_env_hparams, sess)
    sess.run(tf.global_variables_initializer())
    env.initialize()

    env_model_loader = tf.train.Saver(
      tf.global_variables("next_frame*"))
    trainer_lib.restore_checkpoint(world_model_dir, env_model_loader, sess,
      must_restore=True)

    model_saver = tf.train.Saver(
      tf.global_variables(".*network_parameters.*"))
    trainer_lib.restore_checkpoint(ppo_model_dir, model_saver, sess)

    key_mapping = gym_problem.env.env.get_keys_to_action()
    #map special codes
    key_mapping[()] = 100
    key_mapping[(ord('r'),)] = 101
    key_mapping[(ord('p'),)] = 102

    play.play(env, zoom=2, fps=10, keys_to_action=key_mapping)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
