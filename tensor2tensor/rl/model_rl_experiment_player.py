
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import datetime
import math
import os
import time

import six

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import gym_problems_specs
from tensor2tensor.layers import discretization
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl.envs.batch_env_factory import batch_env_factory
from tensor2tensor.rl.envs.utils import get_policy
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.rl.trainer_model_based import FLAGS

import tensorflow as tf


HP_SCOPES = ["loop", "model", "ppo"]


def setup_directories(base_dir, subdirs):
  base_dir = os.path.expanduser(base_dir)
  tf.gfile.MakeDirs(base_dir)

  all_dirs = {}
  for subdir in subdirs:
    dir_name = os.path.join(base_dir, subdir)
    tf.gfile.MakeDirs(dir_name)
    all_dirs[subdir] = dir_name
  return all_dirs


def make_relative_timing_fn():
  """Make a function that logs the duration since it was made."""
  start_time = time.time()

  def format_relative_time():
    time_delta = time.time() - start_time
    return str(datetime.timedelta(seconds=time_delta))

  def log_relative_time():
    tf.logging.info("Timing: %s", format_relative_time())

  return log_relative_time


@contextlib.contextmanager
def temporary_flags(flag_settings):
  old_values = {}
  for flag_name, flag_value in flag_settings.items():
    old_values[flag_name] = getattr(FLAGS, flag_name)
    setattr(FLAGS, flag_name, flag_value)
  yield
  for flag_name, flag_value in old_values.items():
    setattr(FLAGS, flag_name, flag_value)


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

  # ppo_hparams.epochs_num = _ppo_training_epochs(hparams, epoch,
  #                                                 is_final_epoch, False)
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
    env.initialize()

    r = env.step(0)
    r = env.reset()
    print("R:{}".format(r))

from gym.core import Env


class DebugBatchEnv(Env):

  def __init__(self, hparams, sess = None):
    if sess == None:
      self.sess = tf.Session()
    else:
      self.sess = sess

    batch_env = batch_env_factory(hparams)

    self.action = tf.placeholder(shape=(1,), dtype=tf.int32)

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
    x = 1

  def render(self, mode='human'):
    raise NotImplemented()

  def reset(self):
    observ = self.sess.run(self.reset_op)
    return observ


  def _step_fake(self, action):
    import numpy as np
    observ = np.zeros(shape=(210, 160, 3), dtype=np.uint8)
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

  def step(self, action):
    observ, rew, done, probs, vf = self.sess.\
      run([self.observation, self.reward, self.done, self.policy_probs, self.value],
          feed_dict={self.action: [action]})

    return observ[0, ...], rew[0, ...], done[0, ...], probs, vf



def training_loop(hparams, output_dir, report_fn=None, report_metric=None):
  """Run the main training loop."""
  if report_fn:
    assert report_metric is not None

  # Global state

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
    problem_name = (
        "gym_discrete_problem_with_agent_on_%s_with_autoencoder"
        % game_with_mode)
    world_model_problem = (
        "gym_discrete_problem_with_agent_on_%s_autoencoded" % game_with_mode)
    simulated_problem_name = (
        "gym_simulated_discrete_problem_with_agent_on_%s_autoencoded"
        % game_with_mode)
  else:
    problem_name = ("gym_discrete_problem_with_agent_on_%s" % game_with_mode)
    world_model_problem = problem_name
    simulated_problem_name = ("gym_simulated_discrete_problem_with_agent_on_%s"
                              % game_with_mode)
    if problem_name not in registry.list_problems():
      tf.logging.info("Game Problem %s not found; dynamically registering",
                      problem_name)
      gym_problems_specs.create_problems_for_game(hparams.game,
                                                  game_mode="Deterministic-v4")

  # Autoencoder model dir
  autoencoder_model_dir = directories.get("autoencoder")

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []
  epoch_data_dirs = []


  # Collect data from the real environment with random policy
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
