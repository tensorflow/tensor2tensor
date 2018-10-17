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

r"""Training of model-based RL agents.

Example invocation:

python -m tensor2tensor.rl.trainer_model_based_new \
    --output_dir=$HOME/t2t/rl_v1 \
    --loop_hparams_set=rlmb_base \
    --loop_hparams='num_real_env_frames=10000,epochs=3'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import datetime
import math
import os
import time

import gym
import numpy as np

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.models.research import rl
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl import trainer_model_based_params
from tensor2tensor.rl.envs.utils import InitialFrameChooser
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


@contextlib.contextmanager
def temporary_flags(flag_settings):
  old_values = {}
  for flag_name, flag_value in flag_settings.items():
    old_values[flag_name] = getattr(FLAGS, flag_name)
    setattr(FLAGS, flag_name, flag_value)
  yield
  for flag_name, flag_value in old_values.items():
    setattr(FLAGS, flag_name, flag_value)


def _ppo_training_epochs(hparams, epoch, is_final_epoch, real_env_training):
  """Helper for PPO restarts."""
  if hparams.gather_ppo_real_env_data:
    assert hparams.real_ppo_epochs_num is 0, (
        "Should be put to 0 to enforce better readability")
    real_training_ppo_epochs_num = int(math.ceil(
        hparams.num_real_env_frames /
        (hparams.epochs*hparams.real_ppo_epoch_length)))
  else:
    real_training_ppo_epochs_num = hparams.real_ppo_epochs_num

  simulated_training_ppo_epochs_num = hparams.ppo_epochs_num

  if epoch == -1:
    assert real_env_training, (
        "Epoch -1 should only be used for PPO collection in real environment.")
    return real_training_ppo_epochs_num
  ppo_training_epochs = (epoch + 1) * (simulated_training_ppo_epochs_num
                                       + real_training_ppo_epochs_num)
  if is_final_epoch:  # Length of training in the final epoch is doubled.
    ppo_training_epochs += simulated_training_ppo_epochs_num
  if real_env_training:
    ppo_training_epochs += real_training_ppo_epochs_num
  return ppo_training_epochs


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


def make_log_fn(epoch, log_relative_time_fn):

  def log(msg, *args):
    msg %= args
    tf.logging.info("%s Epoch %d: %s", ">>>>>>>", epoch, msg)
    log_relative_time_fn()

  return log


def train_supervised(problem, model_name, hparams, data_dir, output_dir,
                     train_steps, eval_steps, local_eval_frequency=None,
                     schedule="continuous_train_and_eval"):
  """Train supervised."""
  if local_eval_frequency is None:
    local_eval_frequency = getattr(FLAGS, "local_eval_frequency")

  exp_fn = trainer_lib.create_experiment_fn(
      model_name, problem, data_dir, train_steps, eval_steps,
      min_eval_frequency=local_eval_frequency
  )
  run_config = trainer_lib.create_run_config(model_name, model_dir=output_dir)
  exp = exp_fn(run_config, hparams)
  getattr(exp, schedule)()


def train_agent(environment_spec, agent_model_dir,
                event_dir, world_model_dir, data_dir, hparams, epoch=0,
                is_final_epoch=False):
  """Train the PPO agent in the simulated environment."""
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_params_names = ["epochs_num", "epoch_length",
                      "learning_rate", "num_agents",
                      "optimization_epochs", "eval_every_epochs"]

  for param_name in ppo_params_names:
    ppo_param_name = "ppo_" + param_name
    if ppo_param_name in hparams:
      ppo_hparams.set_hparam(param_name, hparams.get(ppo_param_name))

  ppo_hparams.epochs_num = _ppo_training_epochs(hparams, epoch,
                                                is_final_epoch, False)
  ppo_hparams.save_models_every_epochs = 10
  ppo_hparams.world_model_dir = world_model_dir
  ppo_hparams.add_hparam("force_beginning_resets", True)

  # Adding model hparams for model specific adjustments
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  ppo_hparams.add_hparam("model_hparams", model_hparams)

  environment_spec = copy.copy(environment_spec)
  environment_spec_param_names = [
      "simulation_random_starts", "simulation_flip_first_random_for_beginning",
      "intrinsic_reward_scale"
  ]
  for param_name in environment_spec_param_names:
    environment_spec.set_hparam(param_name, hparams.get(param_name))
  ppo_hparams.add_hparam("environment_spec", environment_spec)

  ppo_hparams.add_hparam("initial_frame_chooser", InitialFrameChooser(
      environment_spec, mode=tf.estimator.ModeKeys.EVAL
  ))

  # TODO(koz4k): Pass by arguments.
  with temporary_flags({
      "problem": environment_spec.initial_frames_problem,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "output_dir": world_model_dir,
      "data_dir": data_dir,
  }):
    rl_trainer_lib.train(ppo_hparams, event_dir + "sim", agent_model_dir,
                         name_scope="ppo_sim%d" % (epoch + 1))


def train_agent_real_env(
    env, agent_model_dir, event_dir, data_dir,
    hparams, epoch=0, is_final_epoch=False):
  """Train the PPO agent in the real environment."""
  del data_dir
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_params_names = ["epochs_num", "epoch_length",
                      "learning_rate", "num_agents", "eval_every_epochs",
                      "optimization_epochs", "effective_num_agents"]

  # This should be overridden.
  ppo_hparams.add_hparam("effective_num_agents", None)
  for param_name in ppo_params_names:
    ppo_param_name = "real_ppo_"+ param_name
    if ppo_param_name in hparams:
      ppo_hparams.set_hparam(param_name, hparams.get(ppo_param_name))

  ppo_hparams.epochs_num = _ppo_training_epochs(hparams, epoch,
                                                is_final_epoch, True)
  # We do not save model, as that resets frames that we need at restarts.
  # But we need to save at the last step, so we set it very high.
  ppo_hparams.save_models_every_epochs = 1000000

  environment_spec = rl.standard_atari_env_spec(
      batch_env=env, include_clipping=False
  )

  ppo_hparams.add_hparam("environment_spec", environment_spec)

  rl_trainer_lib.train(ppo_hparams, event_dir + "real", agent_model_dir,
                       name_scope="ppo_real%d" % (epoch + 1))

  # Save unfinished rollouts to history.
  env.reset()


def train_world_model(env, data_dir, output_dir, hparams, epoch):
  """Train the world model on problem_name."""
  train_steps = hparams.model_train_steps * (
      epoch + hparams.inital_epoch_train_steps_multiplier)
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  model_hparams.learning_rate = model_hparams.learning_rate_constant
  if epoch > 0:
    model_hparams.learning_rate *= hparams.learning_rate_bump

  train_supervised(
      problem=env,
      model_name=hparams.generative_model,
      hparams=model_hparams,
      data_dir=data_dir,
      output_dir=output_dir,
      train_steps=train_steps,
      eval_steps=100,
      local_eval_frequency=2000
  )


def make_gym_env(hparams):
  """Make env."""
  game_mode = "Deterministic-v4"
  camel_game_name = "".join(
      [w[0].upper() + w[1:] for w in hparams.game.split("_")])
  camel_game_name += game_mode
  env_name = camel_game_name
  env = gym.make(env_name)
  if hparams.env_timesteps_limit != -1:
    # Replace TimeLimit Wrapper with one of proper time step limit.
    if isinstance(env, gym.wrappers.TimeLimit):
      env = env.env
    env = gym.wrappers.TimeLimit(env,
                                 max_episode_steps=hparams.env_timesteps_limit)
  return env


def setup_env(hparams, data_dir):
  """Setup."""
  env = T2TGymEnv([make_gym_env(hparams)
                   for _ in range(hparams.real_ppo_num_agents)],
                  data_dir,
                  grayscale=hparams.grayscale,
                  resize_width_factor=hparams.resize_width_factor,
                  resize_height_factor=hparams.resize_height_factor)
  return env


def eval_reward(env, clipped):
  """Calculates mean rewards from given epoch."""
  reward_name = "reward" if clipped else "unclipped_reward"
  rewards = []
  for rollout in env.current_epoch_rollouts():
    if rollout[-1].done:
      rollout_reward = sum(getattr(frame, reward_name) for frame in rollout)
      rewards.append(rollout_reward)
  if rewards:
    mean_rewards = np.mean(rewards)
  else:
    mean_rewards = 0
  return mean_rewards


def training_loop(hparams, output_dir, report_fn=None, report_metric=None):
  """Run the main training loop."""
  if report_fn:
    assert report_metric is not None

  # Directories
  subdirectories = ["data", "tmp", "world_model", "ppo"]
  directories = setup_directories(output_dir, subdirectories)

  epoch = -1
  data_dir = directories["data"]
  env = setup_env(hparams, data_dir)
  env.start_new_epoch(epoch)

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []

  # Collect data from the real environment with PPO or random policy.
  # TODO(lukaszkaiser): do we need option not to gather_ppo_real_env_data?
  # We could set learning_rate=0 if this flag == False.
  assert hparams.gather_ppo_real_env_data
  ppo_model_dir = directories["ppo"]
  tf.logging.info("Initial training of PPO in real environment.")
  ppo_event_dir = os.path.join(directories["world_model"],
                               "ppo_summaries/initial")
  train_agent_real_env(
      env, ppo_model_dir,
      ppo_event_dir, data_dir,
      hparams, epoch=epoch, is_final_epoch=False)
  mean_unclipped_reward = eval_reward(env, clipped=False)
  tf.logging.info("Mean reward (initial): {}".format(mean_unclipped_reward))

  eval_metrics_event_dir = os.path.join(directories["world_model"],
                                        "eval_metrics_event_dir")
  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_event_dir)

  mean_unclipped_reward_summary = tf.Summary()
  mean_unclipped_reward_summary.value.add(tag="mean_unclipped_reward",
                                          simple_value=None)
  mean_clipped_reward_summary = tf.Summary()
  mean_clipped_reward_summary.value.add(tag="mean_clipped_reward",
                                        simple_value=None)

  sim_env_spec = rl.standard_atari_env_simulated_spec(
      env,
      # Hardcoded for now. TODO(koz4k): Make it a hparam.
      video_num_input_frames=4, video_num_target_frames=1
  )

  for epoch in range(hparams.epochs):
    env.generate_data()

    is_final_epoch = (epoch + 1) == hparams.epochs
    log = make_log_fn(epoch, log_relative_time)

    # Train world model
    log("Training world model")
    train_world_model(env, data_dir,
                      directories["world_model"], hparams, epoch)

    # Train PPO
    log("Training PPO in simulated environment.")
    ppo_event_dir = os.path.join(directories["world_model"],
                                 "ppo_summaries", str(epoch))
    ppo_model_dir = directories["ppo"]
    if not hparams.ppo_continue_training:
      ppo_model_dir = ppo_event_dir

    train_agent(sim_env_spec, ppo_model_dir,
                ppo_event_dir, directories["world_model"], data_dir,
                hparams, epoch=epoch, is_final_epoch=is_final_epoch)

    env.start_new_epoch(epoch)

    # Train PPO on real env (short)
    log("Training PPO in real environment.")
    train_agent_real_env(
        env, ppo_model_dir,
        ppo_event_dir, data_dir,
        hparams, epoch=epoch, is_final_epoch=is_final_epoch)

    if hparams.stop_loop_early:
      return 0.0
    mean_clipped_reward = eval_reward(env, clipped=True)
    log("Mean clipped reward during generation: {}".format(
        mean_clipped_reward))  # this was output of generate_real_env_data(...)

    mean_unclipped_reward = eval_reward(env, clipped=False)
    log("Mean eval reward (unclipped): {}".format(mean_unclipped_reward))

    # Summarize metrics.
    mean_unclipped_reward_summary.value[0].simple_value = mean_unclipped_reward
    eval_metrics_writer.add_summary(mean_unclipped_reward_summary, epoch)
    mean_clipped_reward_summary.value[0].simple_value = int(mean_clipped_reward)
    eval_metrics_writer.add_summary(mean_clipped_reward_summary, epoch)
    eval_metrics_writer.flush()

    # Report metrics
    eval_metrics = {"mean_reward": mean_unclipped_reward}
    epoch_metrics.append(eval_metrics)
    log("Eval metrics: %s", str(eval_metrics))
    if report_fn:
      report_fn(eval_metrics[report_metric], epoch)

  # Return the evaluation metrics from the final epoch
  return epoch_metrics[-1]


def main(_):
  hp = trainer_model_based_params.create_loop_hparams()
  assert not FLAGS.job_dir_to_evaluate
  training_loop(hp, FLAGS.output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
