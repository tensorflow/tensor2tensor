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

python -m tensor2tensor.rl.trainer_model_based \
    --output_dir=$HOME/t2t/rl_v1 \
    --loop_hparams_set=rlmb_base \
    --loop_hparams='num_real_env_frames=10000,epochs=3'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time

from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.models.research import rl
from tensor2tensor.rl import trainer_model_based_params
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("loop_hparams_set", "rlmb_base",
                    "Which RL hparams set to use.")
flags.DEFINE_string("loop_hparams", "", "Overrides for overall loop HParams.")
flags.DEFINE_string("job_dir_to_evaluate", "",
                    "Directory of a job to be evaluated.")
flags.DEFINE_string("eval_results_dir", "/tmp",
                    "Directory to store result of evaluation")


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


def train_agent(environment_spec, agent_model_dir,
                event_dir, world_model_dir, epoch_data_dir, hparams, epoch=0,
                is_final_epoch=False):
  """Train the PPO agent in the simulated environment."""
  # TODO: Implement
  pass


def train_agent_real_env(
    env, agent_model_dir, event_dir, world_model_dir, epoch_data_dir,
    hparams, epoch=0, is_final_epoch=False):
  """Train the PPO agent in the real environment."""
  # TODO: Implement
  pass


def train_world_model(env, data_dir, output_dir, hparams, epoch):
  """Train the world model on problem_name."""
  # TODO: Implement
  train_steps = hparams.model_train_steps * (
      epoch + hparams.inital_epoch_train_steps_multiplier)
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  learning_rate = model_hparams.learning_rate_constant
  if epoch > 0: learning_rate *= hparams.learning_rate_bump


def setup_env(hparams):
  # TODO: Implement, might use scratch below (names are probably not correct),
  # but assure if no global flags are used in standard_atari_env_spec() (for not
  # simulated problem)
  environment_spec = rl.standard_atari_env_spec(
    hparams.env_name,
    resize_height_factor=hparams.resize_height_factor,
    resize_width_factor=hparams.resize_width_factor,
    grayscale=hparams.grayscale)
  env = T2TGymEnv([environment_spec.env_lambda() for _ in range(num_agents)])
  return env


def eval_unclipped_reward(env):
  # TODO: Implement, this should read data from env and aggregate (without
  # playing)
  pass


def training_loop(hparams, output_dir, report_fn=None, report_metric=None):
  """Run the main training loop."""
  # TODO: does anyone need this report_fn?
  if report_fn:
    assert report_metric is not None

  # Directories
  subdirectories = ["data", "tmp", "world_model", "ppo"]
  directories = setup_directories(output_dir, subdirectories)

  env = setup_env(hparams)

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []
  epoch_data_dirs = []

  data_dir = os.path.join(directories["data"], "initial")
  epoch_data_dirs.append(data_dir)
  # Collect data from the real environment with PPO or random policy.
  # TODO: do we need option not to gather_ppo_real_env_data?
  # We could set learning_rate=0 if this flag == False.
  assert hparams.gather_ppo_real_env_data
  ppo_model_dir = directories["ppo"]
  tf.logging.info("Initial training of PPO in real environment.")
  ppo_event_dir = os.path.join(directories["world_model"],
                               "ppo_summaries/initial")
  mean_reward = train_agent_real_env(
      env, ppo_model_dir,
      ppo_event_dir, data_dir,
      hparams, epoch=-1, is_final_epoch=False)
  tf.logging.info("Mean reward (initial): {}".format(mean_reward))

  eval_metrics_event_dir = os.path.join(directories["world_model"],
                                        "eval_metrics_event_dir")
  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_event_dir)

  mean_unclipped_reward_summary = tf.Summary()
  mean_unclipped_reward_summary.value.add(tag="mean_unclipped_reward",
                                          simple_value=None)
  mean_clipped_reward_summary = tf.Summary()
  mean_clipped_reward_summary.value.add(tag="mean_clipped_reward",
                                        simple_value=None)

  for epoch in range(hparams.epochs):
    is_final_epoch = (epoch + 1) == hparams.epochs
    log = make_log_fn(epoch, log_relative_time)

    epoch_data_dir = os.path.join(directories["data"], str(epoch))
    tf.gfile.MakeDirs(epoch_data_dir)
    env.generate_data(epoch_data_dir, directories['tmp'])
    epoch_data_dirs.append(epoch_data_dir)

    # Train world model
    log("Training world model")
    train_world_model(env, epoch_data_dir,
                      directories["world_model"], hparams, epoch)

    # Train PPO
    log("Training PPO in simulated environment.")
    ppo_event_dir = os.path.join(directories["world_model"],
                                 "ppo_summaries", str(epoch))
    ppo_model_dir = directories["ppo"]
    if not hparams.ppo_continue_training:
      ppo_model_dir = ppo_event_dir
    # TODO: build environment_spec (for simulated env)
    train_agent(environment_spec, ppo_model_dir,
                ppo_event_dir, directories["world_model"], epoch_data_dir,
                hparams, epoch=epoch, is_final_epoch=is_final_epoch)

    # Train PPO on real env (short)
    log("Training PPO in real environment.")
    # TODO: pass env, return summaries?
    # TODO(kc): generation_mean_reward vs mean_reward (clipped?)
    mean_clipped_reward = train_agent_real_env(
        env, ppo_model_dir,
        ppo_event_dir, directories["world_model"], epoch_data_dir,
        hparams, epoch=epoch, is_final_epoch=is_final_epoch)

    if hparams.stop_loop_early:
      return 0.0

    log("Mean clipped reward during generation: {}".format(
        mean_clipped_reward))  # this was output of generate_real_env_data(...)

    mean_unclipped_reward = eval_unclipped_reward(env)
    log("Mean eval reward (unclipped): {}".format(mean_unclipped_reward))

    # Summarize metrics.
    mean_unclipped_reward_summary.value[0].simple_value = mean_unclipped_reward
    eval_metrics_writer.add_summary(mean_unclipped_reward_summary, epoch)
    mean_clipped_reward_summary.value[0].simple_value = int(mean_clipped_reward)
    eval_metrics_writer.add_summary(mean_clipped_reward_summary, epoch)
    eval_metrics_writer.flush()

    # Report metrics
    eval_metrics = {"mean_unclipped_reward": mean_unclipped_reward}
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
