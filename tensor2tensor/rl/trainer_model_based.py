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

import copy
import datetime
import math
import os
import pprint
import random
import time

import numpy as np
import six

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.models.research import rl
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl import trainer_model_based_params
from tensor2tensor.rl.envs.simulated_batch_gym_env import SimulatedBatchGymEnv
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


def real_ppo_epoch_increment(hparams):
  """PPO increment."""
  if hparams.gather_ppo_real_env_data:
    assert hparams.real_ppo_epochs_num is 0, (
        "Should be put to 0 to enforce better readability"
    )
    return int(math.ceil(
        hparams.num_real_env_frames /
        (hparams.epochs * hparams.real_ppo_epoch_length)
    ))
  else:
    return hparams.real_ppo_epochs_num


def sim_ppo_epoch_increment(hparams, is_final_epoch):
  increment = hparams.ppo_epochs_num
  if is_final_epoch:
    increment *= 2
  return increment


def world_model_step_increment(hparams, is_initial_epoch):
  if is_initial_epoch:
    multiplier = hparams.initial_epoch_train_steps_multiplier
  else:
    multiplier = 1
  return multiplier * hparams.model_train_steps


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


def random_rollout_subsequences(rollouts, num_subsequences, subsequence_length):
  """Chooses a random frame sequence of given length from a set of rollouts."""
  def choose_subsequence():
    # TODO(koz4k): Weigh rollouts by their lengths so sampling is uniform over
    # frames and not rollouts.
    rollout = random.choice(rollouts)
    try:
      from_index = random.randrange(len(rollout) - subsequence_length + 1)
    except ValueError:
      # Rollout too short; repeat.
      return choose_subsequence()
    return rollout[from_index:(from_index + subsequence_length)]

  return [choose_subsequence() for _ in range(num_subsequences)]


def make_simulated_env_spec(real_env, hparams):
  """Creates a simulated environment_spec."""
  return rl.standard_atari_env_simulated_spec(
      real_env, intrinsic_reward_scale=hparams.intrinsic_reward_scale,
      model_name=hparams.generative_model,
      model_hparams=trainer_lib.create_hparams(hparams.generative_model_params),
      # Hardcoded for now. TODO(koz4k): Make it a hparam.
      video_num_input_frames=4, video_num_target_frames=1
  )


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


def train_agent(real_env, agent_model_dir, event_dir, world_model_dir, data_dir,
                hparams, completed_ppo_epochs_num, epoch=0,
                is_final_epoch=False):
  """Train the PPO agent in the simulated environment."""
  del data_dir
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_params_names = ["epochs_num", "epoch_length",
                      "learning_rate", "num_agents",
                      "optimization_epochs", "eval_every_epochs"]

  for param_name in ppo_params_names:
    ppo_param_name = "ppo_" + param_name
    if ppo_param_name in hparams:
      ppo_hparams.set_hparam(param_name, hparams.get(ppo_param_name))

  completed_ppo_epochs_num += sim_ppo_epoch_increment(hparams, is_final_epoch)
  ppo_hparams.epochs_num = completed_ppo_epochs_num

  ppo_hparams.save_models_every_epochs = 10
  ppo_hparams.world_model_dir = world_model_dir

  environment_spec = make_simulated_env_spec(real_env, hparams)

  num_input_frames = environment_spec.video_num_input_frames
  initial_frame_rollouts = real_env.current_epoch_rollouts(
      split=tf.contrib.learn.ModeKeys.TRAIN,
      minimal_rollout_frames=num_input_frames,
  )
  # TODO(koz4k): Move this to a different module.
  def initial_frame_chooser(batch_size):
    """Frame chooser."""

    deterministic_initial_frames =\
        initial_frame_rollouts[0][:num_input_frames]
    if not hparams.simulation_random_starts:
      # Deterministic starts: repeat first frames from the first rollout.
      initial_frames = [deterministic_initial_frames] * batch_size
    else:
      # Random starts: choose random initial frames from random rollouts.
      initial_frames = random_rollout_subsequences(
          initial_frame_rollouts, batch_size, num_input_frames
      )
      if hparams.simulation_flip_first_random_for_beginning:
        # Flip first entry in the batch for deterministic initial frames.
        initial_frames[0] = deterministic_initial_frames

    return np.stack([
        [frame.observation.decode() for frame in initial_frame_stack]
        for initial_frame_stack in initial_frames
    ])

  environment_spec.add_hparam("initial_frame_chooser", initial_frame_chooser)

  ppo_hparams.add_hparam("environment_spec", environment_spec)

  rl_trainer_lib.train(ppo_hparams, event_dir + "sim", agent_model_dir,
                       name_scope="ppo_sim%d" % (epoch + 1))

  return completed_ppo_epochs_num


def train_agent_real_env(
    env, agent_model_dir, event_dir, data_dir,
    hparams, completed_ppo_epochs_num, epoch=0, is_final_epoch=False):
  """Train the PPO agent in the real environment."""
  del is_final_epoch, data_dir
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

  completed_ppo_epochs_num += real_ppo_epoch_increment(hparams)
  ppo_hparams.epochs_num = completed_ppo_epochs_num
  # We do not save model, as that resets frames that we need at restarts.
  # But we need to save at the last step, so we set it very high.
  ppo_hparams.save_models_every_epochs = 1000000

  environment_spec = rl.standard_atari_env_spec(env)

  ppo_hparams.add_hparam("environment_spec", environment_spec)

  rl_trainer_lib.train(ppo_hparams, event_dir + "real", agent_model_dir,
                       name_scope="ppo_real%d" % (epoch + 1))

  # Save unfinished rollouts to history.
  env.reset()

  return completed_ppo_epochs_num


def train_world_model(
    env, data_dir, output_dir, hparams, world_model_steps_num, epoch
):
  """Train the world model on problem_name."""
  world_model_steps_num += world_model_step_increment(
      hparams, is_initial_epoch=(epoch == 0)
  )
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
      train_steps=world_model_steps_num,
      eval_steps=100,
      local_eval_frequency=2000
  )

  return world_model_steps_num


def setup_env(hparams, batch_size):
  """Setup."""
  game_mode = "Deterministic-v4"
  camel_game_name = "".join(
      [w[0].upper() + w[1:] for w in hparams.game.split("_")])
  camel_game_name += game_mode
  env_name = camel_game_name

  env = T2TGymEnv(base_env_name=env_name,
                  batch_size=batch_size,
                  grayscale=hparams.grayscale,
                  resize_width_factor=hparams.resize_width_factor,
                  resize_height_factor=hparams.resize_height_factor,
                  base_env_timesteps_limit=hparams.env_timesteps_limit,
                  max_num_noops=hparams.max_num_noops)
  return env


def evaluate_single_config(hparams, agent_model_dir):
  """Evaluate the PPO agent in the real environment."""
  eval_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  eval_hparams.num_agents = hparams.num_agents
  env = setup_env(hparams, batch_size=hparams.num_agents)
  environment_spec = rl.standard_atari_env_spec(env)
  eval_hparams.add_hparam("environment_spec", environment_spec)
  eval_hparams.add_hparam(
      "policy_to_actions_lambda", hparams.policy_to_actions_lambda
  )

  env.start_new_epoch(0)
  rl_trainer_lib.evaluate(eval_hparams, agent_model_dir)
  rollouts = env.current_epoch_rollouts()[:hparams.num_agents]
  env.close()

  assert len(rollouts) == hparams.num_agents
  return tuple(
      compute_mean_reward(rollouts, clipped) for clipped in (True, False)
  )


def evaluate_all_configs(hparams, agent_model_dir):
  """Evaluate the agent with multiple eval configurations."""
  def make_eval_hparams(hparams, policy_to_action, max_num_noops):
    hparams = copy.copy(hparams)
    hparams.add_hparam("num_agents", hparams.eval_num_agents)
    hparams.add_hparam("policy_to_actions_lambda", {
        "sample": lambda policy: policy.sample(),
        "mode": lambda policy: policy.mode()
    }[policy_to_action])
    hparams.max_num_noops = max_num_noops
    return hparams

  metrics = {}
  # Iterate over all combinations of picking actions by sampling/mode and
  # whether to do initial no-ops.
  for policy_to_action in ("mode", "sample"):
    for max_num_noops in (hparams.eval_max_num_noops, 0):
      eval_hparams = make_eval_hparams(hparams, policy_to_action, max_num_noops)
      scores = evaluate_single_config(eval_hparams, agent_model_dir)
      for (score, clipped) in zip(scores, (True, False)):
        metric_name = "mean_reward/eval/{}_{}_max_noops_{}".format(
            policy_to_action, max_num_noops,
            "clipped" if clipped else "unclipped"
        )
        metrics[metric_name] = score

  return metrics


def compute_mean_reward(rollouts, clipped):
  """Calculate mean rewards from given epoch."""
  reward_name = "reward" if clipped else "unclipped_reward"
  rewards = []
  for rollout in rollouts:
    if rollout[-1].done:
      rollout_reward = sum(getattr(frame, reward_name) for frame in rollout)
      rewards.append(rollout_reward)
  if rewards:
    mean_rewards = np.mean(rewards)
  else:
    mean_rewards = 0
  return mean_rewards


def evaluate_world_model(real_env, hparams, world_model_dir):
  """Evaluate the world model (reward accuracy)."""
  environment_spec = make_simulated_env_spec(real_env, hparams)
  environment_spec.wrappers = []

  num_input_frames = environment_spec.video_num_input_frames
  rollout_subsequences = []
  def initial_frame_chooser(batch_size):
    assert batch_size == len(rollout_subsequences)
    return np.stack([
        [frame.observation.decode() for frame in subsequence[:num_input_frames]]
        for subsequence in rollout_subsequences
    ])
  environment_spec.add_hparam("initial_frame_chooser", initial_frame_chooser)

  sim_env = SimulatedBatchGymEnv(
      environment_spec, hparams.wm_eval_batch_size, world_model_dir
  )
  subsequence_length = int(
      max(hparams.wm_eval_rollout_ratios) * hparams.ppo_epoch_length
  )
  rollouts = real_env.current_epoch_rollouts(
      split=tf.contrib.learn.ModeKeys.EVAL,
      minimal_rollout_frames=(subsequence_length + num_input_frames)
  )

  reward_accuracies_by_length = {
      int(ratio * hparams.ppo_epoch_length): []
      for ratio in hparams.wm_eval_rollout_ratios
  }
  for _ in range(hparams.wm_eval_epochs_num):
    rollout_subsequences[:] = random_rollout_subsequences(
        rollouts, hparams.wm_eval_batch_size,
        subsequence_length + num_input_frames
    )

    eval_subsequences = [
        subsequence[(num_input_frames - 1):]
        for subsequence in rollout_subsequences
    ]

    # Check that the initial observation is the same in the real and simulated
    # rollout.
    sim_init_obs = sim_env.reset()
    real_init_obs = np.stack([
        subsequence[0].observation.decode()
        for subsequence in eval_subsequences
    ])
    assert np.all(sim_init_obs == real_init_obs)

    num_same_reward = 0
    num_steps = 0
    (sim_cum_rewards, real_cum_rewards) = (
        np.zeros(hparams.wm_eval_batch_size) for _ in range(2)
    )
    for i in range(subsequence_length):
      actions = [subsequence[i].action for subsequence in eval_subsequences]
      (_, sim_rewards, _) = sim_env.step(actions)
      sim_cum_rewards += sim_rewards

      real_cum_rewards += [
          subsequence[i + 1].reward for subsequence in eval_subsequences
      ]
      num_same_reward += np.sum(sim_cum_rewards == real_cum_rewards)
      num_steps += len(real_cum_rewards)
      for (length, reward_accuracies) in six.iteritems(
          reward_accuracies_by_length
      ):
        if i + 1 == length:
          reward_accuracies.append(num_same_reward / num_steps)

  return {
      "reward_accuracy/at_{}".format(length): np.mean(reward_accuracies)
      for (length, reward_accuracies) in six.iteritems(
          reward_accuracies_by_length
      )
  }


def summarize_metrics(eval_metrics_writer, metrics, epoch):
  """Write metrics to summary."""
  for (name, value) in six.iteritems(metrics):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    eval_metrics_writer.add_summary(summary, epoch)
  eval_metrics_writer.flush()


def training_loop(hparams, output_dir, report_fn=None, report_metric=None):
  """Run the main training loop."""
  if report_fn:
    assert report_metric is not None

  # Directories
  subdirectories = ["data", "tmp", "world_model", "ppo"]
  directories = setup_directories(output_dir, subdirectories)

  epoch = -1
  data_dir = directories["data"]
  env = setup_env(hparams, batch_size=hparams.real_ppo_num_agents)
  env.start_new_epoch(epoch, data_dir)

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []
  metrics = {}

  # Collect data from the real environment with PPO or random policy.
  # TODO(lukaszkaiser): do we need option not to gather_ppo_real_env_data?
  # We could set learning_rate=0 if this flag == False.
  assert hparams.gather_ppo_real_env_data
  ppo_model_dir = directories["ppo"]
  tf.logging.info("Initial training of PPO in real environment.")
  ppo_event_dir = os.path.join(directories["world_model"],
                               "ppo_summaries/initial")
  completed_ppo_epochs_num = train_agent_real_env(
      env, ppo_model_dir, ppo_event_dir, data_dir, hparams,
      completed_ppo_epochs_num=0, epoch=epoch, is_final_epoch=False
  )
  metrics["mean_reward/train/clipped"] = compute_mean_reward(
      env.current_epoch_rollouts(), clipped=True
  )
  tf.logging.info("Mean training reward (initial): {}".format(
      metrics["mean_reward/train/clipped"]
  ))
  env.generate_data(data_dir)

  eval_metrics_event_dir = os.path.join(directories["world_model"],
                                        "eval_metrics_event_dir")
  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_event_dir)

  world_model_steps_num = 0

  for epoch in range(hparams.epochs):
    is_final_epoch = (epoch + 1) == hparams.epochs
    log = make_log_fn(epoch, log_relative_time)

    # Train world model
    log("Training world model")
    world_model_steps_num = train_world_model(
        env, data_dir, directories["world_model"], hparams,
        world_model_steps_num, epoch
    )

    # Train PPO
    log("Training PPO in simulated environment.")
    ppo_event_dir = os.path.join(directories["world_model"],
                                 "ppo_summaries", str(epoch))
    ppo_model_dir = directories["ppo"]
    if not hparams.ppo_continue_training:
      ppo_model_dir = ppo_event_dir

    completed_ppo_epochs_num = train_agent(
        env, ppo_model_dir, ppo_event_dir,
        directories["world_model"], data_dir, hparams, completed_ppo_epochs_num,
        epoch=epoch, is_final_epoch=is_final_epoch
    )

    env.start_new_epoch(epoch, data_dir)

    # Train PPO on real env (short)
    log("Training PPO in real environment.")
    completed_ppo_epochs_num = train_agent_real_env(
        env, ppo_model_dir, ppo_event_dir, data_dir, hparams,
        completed_ppo_epochs_num, epoch=epoch, is_final_epoch=is_final_epoch
    )

    if hparams.stop_loop_early:
      return 0.0

    metrics["mean_reward/train/clipped"] = compute_mean_reward(
        env.current_epoch_rollouts(), clipped=True
    )
    log("Mean training reward: {}".format(metrics["mean_reward/train/clipped"]))

    eval_metrics = evaluate_all_configs(hparams, ppo_model_dir)
    log("Agent eval metrics:\n{}".format(pprint.pformat(eval_metrics)))
    metrics.update(eval_metrics)

    env.generate_data(data_dir)

    if hparams.eval_world_model:
      wm_metrics = evaluate_world_model(
          env, hparams, directories["world_model"]
      )
      log("World model eval metrics:\n{}".format(pprint.pformat(wm_metrics)))
      metrics.update(wm_metrics)

    summarize_metrics(eval_metrics_writer, metrics, epoch)

    # Report metrics
    epoch_metrics.append(metrics)
    if report_fn:
      if report_metric == "mean_reward":
        report_fn(eval_metrics["mean_reward/eval/{}_{}_max_noops_{}".format(
            "mode", hparams.eval_max_num_noops, "unclipped")], epoch)
      else:
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
