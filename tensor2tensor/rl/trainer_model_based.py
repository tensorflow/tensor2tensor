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
import math
import os
import pprint
import random
import time

import numpy as np
import six

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.layers import common_video
from tensor2tensor.models.research import rl
from tensor2tensor.rl import trainer_model_based_params
from tensor2tensor.rl.dopamine_connector import DQNLearner
from tensor2tensor.rl.ppo_learner import PPOLearner
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


LEARNERS = {
    "ppo": PPOLearner,
    "dqn": DQNLearner,
}


def update_hparams_from_hparams(target_hparams, source_hparams, prefix):
  """Copy a subset of hparams to target_hparams."""
  for (param_name, param_value) in six.iteritems(source_hparams.values()):
    if param_name.startswith(prefix):
      target_hparams.set_hparam(param_name[len(prefix):], param_value)


def real_env_step_increment(hparams):
  """Real env step increment."""
  return int(math.ceil(
      hparams.num_real_env_frames / hparams.epochs
  ))


def world_model_step_increment(hparams, is_initial_epoch):
  if is_initial_epoch:
    multiplier = hparams.initial_epoch_train_steps_multiplier
  else:
    multiplier = 1
  return multiplier * hparams.model_train_steps


def setup_directories(base_dir, subdirs):
  """Setup directories."""
  base_dir = os.path.expanduser(base_dir)
  tf.gfile.MakeDirs(base_dir)

  all_dirs = {}
  for subdir in subdirs:
    if isinstance(subdir, six.string_types):
      subdir_tuple = (subdir,)
    else:
      subdir_tuple = subdir
    dir_name = os.path.join(base_dir, *subdir_tuple)
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


def make_simulated_env_fn(
    real_env, hparams, batch_size, initial_frame_chooser, model_dir):
  """Creates a simulated env_fn."""
  return rl.make_simulated_env_fn(
      reward_range=real_env.reward_range,
      observation_space=real_env.observation_space,
      action_space=real_env.action_space,
      frame_stack_size=hparams.frame_stack_size,
      frame_height=real_env.frame_height, frame_width=real_env.frame_width,
      initial_frame_chooser=initial_frame_chooser, batch_size=batch_size,
      model_name=hparams.generative_model,
      model_hparams=trainer_lib.create_hparams(hparams.generative_model_params),
      model_dir=model_dir,
      intrinsic_reward_scale=hparams.intrinsic_reward_scale,
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


def train_agent(
    real_env, learner, world_model_dir, hparams, epoch, is_final_epoch):
  """Train the PPO agent in the simulated environment."""
  frame_stack_size = hparams.frame_stack_size
  initial_frame_rollouts = real_env.current_epoch_rollouts(
      split=tf.contrib.learn.ModeKeys.TRAIN,
      minimal_rollout_frames=frame_stack_size,
  )
  # TODO(koz4k): Move this to a different module.
  def initial_frame_chooser(batch_size):
    """Frame chooser."""

    deterministic_initial_frames =\
        initial_frame_rollouts[0][:frame_stack_size]
    if not hparams.simulation_random_starts:
      # Deterministic starts: repeat first frames from the first rollout.
      initial_frames = [deterministic_initial_frames] * batch_size
    else:
      # Random starts: choose random initial frames from random rollouts.
      initial_frames = random_rollout_subsequences(
          initial_frame_rollouts, batch_size, frame_stack_size
      )
      if hparams.simulation_flip_first_random_for_beginning:
        # Flip first entry in the batch for deterministic initial frames.
        initial_frames[0] = deterministic_initial_frames

    return np.stack([
        [frame.observation.decode() for frame in initial_frame_stack]
        for initial_frame_stack in initial_frames
    ])
  env_fn = make_simulated_env_fn(
      real_env, hparams, hparams.simulated_batch_size, initial_frame_chooser,
      world_model_dir
  )
  base_algo_str = hparams.base_algo
  train_hparams = trainer_lib.create_hparams(hparams.base_algo_params)

  update_hparams_from_hparams(
      train_hparams, hparams, base_algo_str + "_"
  )

  env_step_multiplier = 1 if not is_final_epoch else 2
  learner.train(
      env_fn, train_hparams, simulated=True, save_continuously=True,
      epoch=epoch, env_step_multiplier=env_step_multiplier
  )


def train_agent_real_env(env, learner, hparams, epoch):
  """Train the PPO agent in the real environment."""
  base_algo_str = hparams.base_algo

  train_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  update_hparams_from_hparams(
      train_hparams, hparams, "real_" + base_algo_str + "_"
  )

  env_fn = rl.make_real_env_fn(env)
  num_env_steps = real_env_step_increment(hparams)
  learner.train(
      env_fn, train_hparams, simulated=False, save_continuously=False,
      epoch=epoch, num_env_steps=num_env_steps
  )
  # Save unfinished rollouts to history.
  env.reset()


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


def setup_env(hparams, batch_size, max_num_noops):
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
                  max_num_noops=max_num_noops)
  return env


def evaluate_single_config(hparams, stochastic, max_num_noops, agent_model_dir):
  """Evaluate the PPO agent in the real environment."""
  eval_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  env = setup_env(
      hparams, batch_size=hparams.eval_batch_size, max_num_noops=max_num_noops
  )
  env.start_new_epoch(0)
  env_fn = rl.make_real_env_fn(env)
  learner = LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, base_event_dir=None,
      agent_model_dir=agent_model_dir
  )
  learner.evaluate(env_fn, eval_hparams, stochastic)
  rollouts = env.current_epoch_rollouts()
  env.close()

  return tuple(
      compute_mean_reward(rollouts, clipped) for clipped in (True, False)
  )


def get_metric_name(stochastic, max_num_noops, clipped):
  return "mean_reward/eval/stochastic_{}_max_noops_{}_{}".format(
      stochastic, max_num_noops, "clipped" if clipped else "unclipped")


def evaluate_all_configs(hparams, agent_model_dir):
  """Evaluate the agent with multiple eval configurations."""
  metrics = {}
  # Iterate over all combinations of picking actions by sampling/mode and
  # whether to do initial no-ops.
  for stochastic in (True, False):
    for max_num_noops in (hparams.eval_max_num_noops, 0):
      scores = evaluate_single_config(
          hparams, stochastic, max_num_noops, agent_model_dir
      )
      for (score, clipped) in zip(scores, (True, False)):
        metric_name = get_metric_name(stochastic, max_num_noops, clipped)
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


def evaluate_world_model(real_env, hparams, world_model_dir, debug_video_path):
  """Evaluate the world model (reward accuracy)."""
  frame_stack_size = hparams.frame_stack_size
  rollout_subsequences = []
  def initial_frame_chooser(batch_size):
    assert batch_size == len(rollout_subsequences)
    return np.stack([
        [frame.observation.decode() for frame in subsequence[:frame_stack_size]]
        for subsequence in rollout_subsequences
    ])

  env_fn = make_simulated_env_fn(
      real_env, hparams, hparams.wm_eval_batch_size, initial_frame_chooser,
      world_model_dir
  )
  sim_env = env_fn(in_graph=False)
  subsequence_length = int(
      max(hparams.wm_eval_rollout_ratios) * hparams.simulated_rollout_length
  )
  rollouts = real_env.current_epoch_rollouts(
      split=tf.contrib.learn.ModeKeys.EVAL,
      minimal_rollout_frames=(subsequence_length + frame_stack_size)
  )

  video_writer = common_video.WholeVideoWriter(
      fps=10, output_path=debug_video_path, file_format="avi"
  )

  reward_accuracies_by_length = {
      int(ratio * hparams.simulated_rollout_length): []
      for ratio in hparams.wm_eval_rollout_ratios
  }
  for _ in range(hparams.wm_eval_num_batches):
    rollout_subsequences[:] = random_rollout_subsequences(
        rollouts, hparams.wm_eval_batch_size,
        subsequence_length + frame_stack_size
    )

    eval_subsequences = [
        subsequence[(frame_stack_size - 1):]
        for subsequence in rollout_subsequences
    ]

    # Check that the initial observation is the same in the real and simulated
    # rollout.
    sim_init_obs = sim_env.reset()
    def decode_real_obs(index):
      return np.stack([
          subsequence[index].observation.decode()
          for subsequence in eval_subsequences  # pylint: disable=cell-var-from-loop
      ])
    real_init_obs = decode_real_obs(0)
    assert np.all(sim_init_obs == real_init_obs)

    debug_frame_batches = []
    def append_debug_frame_batch(sim_obs, real_obs):
      errs = np.maximum(
          np.abs(sim_obs.astype(np.int) - real_obs, dtype=np.int) - 10, 0
      ).astype(np.uint8)
      debug_frame_batches.append(  # pylint: disable=cell-var-from-loop
          np.concatenate([sim_obs, real_obs, errs], axis=2)
      )
    append_debug_frame_batch(sim_init_obs, real_init_obs)

    (sim_cum_rewards, real_cum_rewards) = (
        np.zeros(hparams.wm_eval_batch_size) for _ in range(2)
    )
    for i in range(subsequence_length):
      actions = [subsequence[i].action for subsequence in eval_subsequences]
      (sim_obs, sim_rewards, _) = sim_env.step(actions)
      sim_cum_rewards += sim_rewards

      real_cum_rewards += [
          subsequence[i + 1].reward for subsequence in eval_subsequences
      ]
      for (length, reward_accuracies) in six.iteritems(
          reward_accuracies_by_length
      ):
        if i + 1 == length:
          reward_accuracies.append(
              np.sum(sim_cum_rewards == real_cum_rewards) /
              len(real_cum_rewards)
          )

      real_obs = decode_real_obs(i + 1)
      append_debug_frame_batch(sim_obs, real_obs)

    for debug_frames in np.stack(debug_frame_batches, axis=1):
      for debug_frame in debug_frames:
        video_writer.write(debug_frame)

  video_writer.finish_to_disk()

  return {
      "reward_accuracy/at_{}".format(length): np.mean(reward_accuracies)
      for (length, reward_accuracies) in six.iteritems(
          reward_accuracies_by_length
      )
  }


def load_metrics(event_dir, epoch):
  """Loads metrics for this epoch if they have already been written.

  This reads the entire event file but it's small with just per-epoch metrics.

  Args:
    event_dir: TODO(koz4k): Document this.
    epoch: TODO(koz4k): Document this.

  Returns:
    metrics.
  """
  metrics = {}
  for filename in tf.gfile.ListDirectory(event_dir):
    path = os.path.join(event_dir, filename)
    for event in tf.train.summary_iterator(path):
      if event.step == epoch and event.HasField("summary"):
        value = event.summary.value[0]
        metrics[value.tag] = value.simple_value
  return metrics


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
  subdirectories = [
      "data", "tmp", "world_model", ("world_model", "debug_videos"),
      "policy", "eval_metrics"
  ]
  directories = setup_directories(output_dir, subdirectories)

  epoch = -1
  data_dir = directories["data"]
  env = setup_env(
      hparams, batch_size=hparams.real_batch_size,
      max_num_noops=hparams.max_num_noops
  )
  env.start_new_epoch(epoch, data_dir)

  learner = LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, directories["policy"],
      directories["policy"]
  )

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []
  metrics = {}

  # Collect data from the real environment.
  policy_model_dir = directories["policy"]
  tf.logging.info("Initial training of the policy in real environment.")
  train_agent_real_env(env, learner, hparams, epoch)
  metrics["mean_reward/train/clipped"] = compute_mean_reward(
      env.current_epoch_rollouts(), clipped=True
  )
  tf.logging.info("Mean training reward (initial): {}".format(
      metrics["mean_reward/train/clipped"]
  ))
  env.generate_data(data_dir)

  eval_metrics_writer = tf.summary.FileWriter(
      directories["eval_metrics"]
  )

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

    # Train agent
    log("Training policy in simulated environment.")
    train_agent(
        env, learner, directories["world_model"], hparams, epoch, is_final_epoch
    )

    env.start_new_epoch(epoch, data_dir)

    # Train agent on real env (short)
    log("Training policy in real environment.")
    train_agent_real_env(env, learner, hparams, epoch)

    if hparams.stop_loop_early:
      return 0.0

    env.generate_data(data_dir)

    metrics = load_metrics(directories["eval_metrics"], epoch)
    if metrics:
      # Skip eval if metrics have already been written for this epoch. Otherwise
      # we'd overwrite them with wrong data.
      log("Metrics found for this epoch, skipping evaluation.")
    else:
      metrics["mean_reward/train/clipped"] = compute_mean_reward(
          env.current_epoch_rollouts(), clipped=True
      )
      log("Mean training reward: {}".format(
          metrics["mean_reward/train/clipped"]
      ))

      eval_metrics = evaluate_all_configs(hparams, policy_model_dir)
      log("Agent eval metrics:\n{}".format(pprint.pformat(eval_metrics)))
      metrics.update(eval_metrics)

      if hparams.eval_world_model:
        debug_video_path = os.path.join(
            directories["world_model", "debug_videos"],
            "{}.avi".format(env.current_epoch)
        )
        wm_metrics = evaluate_world_model(
            env, hparams, directories["world_model"], debug_video_path
        )
        log("World model eval metrics:\n{}".format(pprint.pformat(wm_metrics)))
        metrics.update(wm_metrics)

      summarize_metrics(eval_metrics_writer, metrics, epoch)

      # Report metrics
      if report_fn:
        if report_metric == "mean_reward":
          metric_name = get_metric_name(
              stochastic=True, max_num_noops=hparams.eval_max_num_noops,
              clipped=False
          )
          report_fn(eval_metrics[metric_name], epoch)
        else:
          report_fn(eval_metrics[report_metric], epoch)

    epoch_metrics.append(metrics)

  # Return the evaluation metrics from the final epoch
  return epoch_metrics[-1]


def main(_):
  hp = trainer_model_based_params.create_loop_hparams()
  assert not FLAGS.job_dir_to_evaluate
  training_loop(hp, FLAGS.output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
