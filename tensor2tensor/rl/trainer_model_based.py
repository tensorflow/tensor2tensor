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
from tensor2tensor.models.research.rl import make_simulated_env_fn_from_hparams
from tensor2tensor.rl import rl_utils
from tensor2tensor.rl import trainer_model_based_params
from tensor2tensor.rl.envs.simulated_batch_env import PIL_Image
from tensor2tensor.rl.envs.simulated_batch_env import PIL_ImageDraw
from tensor2tensor.rl.restarter import Restarter
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


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


def train_agent(real_env, learner, world_model_dir, hparams, epoch):
  """Train the PPO agent in the simulated environment."""
  initial_frame_chooser = rl_utils.make_initial_frame_chooser(
      real_env, hparams.frame_stack_size, hparams.simulation_random_starts,
      hparams.simulation_flip_first_random_for_beginning
  )
  env_fn = make_simulated_env_fn_from_hparams(
      real_env, hparams, hparams.simulated_batch_size, initial_frame_chooser,
      world_model_dir, os.path.join(learner.agent_model_dir,
                                    "sim_videos_{}".format(epoch))
  )
  base_algo_str = hparams.base_algo
  train_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  if hparams.wm_policy_param_sharing:
    train_hparams.optimizer_zero_grads = True

  rl_utils.update_hparams_from_hparams(
      train_hparams, hparams, base_algo_str + "_"
  )

  final_epoch = hparams.epochs - 1
  is_special_epoch = (epoch + 3) == final_epoch or (epoch + 7) == final_epoch
  is_final_epoch = epoch == final_epoch
  env_step_multiplier = 3 if is_final_epoch else 2 if is_special_epoch else 1
  learner.train(
      env_fn, train_hparams, simulated=True, save_continuously=True,
      epoch=epoch, env_step_multiplier=env_step_multiplier
  )


def train_agent_real_env(env, learner, hparams, epoch):
  """Train the PPO agent in the real environment."""
  base_algo_str = hparams.base_algo

  train_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  rl_utils.update_hparams_from_hparams(
      train_hparams, hparams, "real_" + base_algo_str + "_"
  )
  if hparams.wm_policy_param_sharing:
    train_hparams.optimizer_zero_grads = True

  env_fn = rl.make_real_env_fn(env)
  num_env_steps = real_env_step_increment(hparams)
  learner.train(
      env_fn,
      train_hparams,
      simulated=False,
      save_continuously=False,
      epoch=epoch,
      sampling_temp=hparams.real_sampling_temp,
      num_env_steps=num_env_steps,
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
  if hparams.wm_policy_param_sharing:
    model_hparams.optimizer_zero_grads = True

  restarter = Restarter("world_model", output_dir, world_model_steps_num)
  if restarter.should_skip:
    return world_model_steps_num
  with restarter.training_loop():
    train_supervised(
        problem=env,
        model_name=hparams.generative_model,
        hparams=model_hparams,
        data_dir=data_dir,
        output_dir=output_dir,
        train_steps=restarter.target_global_step,
        eval_steps=100,
        local_eval_frequency=2000
    )

  return world_model_steps_num


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

  env_fn = make_simulated_env_fn_from_hparams(
      real_env, hparams, hparams.wm_eval_batch_size, initial_frame_chooser,
      world_model_dir
  )
  sim_env = env_fn(in_graph=False)
  subsequence_length = int(
      max(hparams.wm_eval_rollout_ratios) * hparams.simulated_rollout_length
  )
  rollouts = real_env.current_epoch_rollouts(
      split=tf.estimator.ModeKeys.EVAL,
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
    def append_debug_frame_batch(sim_obs, real_obs, sim_cum_rews,
                                 real_cum_rews, sim_rews, real_rews):
      """Add a debug frame."""
      rews = [[sim_cum_rews, sim_rews], [real_cum_rews, real_rews]]
      headers = []
      for j in range(len(sim_obs)):
        local_nps = []
        for i in range(2):
          img = PIL_Image().new("RGB", (sim_obs.shape[-2], 11),)
          draw = PIL_ImageDraw().Draw(img)
          draw.text((0, 0), "c:{:3}, r:{:3}".format(int(rews[i][0][j]),
                                                    int(rews[i][1][j])),
                    fill=(255, 0, 0))
          local_nps.append(np.asarray(img))
        local_nps.append(np.zeros_like(local_nps[0]))
        headers.append(np.concatenate(local_nps, axis=1))
      errs = np.maximum(
          np.abs(sim_obs.astype(np.int) - real_obs, dtype=np.int) - 10, 0
      ).astype(np.uint8)
      headers = np.stack(headers)
      debug_frame_batches.append(  # pylint: disable=cell-var-from-loop
          np.concatenate([headers,
                          np.concatenate([sim_obs, real_obs, errs], axis=2)],
                         axis=1)
      )
    append_debug_frame_batch(sim_init_obs, real_init_obs,
                             np.zeros(hparams.wm_eval_batch_size),
                             np.zeros(hparams.wm_eval_batch_size),
                             np.zeros(hparams.wm_eval_batch_size),
                             np.zeros(hparams.wm_eval_batch_size))

    (sim_cum_rewards, real_cum_rewards) = (
        np.zeros(hparams.wm_eval_batch_size) for _ in range(2)
    )
    for i in range(subsequence_length):
      actions = [subsequence[i].action for subsequence in eval_subsequences]
      (sim_obs, sim_rewards, _) = sim_env.step(actions)
      sim_cum_rewards += sim_rewards

      real_rewards = np.array([
          subsequence[i + 1].reward for subsequence in eval_subsequences
      ])
      real_cum_rewards += real_rewards
      for (length, reward_accuracies) in six.iteritems(
          reward_accuracies_by_length
      ):
        if i + 1 == length:
          reward_accuracies.append(
              np.sum(sim_cum_rewards == real_cum_rewards) /
              len(real_cum_rewards)
          )

      real_obs = decode_real_obs(i + 1)
      append_debug_frame_batch(sim_obs, real_obs, sim_cum_rewards,
                               real_cum_rewards, sim_rewards, real_rewards)

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
  env = T2TGymEnv.setup_env_from_hparams(
      hparams, batch_size=hparams.real_batch_size,
      max_num_noops=hparams.max_num_noops
  )
  env.start_new_epoch(epoch, data_dir)

  if hparams.wm_policy_param_sharing:
    policy_model_dir = directories["world_model"]
  else:
    policy_model_dir = directories["policy"]
  learner = rl_utils.LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, policy_model_dir,
      policy_model_dir, hparams.epochs
  )

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []
  metrics = {}

  # Collect data from the real environment.
  tf.logging.info("Initial training of the policy in real environment.")
  train_agent_real_env(env, learner, hparams, epoch)
  metrics["mean_reward/train/clipped"] = rl_utils.compute_mean_reward(
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
    log = make_log_fn(epoch, log_relative_time)

    # Train world model
    log("Training world model")
    world_model_steps_num = train_world_model(
        env, data_dir, directories["world_model"], hparams,
        world_model_steps_num, epoch
    )

    # Train agent
    log("Training policy in simulated environment.")
    train_agent(env, learner, directories["world_model"], hparams, epoch)

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
      metrics["mean_reward/train/clipped"] = rl_utils.compute_mean_reward(
          env.current_epoch_rollouts(), clipped=True
      )
      log("Mean training reward: {}".format(
          metrics["mean_reward/train/clipped"]
      ))

      eval_metrics = rl_utils.evaluate_all_configs(hparams, policy_model_dir)
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
          metric_name = rl_utils.get_metric_name(
              sampling_temp=hparams.eval_sampling_temps[0],
              max_num_noops=hparams.eval_max_num_noops,
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
