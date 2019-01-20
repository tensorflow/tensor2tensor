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

r"""Evaluation script for RL agents.

Example invocation:

python -m tensor2tensor.rl.evaluator \
    --policy_dir=$HOME/t2t/rl_v1/policy \
    --eval_metrics_dir=$HOME/t2t/rl_v1/full_eval_metrics \
    --hparams_set=rlmb_base \
    --hparams='batch_size=64'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from tensor2tensor.data_generators import gym_env
from tensor2tensor.layers import common_video
from tensor2tensor.models.research import rl  # pylint: disable=unused-import
from tensor2tensor.rl import rl_utils
from tensor2tensor.rl import trainer_model_based_params  # pylint: disable=unused-import
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "", "Main directory for multi-runs.")
flags.DEFINE_integer("total_num_workers", 1, "How many workers in total.")
flags.DEFINE_string("worker_to_game_map", "", "How to map workers to games.")
flags.DEFINE_string("policy_dir", "", "Directory with policy checkpoints.")
flags.DEFINE_string("model_dir", "", "Directory with model checkpoints.")
flags.DEFINE_string(
    "eval_metrics_dir", "", "Directory to output the eval metrics at."
)
flags.DEFINE_bool("full_eval", True, "Whether to ignore the timestep limit.")
flags.DEFINE_enum(
    "agent", "policy", ["random", "policy", "planner"], "Agent type to use."
)
flags.DEFINE_bool(
    "eval_with_learner", True,
    "Whether to use the PolicyLearner.evaluate function instead of an "
    "out-of-graph one. Works only with --agent=policy."
)
flags.DEFINE_string(
    "planner_hparams_set", "planner_small", "Planner hparam set."
)
flags.DEFINE_string("planner_hparams", "", "Planner hparam overrides.")
flags.DEFINE_integer(
    "log_every_steps", 20, "Log every how many environment steps."
)
flags.DEFINE_string(
    "debug_video_path", "", "Path to save the planner debug video at."
)

# Unused flags needed to pass for multi-run infrastructure.
flags.DEFINE_bool("autotune", False, "Unused here.")
flags.DEFINE_string("objective", "", "Unused here.")
flags.DEFINE_string("client_handle", "client_0", "Unused.")
flags.DEFINE_bool("maximize_tuner_objective", True, "Unused.")
flags.DEFINE_integer("vizier_search_algorithm", 0, "Unused.")


@registry.register_hparams
def planner_tiny():
  return tf.contrib.training.HParams(
      num_rollouts=1,
      planning_horizon=2,
      rollout_agent_type="random",
      batch_size=1,
      env_type="simulated",
  )


@registry.register_hparams
def planner_small():
  return tf.contrib.training.HParams(
      num_rollouts=64,
      planning_horizon=16,
      rollout_agent_type="policy",
      batch_size=64,
      env_type="simulated",
  )


def make_env(env_type, real_env, sim_env_kwargs):
  """Factory function for envs."""
  return {
      "real": lambda: real_env.new_like(  # pylint: disable=g-long-lambda
          batch_size=sim_env_kwargs["batch_size"],
          store_rollouts=False,
      ),
      "simulated": lambda: rl_utils.SimulatedBatchGymEnvWithFixedInitialFrames(  # pylint: disable=g-long-lambda
          **sim_env_kwargs
      ),
  }[env_type]()


def make_agent(
    agent_type, env, policy_hparams, policy_dir, sampling_temp,
    sim_env_kwargs=None, frame_stack_size=None, planning_horizon=None,
    rollout_agent_type=None, batch_size=None, num_rollouts=None,
    inner_batch_size=None, video_writer=None, env_type=None):
  """Factory function for Agents."""
  if batch_size is None:
    batch_size = env.batch_size
  return {
      "random": lambda: rl_utils.RandomAgent(  # pylint: disable=g-long-lambda
          batch_size, env.observation_space, env.action_space
      ),
      "policy": lambda: rl_utils.PolicyAgent(  # pylint: disable=g-long-lambda
          batch_size, env.observation_space, env.action_space,
          policy_hparams, policy_dir, sampling_temp
      ),
      "planner": lambda: rl_utils.PlannerAgent(  # pylint: disable=g-long-lambda
          batch_size, make_agent(
              rollout_agent_type, env, policy_hparams, policy_dir,
              sampling_temp, batch_size=inner_batch_size
          ), make_env(env_type, env.env, sim_env_kwargs),
          lambda env: rl_utils.BatchStackWrapper(env, frame_stack_size),
          num_rollouts, planning_horizon,
          discount_factor=policy_hparams.gae_gamma, video_writer=video_writer
      ),
  }[agent_type]()


def make_eval_fn_with_agent(
    agent_type, planner_hparams, model_dir, log_every_steps=None,
    video_writer=None
):
  """Returns an out-of-graph eval_fn using the Agent API."""
  def eval_fn(env, loop_hparams, policy_hparams, policy_dir, sampling_temp):
    """Eval function."""
    base_env = env
    env = rl_utils.BatchStackWrapper(env, loop_hparams.frame_stack_size)
    sim_env_kwargs = rl.make_simulated_env_kwargs(
        base_env, loop_hparams, batch_size=planner_hparams.batch_size,
        model_dir=model_dir
    )
    agent = make_agent(
        agent_type, env, policy_hparams, policy_dir, sampling_temp,
        sim_env_kwargs, loop_hparams.frame_stack_size,
        planner_hparams.planning_horizon, planner_hparams.rollout_agent_type,
        num_rollouts=planner_hparams.num_rollouts,
        inner_batch_size=planner_hparams.batch_size, video_writer=video_writer,
        env_type=planner_hparams.env_type
    )
    rl_utils.run_rollouts(
        env, agent, env.reset(), log_every_steps=log_every_steps
    )
    assert len(base_env.current_epoch_rollouts()) == env.batch_size
  return eval_fn


def evaluate(
    loop_hparams, planner_hparams, policy_dir, model_dir, eval_metrics_dir,
    agent_type, eval_with_learner, log_every_steps, debug_video_path,
    report_fn=None, report_metric=None
):
  """Evaluate."""
  if eval_with_learner:
    assert agent_type == "policy"

  if report_fn:
    assert report_metric is not None

  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_dir)
  video_writer = None
  kwargs = {}
  if not eval_with_learner:
    if debug_video_path:
      video_writer = common_video.WholeVideoWriter(
          fps=10, output_path=debug_video_path, file_format="avi")
    kwargs["eval_fn"] = make_eval_fn_with_agent(
        agent_type, planner_hparams, model_dir, log_every_steps=log_every_steps,
        video_writer=video_writer
    )
  eval_metrics = rl_utils.evaluate_all_configs(
      loop_hparams, policy_dir, **kwargs
  )
  rl_utils.summarize_metrics(eval_metrics_writer, eval_metrics, 0)

  if video_writer is not None:
    video_writer.finish_to_disk()

  # Report metrics
  if report_fn:
    if report_metric == "mean_reward":
      metric_name = rl_utils.get_metric_name(
          sampling_temp=loop_hparams.eval_sampling_temps[0],
          max_num_noops=loop_hparams.eval_max_num_noops,
          clipped=False
      )
      report_fn(eval_metrics[metric_name], 0)
    else:
      report_fn(eval_metrics[report_metric], 0)
  return eval_metrics


def get_game_for_worker(map_name, directory_id):
  """Get game for the given worker (directory) id."""
  if map_name == "v100unfriendly":
    games = ["chopper_command", "boxing", "asterix", "seaquest"]
    worker_per_game = 5
  elif map_name == "human_nice":
    games = gym_env.ATARI_GAMES_WITH_HUMAN_SCORE_NICE
    worker_per_game = 5
  else:
    raise ValueError("Unknown worker to game map name: %s" % map_name)
  games.sort()
  game_id = (directory_id - 1) // worker_per_game
  tf.logging.info("Getting game %d from %s." % (game_id, games))
  return games[game_id]


def main(_):
  now = datetime.datetime.now()
  now_tag = now.strftime("%Y_%m_%d_%H_%M")
  loop_hparams = trainer_lib.create_hparams(
      FLAGS.loop_hparams_set, FLAGS.loop_hparams
  )
  if FLAGS.worker_to_game_map and FLAGS.total_num_workers > 1:
    loop_hparams.game = get_game_for_worker(
        FLAGS.worker_to_game_map, FLAGS.worker_id + 1)
    tf.logging.info("Set game to %s." % loop_hparams.game)
  if FLAGS.full_eval:
    loop_hparams.eval_rl_env_max_episode_steps = -1
  planner_hparams = trainer_lib.create_hparams(
      FLAGS.planner_hparams_set, FLAGS.planner_hparams
  )
  policy_dir = FLAGS.policy_dir
  model_dir = FLAGS.model_dir
  eval_metrics_dir = FLAGS.eval_metrics_dir
  if FLAGS.output_dir:
    cur_dir = FLAGS.output_dir
    if FLAGS.total_num_workers > 1:
      cur_dir = os.path.join(cur_dir, "%d" % (FLAGS.worker_id + 1))
    policy_dir = os.path.join(cur_dir, "policy")
    model_dir = os.path.join(cur_dir, "world_model")
    eval_metrics_dir = os.path.join(cur_dir, "evaluator_" + now_tag)
    tf.logging.info("Writing metrics to %s." % eval_metrics_dir)
    if not tf.gfile.Exists(eval_metrics_dir):
      tf.gfile.MkDir(eval_metrics_dir)
  evaluate(
      loop_hparams, planner_hparams, policy_dir, model_dir,
      eval_metrics_dir, FLAGS.agent, FLAGS.eval_with_learner,
      FLAGS.log_every_steps if FLAGS.log_every_steps > 0 else None,
      debug_video_path=FLAGS.debug_video_path
  )


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
