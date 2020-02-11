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
from tensor2tensor.utils import hparam
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf


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
flags.DEFINE_integer("eval_batch_size", 64, "Number of games to evaluate.")
flags.DEFINE_integer("eval_step_limit", 50000,
                     "Maximum number of time steps, ignored if -1.")
flags.DEFINE_enum(
    "agent", "policy", ["random", "policy", "planner"], "Agent type to use."
)
# Evaluator doesn't report metrics for agent on the simulated env because we
# don't collect rollouts there. It's just for generating videos.
# TODO(koz4k): Enable reporting metrics from simulated env by refactoring
# T2TEnv to a wrapper storing rollouts and providing Problem interface for any
# batch env.
flags.DEFINE_enum(
    "mode", "agent_real", ["agent_real", "agent_simulated", "model"],
    "Evaluation mode; report agent's score on real or simulated env, or model's"
    " reward accuracy."
)
# TODO(koz4k): Switch to out-of-graph evaluation everywhere and remove this
# flag.
flags.DEFINE_bool(
    "eval_with_learner", False,
    "Whether to use the PolicyLearner.evaluate function instead of an "
    "out-of-graph one. Works only with --agent=policy."
)
flags.DEFINE_string(
    "planner_hparams_set", "planner_small", "Planner hparam set."
)
flags.DEFINE_string("planner_hparams", "", "Planner hparam overrides.")
flags.DEFINE_integer(
    "log_every_steps", 5, "Log every how many environment steps."
)
flags.DEFINE_string(
    "debug_video_path", "", "Path to save the debug video at."
)
flags.DEFINE_integer(
    "num_debug_videos", 1, "Number of debug videos to generate."
)
flags.DEFINE_integer(
    "random_starts_step_limit", 10000,
    "Number of frames to choose from for random starts of the simulated env."
)
flags.DEFINE_bool(
    "all_epochs", False,
    "Whether to run the evaluator on policy checkpoints from all epochs."
)

# Unused flags needed to pass for multi-run infrastructure.
flags.DEFINE_bool("autotune", False, "Unused here.")
flags.DEFINE_string("objective", "", "Unused here.")
flags.DEFINE_string("client_handle", "client_0", "Unused.")
flags.DEFINE_bool("maximize_tuner_objective", True, "Unused.")
flags.DEFINE_integer("vizier_search_algorithm", 0, "Unused.")


@registry.register_hparams
def planner_tiny():
  return hparam.HParams(
      num_rollouts=1,
      planning_horizon=2,
      rollout_agent_type="random",
      batch_size=1,
      env_type="simulated",
      uct_const=0.0,
      uniform_first_action=True,
  )


@registry.register_hparams
def planner_small():
  return hparam.HParams(
      num_rollouts=64,
      planning_horizon=16,
      rollout_agent_type="policy",
      batch_size=64,
      env_type="simulated",
      uct_const=0.0,
      uniform_first_action=True,
  )


@registry.register_hparams
def planner_base():
  return hparam.HParams(
      num_rollouts=96,
      batch_size=96,
      planning_horizon=8,
      rollout_agent_type="policy",
      env_type="simulated",
      uct_const=0.,
      uniform_first_action=True,
  )


# Tuning of uniform_first_action and uct_const. Default params repeated for
# clarity.


@registry.register_hparams
def planner_guess1():
  hparams = planner_base()
  hparams.uniform_first_action = False
  hparams.uct_const = 0.
  return hparams


@registry.register_hparams
def planner_guess2():
  hparams = planner_base()
  hparams.uniform_first_action = True
  hparams.uct_const = 3.
  return hparams


@registry.register_hparams
def planner_guess3():
  hparams = planner_base()
  hparams.uniform_first_action = False
  hparams.uct_const = 2.
  return hparams


# Tuning of uct_const, num_collouts and normalizer_window_size.


@registry.register_hparams
def planner_guess4():
  hparams = planner_base()
  hparams.uct_const = 2
  hparams.num_rollouts = 96
  hparams.normalizer_window_size = 30
  return hparams


@registry.register_hparams
def planner_guess5():
  hparams = planner_base()
  hparams.uct_const = 2
  hparams.num_rollouts = 3 * 96
  hparams.normalizer_window_size = 30
  return hparams


@registry.register_hparams
def planner_guess6():
  hparams = planner_base()
  hparams.uct_const = 4
  hparams.num_rollouts = 96
  hparams.normalizer_window_size = 30
  return hparams


@registry.register_hparams
def planner_guess7():
  hparams = planner_base()
  hparams.uct_const = 4
  hparams.num_rollouts = 3 * 96
  hparams.normalizer_window_size = 30
  return hparams


@registry.register_hparams
def planner_guess8():
  hparams = planner_base()
  hparams.uct_const = 2
  hparams.num_rollouts = 3 * 96
  hparams.normalizer_window_size = 300
  return hparams


@registry.register_hparams
def planner_guess9():
  hparams = planner_base()
  hparams.uct_const = 4
  hparams.num_rollouts = 3 * 96
  hparams.normalizer_window_size = 300
  return hparams


@registry.register_hparams
def planner_guess0():
  hparams = planner_base()
  hparams.uct_const = 6
  hparams.num_rollouts = 4 * 96
  hparams.normalizer_window_size = 30
  return hparams


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
    sim_env_kwargs_fn=None, frame_stack_size=None, rollout_agent_type=None,
    batch_size=None, inner_batch_size=None, env_type=None, **planner_kwargs
):
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
          ), make_env(env_type, env.env, sim_env_kwargs_fn()),
          lambda env: rl_utils.BatchStackWrapper(env, frame_stack_size),
          discount_factor=policy_hparams.gae_gamma, **planner_kwargs
      ),
  }[agent_type]()


def collect_frames_for_random_starts(
    storage_env, stacked_env, agent, frame_stack_size, random_starts_step_limit,
    log_every_steps=None
):
  """Collects frames from real env for random starts of simulated env."""
  del frame_stack_size
  storage_env.start_new_epoch(0)
  tf.logging.info(
      "Collecting %d frames for random starts.", random_starts_step_limit
  )
  rl_utils.run_rollouts(
      stacked_env, agent, stacked_env.reset(),
      step_limit=random_starts_step_limit,
      many_rollouts_from_each_env=True,
      log_every_steps=log_every_steps,
  )
  # Save unfinished rollouts to history.
  stacked_env.reset()


def make_agent_from_hparams(
    agent_type, base_env, stacked_env, loop_hparams, policy_hparams,
    planner_hparams, model_dir, policy_dir, sampling_temp, video_writers=()
):
  """Creates an Agent from hparams."""
  def sim_env_kwargs_fn():
    return rl.make_simulated_env_kwargs(
        base_env, loop_hparams, batch_size=planner_hparams.batch_size,
        model_dir=model_dir
    )
  planner_kwargs = planner_hparams.values()
  planner_kwargs.pop("batch_size")
  planner_kwargs.pop("rollout_agent_type")
  planner_kwargs.pop("env_type")
  return make_agent(
      agent_type, stacked_env, policy_hparams, policy_dir, sampling_temp,
      sim_env_kwargs_fn, loop_hparams.frame_stack_size,
      planner_hparams.rollout_agent_type,
      inner_batch_size=planner_hparams.batch_size,
      env_type=planner_hparams.env_type,
      video_writers=video_writers, **planner_kwargs
  )


def make_eval_fn_with_agent(
    agent_type, eval_mode, planner_hparams, model_dir, log_every_steps=None,
    video_writers=(), random_starts_step_limit=None
):
  """Returns an out-of-graph eval_fn using the Agent API."""
  def eval_fn(env, loop_hparams, policy_hparams, policy_dir, sampling_temp):
    """Eval function."""
    base_env = env
    env = rl_utils.BatchStackWrapper(env, loop_hparams.frame_stack_size)
    agent = make_agent_from_hparams(
        agent_type, base_env, env, loop_hparams, policy_hparams,
        planner_hparams, model_dir, policy_dir, sampling_temp, video_writers
    )

    if eval_mode == "agent_simulated":
      real_env = base_env.new_like(batch_size=1)
      stacked_env = rl_utils.BatchStackWrapper(
          real_env, loop_hparams.frame_stack_size
      )
      collect_frames_for_random_starts(
          real_env, stacked_env, agent, loop_hparams.frame_stack_size,
          random_starts_step_limit, log_every_steps
      )
      initial_frame_chooser = rl_utils.make_initial_frame_chooser(
          real_env, loop_hparams.frame_stack_size,
          simulation_random_starts=True,
          simulation_flip_first_random_for_beginning=False,
          split=None,
      )
      env_fn = rl.make_simulated_env_fn_from_hparams(
          real_env, loop_hparams, batch_size=loop_hparams.eval_batch_size,
          initial_frame_chooser=initial_frame_chooser, model_dir=model_dir
      )
      sim_env = env_fn(in_graph=False)
      env = rl_utils.BatchStackWrapper(sim_env, loop_hparams.frame_stack_size)

    kwargs = {}
    if not agent.records_own_videos:
      kwargs["video_writers"] = video_writers
    step_limit = base_env.rl_env_max_episode_steps
    if step_limit == -1:
      step_limit = None
    rl_utils.run_rollouts(
        env, agent, env.reset(), log_every_steps=log_every_steps,
        step_limit=step_limit, **kwargs
    )
    if eval_mode == "agent_real":
      assert len(base_env.current_epoch_rollouts()) == env.batch_size
  return eval_fn


def evaluate_world_model(
    agent_type, loop_hparams, planner_hparams, model_dir, policy_dir,
    random_starts_step_limit, debug_video_path, log_every_steps
):
  """Evaluates the world model."""
  if debug_video_path:
    debug_video_path = os.path.join(debug_video_path, "0.avi")

  storage_env = rl_utils.setup_env(loop_hparams, batch_size=1, max_num_noops=0)
  stacked_env = rl_utils.BatchStackWrapper(
      storage_env, loop_hparams.frame_stack_size
  )
  policy_hparams = trainer_lib.create_hparams(loop_hparams.base_algo_params)
  agent = make_agent_from_hparams(
      agent_type, storage_env, stacked_env, loop_hparams, policy_hparams,
      planner_hparams, model_dir, policy_dir,
      # TODO(koz4k): Loop over eval_sampling_temps?
      sampling_temp=loop_hparams.eval_sampling_temps[0],
  )
  collect_frames_for_random_starts(
      storage_env, stacked_env, agent, loop_hparams.frame_stack_size,
      random_starts_step_limit, log_every_steps
  )
  return rl_utils.evaluate_world_model(
      storage_env, loop_hparams, model_dir, debug_video_path, split=None
  )


def evaluate(
    loop_hparams, planner_hparams, policy_dir, model_dir, eval_metrics_dir,
    agent_type, eval_mode, eval_with_learner, log_every_steps, debug_video_path,
    num_debug_videos=1, random_starts_step_limit=None,
    report_fn=None, report_metric=None
):
  """Evaluate."""
  if eval_with_learner:
    assert agent_type == "policy"

  if report_fn:
    assert report_metric is not None

  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_dir)
  video_writers = ()
  kwargs = {}
  if eval_mode in ["agent_real", "agent_simulated"]:
    if not eval_with_learner:
      if debug_video_path:
        tf.gfile.MakeDirs(debug_video_path)
        video_writers = [
            common_video.WholeVideoWriter(  # pylint: disable=g-complex-comprehension
                fps=10,
                output_path=os.path.join(debug_video_path, "{}.avi".format(i)),
                file_format="avi",
            )
            for i in range(num_debug_videos)
        ]
      kwargs["eval_fn"] = make_eval_fn_with_agent(
          agent_type, eval_mode, planner_hparams, model_dir,
          log_every_steps=log_every_steps,
          video_writers=video_writers,
          random_starts_step_limit=random_starts_step_limit
      )
    eval_metrics = rl_utils.evaluate_all_configs(
        loop_hparams, policy_dir, **kwargs
    )
  else:
    eval_metrics = evaluate_world_model(
        agent_type, loop_hparams, planner_hparams, model_dir, policy_dir,
        random_starts_step_limit, debug_video_path, log_every_steps
    )
  rl_utils.summarize_metrics(eval_metrics_writer, eval_metrics, 0)

  for video_writer in video_writers:
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


def evaluate_all_epochs(
    loop_hparams, planner_hparams, policy_dir, model_dir, eval_metrics_dir,
    *args, **kwargs
):
  epoch_policy_dirs = tf.gfile.Glob(os.path.join(policy_dir, "epoch_*"))
  for epoch_policy_dir in epoch_policy_dirs:
    epoch_metrics_dir = os.path.join(eval_metrics_dir, "epoch_{}".format(
        epoch_policy_dir.split("_")[-1]
    ))
    evaluate(
        loop_hparams, planner_hparams, epoch_policy_dir, model_dir,
        epoch_metrics_dir, *args, **kwargs
    )


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
  loop_hparams.eval_rl_env_max_episode_steps = FLAGS.eval_step_limit
  loop_hparams.eval_batch_size = FLAGS.eval_batch_size
  planner_hparams = trainer_lib.create_hparams(
      FLAGS.planner_hparams_set, FLAGS.planner_hparams
  )
  policy_dir = FLAGS.policy_dir
  model_dir = FLAGS.model_dir
  eval_metrics_dir = FLAGS.eval_metrics_dir
  debug_video_path = FLAGS.debug_video_path
  evaluate_fn = evaluate
  if FLAGS.output_dir:
    cur_dir = FLAGS.output_dir
    if FLAGS.total_num_workers > 1:
      cur_dir = os.path.join(cur_dir, "%d" % (FLAGS.worker_id + 1))
    policy_dir = os.path.join(cur_dir, "policy")
    model_dir = os.path.join(cur_dir, "world_model")
    eval_dir_basename = "evaluator_"
    if FLAGS.agent == "planner":
      eval_dir_basename = FLAGS.planner_hparams_set + "_"
    eval_metrics_dir = os.path.join(cur_dir, eval_dir_basename + now_tag)
    debug_video_path = eval_metrics_dir
    tf.logging.info("Writing metrics to %s." % eval_metrics_dir)
    if not tf.gfile.Exists(eval_metrics_dir):
      tf.gfile.MkDir(eval_metrics_dir)
    if FLAGS.all_epochs:
      evaluate_fn = evaluate_all_epochs
  evaluate_fn(
      loop_hparams, planner_hparams, policy_dir, model_dir,
      eval_metrics_dir, FLAGS.agent, FLAGS.mode, FLAGS.eval_with_learner,
      FLAGS.log_every_steps if FLAGS.log_every_steps > 0 else None,
      debug_video_path=debug_video_path,
      num_debug_videos=FLAGS.num_debug_videos,
      random_starts_step_limit=FLAGS.random_starts_step_limit,
  )


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
