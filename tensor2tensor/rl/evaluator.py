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

import numpy as np

from tensor2tensor.models.research import rl  # pylint: disable=unused-import
from tensor2tensor.rl import rl_utils
from tensor2tensor.rl import trainer_model_based_params  # pylint: disable=unused-import
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("policy_dir", "", "Directory with policy checkpoints.")
flags.DEFINE_string(
    "eval_metrics_dir", "", "Directory to output the eval metrics at."
)
flags.DEFINE_bool("full_eval", True, "Whether to ignore the timestep limit.")
flags.DEFINE_enum("agent", "policy", ["random", "policy"], "Agent type to use.")
flags.DEFINE_bool(
    "eval_with_learner", True,
    "Whether to use the PolicyLearner.evaluate function instead of an "
    "out-of-graph one. Works only with --agent=policy."
)


def make_agent(
    agent_type, env, policy_hparams, policy_dir, sampling_temp
):
  """Factory function for Agents."""
  return {
      "random": lambda: rl_utils.RandomAgent(
          env.batch_size, env.observation_space, env.action_space
      ),
      "policy": lambda: rl_utils.PolicyAgent(
          env.batch_size, env.observation_space, env.action_space,
          policy_hparams, policy_dir, sampling_temp
      ),
  }[agent_type]()


def make_eval_fn_with_agent(agent_type):
  """Returns an out-of-graph eval_fn using the Agent API."""
  def eval_fn(env, hparams, policy_hparams, policy_dir, sampling_temp):
    """Eval function."""
    base_env = env
    env = rl_utils.BatchStackWrapper(env, hparams.frame_stack_size)
    agent = make_agent(
        agent_type, env, policy_hparams, policy_dir, sampling_temp
    )
    num_dones = 0
    first_dones = [False] * env.batch_size
    observations = env.reset()
    while num_dones < env.batch_size:
      actions = agent.act(observations)
      (observations, _, dones) = env.step(actions)
      observations = list(observations)
      now_done_indices = []
      for (i, done) in enumerate(dones):
        if done and not first_dones[i]:
          now_done_indices.append(i)
          first_dones[i] = True
          num_dones += 1
      if now_done_indices:
        # Reset only envs done the first time in this timestep to ensure that
        # we collect exactly 1 rollout from each env.
        reset_observations = env.reset(now_done_indices)
        for (i, observation) in zip(now_done_indices, reset_observations):
          observations[i] = observation
      observations = np.array(observations)
    assert len(base_env.current_epoch_rollouts()) == env.batch_size
  return eval_fn


def evaluate(
    hparams, policy_dir, eval_metrics_dir, agent_type, eval_with_learner,
    report_fn=None, report_metric=None
):
  """Evaluate."""
  if eval_with_learner:
    assert agent_type == "policy"

  if report_fn:
    assert report_metric is not None

  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_dir)
  kwargs = {}
  if not eval_with_learner:
    kwargs["eval_fn"] = make_eval_fn_with_agent(agent_type)
  eval_metrics = rl_utils.evaluate_all_configs(hparams, policy_dir, **kwargs)
  rl_utils.summarize_metrics(eval_metrics_writer, eval_metrics, 0)

  # Report metrics
  if report_fn:
    if report_metric == "mean_reward":
      metric_name = rl_utils.get_metric_name(
          sampling_temp=hparams.eval_sampling_temps[0],
          max_num_noops=hparams.eval_max_num_noops,
          clipped=False
      )
      report_fn(eval_metrics[metric_name], 0)
    else:
      report_fn(eval_metrics[report_metric], 0)
  return eval_metrics


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)
  if FLAGS.full_eval:
    hparams.eval_rl_env_max_episode_steps = -1
  evaluate(
      hparams, FLAGS.policy_dir, FLAGS.eval_metrics_dir, FLAGS.agent,
      FLAGS.eval_with_learner
  )


if __name__ == "__main__":
  tf.app.run()
