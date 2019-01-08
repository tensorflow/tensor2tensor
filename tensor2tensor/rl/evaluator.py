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


def evaluate(hparams, policy_dir, eval_metrics_dir, report_fn=None,
             report_metric=None):
  """Evaluate."""
  if report_fn:
    assert report_metric is not None

  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_dir)
  eval_metrics = rl_utils.evaluate_all_configs(hparams, policy_dir)
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
  evaluate(hparams, FLAGS.policy_dir, FLAGS.eval_metrics_dir)


if __name__ == "__main__":
  tf.app.run()
