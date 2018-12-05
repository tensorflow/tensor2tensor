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

r"""Training of RL agent with PPO algorithm.

Example invocation:

python -m tensor2tensor.rl.trainer_model_free \
    --output_dir=$HOME/t2t/rl_v1 \
    --hparams_set=pong_model_free \
    --loop_hparams='batch_size=15'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

from tensor2tensor.models.research import rl
from tensor2tensor.rl import rl_utils
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erring. Apologies for the ugliness.
try:
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
except:  # pylint: disable=bare-except
  pass


def initialize_env_specs(hparams):
  """Initializes env_specs using T2TGymEnvs."""
  env = rl_utils.setup_env(hparams, hparams.batch_size,
                           hparams.eval_max_num_noops)
  env.start_new_epoch(0)
  hparams.add_hparam("env_fn", rl.make_real_env_fn(env))
  return hparams


def train(hparams, output_dir, report_fn=None):
  """Train."""
  hparams = initialize_env_specs(hparams)
  learner = rl_utils.LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, FLAGS.output_dir, output_dir
  )
  policy_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  rl_utils.update_hparams_from_hparams(
      policy_hparams, hparams, hparams.base_algo + "_"
  )
  total_steps = policy_hparams.epochs_num
  eval_every_epochs = policy_hparams.eval_every_epochs
  if eval_every_epochs == 0:
    eval_every_epochs = total_steps
  policy_hparams.eval_every_epochs = 0

  steps = list(range(eval_every_epochs, total_steps+1, eval_every_epochs))
  if not steps or steps[-1] < eval_every_epochs:
    steps.append(eval_every_epochs)
  metric_name = rl_utils.get_metric_name(
      stochastic=True, max_num_noops=hparams.eval_max_num_noops,
      clipped=False
  )
  for step in steps:
    policy_hparams.epochs_num = step
    learner.train(
        hparams.env_fn, policy_hparams, simulated=False, save_continuously=True,
        epoch=0
    )
    eval_metrics = rl_utils.evaluate_all_configs(hparams, output_dir)
    tf.logging.info("Agent eval metrics:\n{}".format(
        pprint.pformat(eval_metrics)))
    if report_fn:
      report_fn(eval_metrics[metric_name], step)


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)
  train(hparams, FLAGS.output_dir)


if __name__ == "__main__":
  tf.app.run()
