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

r"""Training of RL agent with PPO algorithm.

Example invocation:

python -m tensor2tensor.rl.trainer_model_free \
    --output_dir=$HOME/t2t/rl_v1 \
    --hparams_set=pong_model_free \
    --hparams='batch_size=15'

Example invocation with EnvProblem interface:

python -m tensor2tensor.rl.trainer_model_free \
  --env_problem_name=tic_tac_toe_env_problem \
  --hparams_set=rlmf_tictactoe \
  --output_dir=${OUTPUTDIR} \
  --log_dir=${LOGDIR} \
  --alsologtostderr \
  --vmodule=*/tensor2tensor/*=2 \
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

from tensor2tensor.models.research import rl
from tensor2tensor.rl import rl_utils
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import misc_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow.compat.v1 as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("env_problem_name", "",
                    "Which registered env_problem do we want?")

# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erring. Apologies for the ugliness.
try:
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
except:  # pylint: disable=bare-except
  pass


def initialize_env_specs(hparams, env_problem_name):
  """Initializes env_specs using the appropriate env."""
  if env_problem_name:
    env = registry.env_problem(env_problem_name, batch_size=hparams.batch_size)
  else:
    env = rl_utils.setup_env(hparams, hparams.batch_size,
                             hparams.eval_max_num_noops,
                             hparams.rl_env_max_episode_steps,
                             env_name=hparams.rl_env_name)
    env.start_new_epoch(0)

  return rl.make_real_env_fn(env)


step = 0


def train(hparams, output_dir, env_problem_name, report_fn=None):
  """Train."""
  env_fn = initialize_env_specs(hparams, env_problem_name)

  tf.logging.vlog(1, "HParams in trainer_model_free.train : %s",
                  misc_utils.pprint_hparams(hparams))
  tf.logging.vlog(1, "Using hparams.base_algo: %s", hparams.base_algo)
  learner = rl_utils.LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, output_dir, output_dir, total_num_epochs=1,
      distributional_size=hparams.get("distributional_size", 1),
      distributional_subscale=hparams.get("distributional_subscale", 0.04),
      distributional_threshold=hparams.get("distributional_threshold", 0.0),
  )

  policy_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  rl_utils.update_hparams_from_hparams(
      policy_hparams, hparams, hparams.base_algo + "_"
  )

  tf.logging.vlog(1, "Policy HParams : %s",
                  misc_utils.pprint_hparams(policy_hparams))

  # TODO(konradczechowski): remove base_algo dependance, when evaluation method
  # will be decided
  if hparams.base_algo == "ppo":
    total_steps = policy_hparams.epochs_num
    tf.logging.vlog(2, "total_steps: %d", total_steps)

    eval_every_epochs = policy_hparams.eval_every_epochs
    tf.logging.vlog(2, "eval_every_epochs: %d", eval_every_epochs)

    if eval_every_epochs == 0:
      eval_every_epochs = total_steps
    policy_hparams.eval_every_epochs = 0

    metric_name = rl_utils.get_metric_name(
        sampling_temp=hparams.eval_sampling_temps[0],
        max_num_noops=hparams.eval_max_num_noops,
        clipped=False
    )

    tf.logging.vlog(1, "metric_name: %s", metric_name)

    eval_metrics_dir = os.path.join(output_dir, "eval_metrics")
    eval_metrics_dir = os.path.expanduser(eval_metrics_dir)
    tf.gfile.MakeDirs(eval_metrics_dir)
    eval_metrics_writer = tf.summary.FileWriter(eval_metrics_dir)

    def evaluate_on_new_model(model_dir_path):
      global step
      eval_metrics = rl_utils.evaluate_all_configs(hparams, model_dir_path)
      tf.logging.info(
          "Agent eval metrics:\n{}".format(pprint.pformat(eval_metrics)))
      rl_utils.summarize_metrics(eval_metrics_writer, eval_metrics, step)
      if report_fn:
        report_fn(eval_metrics[metric_name], step)
      step += 1

    policy_hparams.epochs_num = total_steps
    policy_hparams.save_models_every_epochs = eval_every_epochs
  else:
    def evaluate_on_new_model(model_dir_path):
      del model_dir_path
      raise NotImplementedError(
          "This function is currently implemented only for ppo")

  learner.train(env_fn,
                policy_hparams,
                simulated=False,
                save_continuously=True,
                epoch=0,
                model_save_fn=evaluate_on_new_model)


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)

  tf.logging.info("Starting model free training.")
  train(hparams, FLAGS.output_dir, FLAGS.env_problem_name)
  tf.logging.info("Ended model free training.")


if __name__ == "__main__":
  tf.app.run()
