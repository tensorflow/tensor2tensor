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
from tensor2tensor.utils import misc_utils
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

  # TODO(afrozm): Decouple env_fn from hparams and return both, is there
  # even a need to return hparams? Just return the env_fn?
  hparams.add_hparam("env_fn", rl.make_real_env_fn(env))
  return hparams


def train(hparams, output_dir, report_fn=None):
  """Train."""
  hparams = initialize_env_specs(hparams)

  tf.logging.vlog(1, "HParams in trainer_model_free.train : %s",
                  misc_utils.pprint_hparams(hparams))

  tf.logging.vlog(1, "Using hparams.base_algo: %s", hparams.base_algo)
  learner = rl_utils.LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, output_dir, output_dir, total_num_epochs=1
  )

  policy_hparams = trainer_lib.create_hparams(hparams.base_algo_params)

  rl_utils.update_hparams_from_hparams(
      policy_hparams, hparams, hparams.base_algo + "_"
  )

  tf.logging.vlog(1, "Policy HParams : %s",
                  misc_utils.pprint_hparams(policy_hparams))

  total_steps = policy_hparams.epochs_num
  tf.logging.vlog(2, "total_steps: %d", total_steps)

  eval_every_epochs = policy_hparams.eval_every_epochs
  tf.logging.vlog(2, "eval_every_epochs: %d", eval_every_epochs)

  if eval_every_epochs == 0:
    eval_every_epochs = total_steps
  policy_hparams.eval_every_epochs = 0

  steps = list(range(eval_every_epochs, total_steps+1, eval_every_epochs))
  if not steps or steps[-1] < eval_every_epochs:
    steps.append(eval_every_epochs)

  tf.logging.vlog(1, "steps: [%s]", ",".join([str(s) for s in steps]))

  metric_name = rl_utils.get_metric_name(
      sampling_temp=hparams.eval_sampling_temps[0],
      max_num_noops=hparams.eval_max_num_noops,
      clipped=False
  )

  tf.logging.vlog(1, "metric_name: %s", metric_name)

  for i, step in enumerate(steps):
    tf.logging.info("Starting training iteration [%d] for [%d] steps.", i, step)

    policy_hparams.epochs_num = step
    learner.train(hparams.env_fn,
                  policy_hparams,
                  simulated=False,
                  save_continuously=True,
                  epoch=0)

    tf.logging.info("Ended training iteration [%d] for [%d] steps.", i, step)

    eval_metrics = rl_utils.evaluate_all_configs(hparams, output_dir)

    tf.logging.info(
        "Agent eval metrics:\n{}".format(pprint.pformat(eval_metrics)))

    if report_fn:
      report_fn(eval_metrics[metric_name], step)


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)

  tf.logging.log("Starting model free training.")
  train(hparams, FLAGS.output_dir)
  tf.logging.log("Ended model free training.")


if __name__ == "__main__":
  tf.app.run()
